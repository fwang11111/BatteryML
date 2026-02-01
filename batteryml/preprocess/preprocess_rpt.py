# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from batteryml import BatteryData, CycleData
from batteryml.builders import PREPROCESSORS
from batteryml.preprocess.base import BasePreprocessor


@dataclass(frozen=True)
class _RPTColumns:
    """Canonical column names for RPT discharge files."""

    time_s: str = 'time'
    current_a: str = 'I'
    voltage_v: str = 'V'
    temperature_c: str = 'T'
    discharge_capacity_ah: str = 'Q'


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    raise ValueError(f'Unsupported file type: {path}')


def _safe_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f'Missing required column: {col}')
    return pd.to_numeric(df[col], errors='coerce')


def _infer_discharge_current_sign(current: np.ndarray) -> str:
    nonzero = current[np.abs(current) > 1e-12]
    if len(nonzero) == 0:
        return 'unknown'
    return 'negative' if np.nanmedian(nonzero) < 0 else 'positive'


def _parse_cycle_number_from_filename(path: Path, regex: str) -> int:
    m = re.search(regex, path.stem)
    if m is None:
        raise ValueError(
            f'Cannot parse cycle number from {path.name} using regex: {regex}')
    return int(m.group(1))


def _parse_cell_id_from_filename(path: Path, regex: str) -> str:
    m = re.search(regex, path.name)
    if m is None:
        raise ValueError(
            f'Cannot parse cell id from {path.name} using regex: {regex}')
    return str(m.group(1))


def _load_yaml_config(config_path) -> dict:
    if config_path is None or str(config_path) == 'None':
        return {}
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Invalid YAML config (expected dict): {config_path}')
    return data


def _iter_rpt_files(raw_dir: Path) -> Iterable[Path]:
    exts = {'.csv', '.xlsx', '.xls'}
    for p in raw_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


@PREPROCESSORS.register()
class RPTPreprocessor(BasePreprocessor):
    """Preprocess sparse discharge-only RPT files into `BatteryData`.

    Assumes multiple files per cell, each file corresponds to one RPT checkpoint
    (cycle 0/100/200/...). By default:

    - `cell_id` is the *parent directory name* of each file.
    - `cycle_number` is parsed from filename using `cycle_number_regex`.

    The preprocessor expects columns for time/current/voltage/temperature/Q.
    These can be remapped and scaled via a YAML config passed to the CLI.
    """

    def process(self, parentdir, config_path=None, **kwargs) -> tuple[int, int]:
        raw_dir = Path(parentdir)
        if not raw_dir.exists():
            raise FileNotFoundError(f'Raw directory not found: {raw_dir}')

        cfg = _load_yaml_config(config_path)

        # Defaults
        cycle_number_regex = (
            kwargs.get('cycle_number_regex')
            or cfg.get('cycle_number_regex')
            or r'(?:cycle|c|cyc|rpt)[-_]?(\d+)'
        )
        discharge_current_sign = (
            kwargs.get('discharge_current_sign')
            or cfg.get('discharge_current_sign')
            or 'auto'
        )

        cell_id_from = (
            kwargs.get('cell_id_from')
            or cfg.get('cell_id_from')
            or 'parent_dir'
        )
        cell_id_regex = (
            kwargs.get('cell_id_regex')
            or cfg.get('cell_id_regex')
            or r'^([^_]+)'
        )

        # Column mapping + scaling (kept intentionally simple; override via kwargs)
        cols_cfg = cfg.get('columns', {}) or {}
        scales_cfg = cfg.get('scales', {}) or {}
        cols = _RPTColumns(
            time_s=kwargs.get('time_col') or cols_cfg.get('time_s') or 'time',
            current_a=kwargs.get('current_col') or cols_cfg.get('current_a') or 'I',
            voltage_v=kwargs.get('voltage_col') or cols_cfg.get('voltage_v') or 'V',
            temperature_c=kwargs.get('temperature_col') or cols_cfg.get('temperature_c') or 'T',
            discharge_capacity_ah=kwargs.get('capacity_col') or cols_cfg.get('discharge_capacity_ah') or 'Q',
        )
        time_scale = float(kwargs.get('time_scale') or scales_cfg.get('time_s') or 1.0)
        current_scale = float(kwargs.get('current_scale') or scales_cfg.get('current_a') or 1.0)
        voltage_scale = float(kwargs.get('voltage_scale') or scales_cfg.get('voltage_v') or 1.0)
        temperature_scale = float(kwargs.get('temperature_scale') or scales_cfg.get('temperature_c') or 1.0)
        capacity_scale = float(kwargs.get('capacity_scale') or scales_cfg.get('discharge_capacity_ah') or 1.0)

        # Cell-level metadata (can be overridden per-file later if needed)
        cell_id_prefix = kwargs.get('cell_id_prefix') or cfg.get('cell_id_prefix') or 'RPT'
        form_factor = kwargs.get('form_factor') or cfg.get('form_factor') or 'pouch'
        anode_material = kwargs.get('anode_material') or cfg.get('anode_material') or 'graphite'
        cathode_material = kwargs.get('cathode_material') or cfg.get('cathode_material') or 'NMC'
        nominal_capacity_in_Ah = kwargs.get('nominal_capacity_in_Ah') or cfg.get('nominal_capacity_in_Ah')
        min_voltage_limit_in_V = kwargs.get('min_voltage_limit_in_V') or cfg.get('min_voltage_limit_in_V')
        max_voltage_limit_in_V = kwargs.get('max_voltage_limit_in_V') or cfg.get('max_voltage_limit_in_V')

        # Group files by cell_id (parent directory)
        by_cell: dict[str, list[Path]] = {}
        for f in _iter_rpt_files(raw_dir):
            if cell_id_from == 'parent_dir':
                cell_key = f.parent.name
            elif cell_id_from == 'filename_regex':
                cell_key = _parse_cell_id_from_filename(f, cell_id_regex)
            else:
                raise ValueError(
                    f'Unsupported cell_id_from={cell_id_from}. '
                    "Use 'parent_dir' or 'filename_regex'."
                )
            by_cell.setdefault(cell_key, []).append(f)

        process_batteries_num = 0
        skip_batteries_num = 0

        for cell_key, files in sorted(by_cell.items()):
            cell_id = f'{cell_id_prefix}_{cell_key}'
            if self.check_processed_file(cell_id):
                skip_batteries_num += 1
                continue

            cycles: list[CycleData] = []
            for f in sorted(files):
                cycle_number = _parse_cycle_number_from_filename(f, cycle_number_regex)
                df = _read_table(f)

                t = _safe_float_series(df, cols.time_s).to_numpy(dtype=float) * time_scale
                I = _safe_float_series(df, cols.current_a).to_numpy(dtype=float) * current_scale
                V = _safe_float_series(df, cols.voltage_v).to_numpy(dtype=float) * voltage_scale

                if cols.temperature_c in df.columns:
                    T = _safe_float_series(df, cols.temperature_c).to_numpy(dtype=float) * temperature_scale
                else:
                    T = np.full_like(t, np.nan, dtype=float)

                if cols.discharge_capacity_ah in df.columns:
                    Q = _safe_float_series(df, cols.discharge_capacity_ah).to_numpy(dtype=float) * capacity_scale
                else:
                    Q = None

                # Drop rows with missing time/current/voltage
                mask = (~np.isnan(t)) & (~np.isnan(I)) & (~np.isnan(V))
                t, I, V, T = t[mask], I[mask], V[mask], T[mask]
                if Q is not None:
                    Q = Q[mask]

                # Ensure time starts at 0 and monotonically increases
                order = np.argsort(t)
                t, I, V, T = t[order], I[order], V[order], T[order]
                if Q is not None:
                    Q = Q[order]
                t = t - t[0]

                # If Q is missing, integrate discharge current to compute capacity
                if Q is None:
                    sign = discharge_current_sign
                    if sign == 'auto':
                        sign = _infer_discharge_current_sign(I)
                    if sign not in {'positive', 'negative'}:
                        raise ValueError(
                            f'Cannot infer discharge current sign for {f}. '
                            'Provide capacity column or set discharge_current_sign.'
                        )
                    Q = np.zeros_like(I)
                    for i in range(1, len(I)):
                        dt = t[i] - t[i - 1]
                        if dt < 0:
                            dt = 0
                        if sign == 'negative':
                            Q[i] = Q[i - 1] + (-I[i]) * dt / 3600.0
                        else:
                            Q[i] = Q[i - 1] + (I[i]) * dt / 3600.0

                cycles.append(CycleData(
                    cycle_number=int(cycle_number),
                    voltage_in_V=V.tolist(),
                    current_in_A=I.tolist(),
                    temperature_in_C=T.tolist(),
                    discharge_capacity_in_Ah=Q.tolist(),
                    time_in_s=t.tolist(),
                ))

            if len(cycles) == 0:
                logging.warning(f'No RPT files found for cell {cell_id}.')
                continue

            cycles.sort(key=lambda c: c.cycle_number)

            battery_kwargs = {
                'cell_id': cell_id,
                'cycle_data': cycles,
                'form_factor': form_factor,
                'anode_material': anode_material,
                'cathode_material': cathode_material,
            }
            if nominal_capacity_in_Ah is not None:
                battery_kwargs['nominal_capacity_in_Ah'] = float(nominal_capacity_in_Ah)
            if min_voltage_limit_in_V is not None:
                battery_kwargs['min_voltage_limit_in_V'] = float(min_voltage_limit_in_V)
            if max_voltage_limit_in_V is not None:
                battery_kwargs['max_voltage_limit_in_V'] = float(max_voltage_limit_in_V)

            battery = BatteryData(**battery_kwargs)
            self.dump_single_file(battery)
            process_batteries_num += 1

        return process_batteries_num, skip_batteries_num
