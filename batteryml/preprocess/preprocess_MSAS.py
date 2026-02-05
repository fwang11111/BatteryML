# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import yaml

from batteryml import BatteryData, CycleData
from batteryml.builders import PREPROCESSORS
from batteryml.preprocess.base import BasePreprocessor


@dataclass(frozen=True)
class _MSASColumns:
    run_time_s: str = 'run_time'
    voltage_v: str = 'c_vol'
    current_a: str = 'c_cur'
    surf_temp_c: str = 'c_surf_temp'
    amb_temp_c: str = 'amb_temp'
    step_type: str = 'step_type'


_FILENAME_RE = re.compile(
    r'^(?P<serial>TP_[kz]\d+_\d+)_(?P<order>\d+)_(?P<tag>.+)\.csv$',
    flags=re.IGNORECASE,
)


def _load_yaml_config(config_path) -> dict[str, Any]:
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


def _safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)


def _parse_time_seconds(series: pd.Series) -> np.ndarray:
    if series.dtype.kind in {'i', 'u', 'f'}:
        return pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)

    numeric = pd.to_numeric(series, errors='coerce')
    numeric_ratio = float(np.isfinite(numeric).mean()) if len(numeric) else 0.0
    if numeric_ratio >= 0.8:
        return numeric.to_numpy(dtype=float)

    timed = pd.to_timedelta(series, errors='coerce')
    seconds = timed.dt.total_seconds()
    return pd.to_numeric(seconds, errors='coerce').to_numpy(dtype=float)


def _integrate_capacity_ah(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    """Integrate discharge capacity (Ah) from time/current.

    Expects `time_s` in seconds and `current_a` in amps.
    Discharge is assumed to be negative current; we integrate -I over time.
    """

    if time_s.size == 0 or current_a.size == 0:
        return np.array([], dtype=float)

    mask = np.isfinite(time_s) & np.isfinite(current_a)
    time_s = time_s[mask]
    current_a = current_a[mask]
    if time_s.size < 2:
        return np.array([], dtype=float)

    order = np.argsort(time_s)
    time_s = time_s[order]
    current_a = current_a[order]

    dt = np.diff(time_s, prepend=time_s[0])
    dt[0] = 0.0

    discharge_a = np.where(current_a < 0, -current_a, 0.0)
    dq = discharge_a * dt / 3600.0
    return np.cumsum(dq)


def _parse_tag_temperature_c(tag: str) -> float | None:
    m = re.search(r'T(\d+)', tag)
    if m is None:
        return None
    return float(m.group(1))


@PREPROCESSORS.register()
class MSASPreprocessor(BasePreprocessor):
    """Preprocess Multi-Stage Aging Study dataset into `BatteryData`.

    Input format (per docs):
    - `experiments_meta.csv`
    - `Stage_1/*.zip` and/or `Stage_2/*.zip`

    Each cell zip contains multiple CSVs named like:
    - `TP_z01_01_02_ZYK.csv` (cycling segment)
    - `TP_z01_01_03_CU.csv` (check-up)
    - `TP_z01_01_01_ET_T23.csv` (check-up at temperature)

    Output:
    - One `BatteryData` per zip.
    - Each check-up CSV becomes one `CycleData` containing the *capacity discharge*
      segment (step_type==22). We also attach metadata such as:
      `msas_tag`, `msas_order`, `msas_stage`, `msas_type`, `msas_lab`, `msas_efc`.
    - Cycling segments (ZYK) are not emitted as cycles; they only contribute to the
      running EFC counter used to annotate subsequent check-ups.

    Notes:
    - This preprocessor prioritizes robustness and portability. It does not try to
      reproduce every protocol detail (HPPC/pOCV parsing) yet.
    - EFC is estimated from ZYK discharge throughput (step_type==42) and
      `nominal_capacity_in_Ah`.
    """

    def process(self, parentdir, config_path=None, **kwargs) -> tuple[int, int]:
        raw_dir = Path(parentdir)
        if not raw_dir.exists():
            raise FileNotFoundError(f'Raw directory not found: {raw_dir}')

        cfg = _load_yaml_config(config_path)

        meta_path = raw_dir / 'experiments_meta.csv'
        if not meta_path.exists():
            raise FileNotFoundError(
                'Expected MSAS metadata file not found: '
                f'{meta_path}. Set raw_dir to the dataset root.')

        cols = _MSASColumns()
        nominal_capacity_in_Ah = float(
            kwargs.get('nominal_capacity_in_Ah')
            or kwargs.get('nominal_capacity_ah')
            or cfg.get('nominal_capacity_in_Ah')
            or cfg.get('nominal_capacity_ah')
            or 4.9
        )
        cell_id_prefix = str(kwargs.get('cell_id_prefix') or cfg.get('cell_id_prefix') or 'MSAS')
        include_tags = kwargs.get('include_tags') or cfg.get('include_tags')  # None or list[str]
        if include_tags is not None and not isinstance(include_tags, list):
            raise ValueError('include_tags must be a list of strings or null')

        include_serials = kwargs.get('include_serials') or cfg.get('include_serials')
        if include_serials is not None and not isinstance(include_serials, list):
            raise ValueError('include_serials must be a list of strings or null')

        max_cells = kwargs.get('max_cells') or cfg.get('max_cells')
        if max_cells is not None:
            max_cells = int(max_cells)

        meta = pd.read_csv(meta_path)
        if 'serial' not in meta.columns:
            raise ValueError(f'Missing required column "serial" in {meta_path}')

        # Index metadata by serial for quick lookup.
        meta_by_serial = meta.set_index('serial', drop=False)

        zip_files: list[Path] = []
        for stage_dir in (raw_dir / 'Stage_1', raw_dir / 'Stage_2'):
            if stage_dir.exists():
                zip_files.extend(sorted(stage_dir.glob('TP_*.zip')))

        if len(zip_files) == 0:
            raise FileNotFoundError(
                f'No TP_*.zip found under {raw_dir / "Stage_1"} or {raw_dir / "Stage_2"}.')

        if include_serials is not None:
            include_serials = set(str(x) for x in include_serials)
            zip_files = [p for p in zip_files if p.stem in include_serials]

        if max_cells is not None and max_cells > 0:
            zip_files = zip_files[:max_cells]

        total_cells = len(zip_files)
        if total_cells == 0:
            raise ValueError('No MSAS cells match include_serials/max_cells filters.')

        processed = 0
        skipped = 0
        iterator = zip_files
        if not self.silent:
            from tqdm import tqdm

            iterator = tqdm(zip_files, desc='Processing MSAS cell zips')

        if not self.silent:
            print(f'Total MSAS cells to process: {total_cells}')

        for idx, zip_path in enumerate(iterator, start=1):
            serial = zip_path.stem

            if not self.silent:
                print(f'Processing {serial} ({idx}/{total_cells})')

            stage_from_path = None
            parent_name = zip_path.parent.name.lower()
            if parent_name == 'stage_1':
                stage_from_path = 1
            elif parent_name == 'stage_2':
                stage_from_path = 2

            stage_suffix = f'S{stage_from_path}' if stage_from_path is not None else 'S0'
            out_cell_id = f'{cell_id_prefix}_{stage_suffix}_{serial}'

            if self.check_processed_file(out_cell_id):
                skipped += 1
                continue

            if serial not in meta_by_serial.index:
                logging.warning('No metadata row for serial %s (zip=%s).', serial, zip_path)
                meta_row = None
            else:
                meta_row = meta_by_serial.loc[serial]

            if meta_row is not None and isinstance(meta_row, pd.DataFrame):
                if stage_from_path is not None and 'stage' in meta_row:
                    filtered = meta_row[meta_row['stage'] == stage_from_path]
                    if not filtered.empty:
                        meta_row = filtered.iloc[0]
                    else:
                        meta_row = meta_row.iloc[0]
                else:
                    meta_row = meta_row.iloc[0]

            if stage_from_path is not None:
                stage = stage_from_path
            else:
                stage = None
            if meta_row is not None and 'stage' in meta_row:
                try:
                    stage = int(meta_row['stage'])
                except (TypeError, ValueError):
                    pass
            lab = str(meta_row['lab']) if meta_row is not None and 'lab' in meta_row else None
            aging_type = str(meta_row['type']) if meta_row is not None and 'type' in meta_row else None

            efc = 0.0
            cycles: list[CycleData] = []

            with zipfile.ZipFile(zip_path, 'r') as zf:
                csv_names = [
                    name for name in zf.namelist()
                    if name.lower().endswith('.csv') and not name.endswith('/')
                ]

                sessions = []
                for name in csv_names:
                    base = Path(name).name
                    m = _FILENAME_RE.match(base)
                    if m is None:
                        continue
                    if m.group('serial') != serial:
                        continue
                    sessions.append((
                        int(m.group('order')),
                        m.group('tag'),
                        name,
                    ))

                sessions.sort(key=lambda x: x[0])

                for order_idx, tag, member_name in sessions:
                    tag_norm = str(tag)
                    is_zyk = tag_norm.upper().startswith('ZYK')
                    if not is_zyk:
                        if include_tags is not None and tag_norm not in set(include_tags):
                            continue

                    with zf.open(member_name, 'r') as fp:
                        df = pd.read_csv(fp)

                    missing_cols = [
                        c for c in (cols.run_time_s, cols.voltage_v, cols.current_a, cols.step_type)
                        if c not in df.columns
                    ]
                    if missing_cols:
                        logging.warning(
                            'Skip %s inside %s: missing columns %s',
                            member_name, zip_path.name, missing_cols)
                        continue

                    step = _safe_numeric(df[cols.step_type])
                    time_s = _parse_time_seconds(df[cols.run_time_s])
                    current_a = _safe_numeric(df[cols.current_a])
                    voltage_v = _safe_numeric(df[cols.voltage_v])
                    surf_temp_c = _safe_numeric(df[cols.surf_temp_c]) if cols.surf_temp_c in df.columns else None

                    if is_zyk:
                        # Cycling segment: accumulate discharge throughput to estimate EFC.
                        z_mask = step == 42
                        q_ah = _integrate_capacity_ah(time_s[z_mask], current_a[z_mask])
                        if q_ah.size > 0:
                            efc += float(q_ah[-1] / nominal_capacity_in_Ah)
                        continue

                    # Check-up: extract capacity discharge segment.
                    d_mask = step == 22
                    if not np.any(d_mask):
                        # Fallback: some check-ups include pOCV discharge as step_type 32.
                        d_mask = step == 32
                    if not np.any(d_mask):
                        logging.warning(
                            'Skip %s inside %s: no discharge segment (step_type 22/32).',
                            member_name, zip_path.name)
                        continue

                    t_seg = time_s[d_mask]
                    i_seg = current_a[d_mask]
                    v_seg = voltage_v[d_mask]
                    q_seg = _integrate_capacity_ah(t_seg, i_seg)
                    if q_seg.size == 0:
                        logging.warning(
                            'Skip %s inside %s: discharge segment too short.',
                            member_name, zip_path.name)
                        continue

                    temp_seg = surf_temp_c[d_mask] if surf_temp_c is not None else None

                    amb_tag_c = _parse_tag_temperature_c(tag_norm)
                    if temp_seg is None:
                        cycle = CycleData(
                            cycle_number=int(order_idx),
                            voltage_in_V=v_seg.tolist(),
                            current_in_A=i_seg.tolist(),
                            time_in_s=t_seg.tolist(),
                            discharge_capacity_in_Ah=q_seg.tolist(),
                        )
                    else:
                        cycle = CycleData(
                            cycle_number=int(order_idx),
                            voltage_in_V=v_seg.tolist(),
                            current_in_A=i_seg.tolist(),
                            time_in_s=t_seg.tolist(),
                            discharge_capacity_in_Ah=q_seg.tolist(),
                            temperature_in_C=temp_seg.tolist(),
                        )
                    cycle.additional_data.update({
                        'msas_tag': tag_norm,
                        'msas_order': int(order_idx),
                        'msas_stage': stage,
                        'msas_type': aging_type,
                        'msas_lab': lab,
                        'msas_efc': float(efc),
                        'msas_tag_temperature_c': amb_tag_c,
                    })
                    cycles.append(cycle)

            if len(cycles) == 0:
                logging.warning('No cycles extracted from %s; skipping output.', zip_path)
                skipped += 1
                continue

            battery = BatteryData(
                cell_id=out_cell_id,
                cycle_data=cycles,
                form_factor='cylindrical_21700',
                anode_material='graphite',
                cathode_material='NMC',
                nominal_capacity_in_Ah=nominal_capacity_in_Ah,
                reference='Multi-Stage Aging Study (Stroebl & Palm, 2024)',
                description=(
                    'Check-up discharge curves extracted from CU/exCU/ET/AT sessions; '
                    'EFC estimated from ZYK discharge throughput.'
                ),
            )
            self.dump_single_file(battery)
            processed += 1

        return processed, skipped
