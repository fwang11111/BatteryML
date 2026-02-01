# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch

from batteryml.builders import FEATURE_EXTRACTORS
from batteryml.data.battery_data import BatteryData, CycleData
from batteryml.feature.base import BaseFeatureExtractor


def _infer_discharge_sign(current: np.ndarray) -> str:
    nonzero = current[np.abs(current) > 1e-12]
    if len(nonzero) == 0:
        return 'unknown'
    return 'negative' if np.nanmedian(nonzero) < 0 else 'positive'


def _pick_cycle(cell_data: BatteryData,
                target_cycle_number: int,
                *,
                allow_nearest: bool,
                max_abs_diff: int) -> CycleData | None:
    cycles = list(cell_data.cycle_data or [])
    if not cycles:
        return None

    exact = [c for c in cycles if int(c.cycle_number) == int(target_cycle_number)]
    if exact:
        return exact[0]

    if not allow_nearest:
        return None

    # Find the nearest cycle_number
    best = cycles[0]
    best_diff = abs(int(best.cycle_number) - int(target_cycle_number))
    for c in cycles[1:]:
        diff = abs(int(c.cycle_number) - int(target_cycle_number))
        if diff < best_diff:
            best = c
            best_diff = diff

    return best if best_diff <= max_abs_diff else None


def _interp_q_on_voltage(voltage_v: np.ndarray,
                         q: np.ndarray,
                         grid_v: np.ndarray) -> np.ndarray:
    """Interpolate q(v) onto grid_v.

    Returns NaN for grid points outside the source voltage range.
    """
    voltage_v = np.asarray(voltage_v, dtype=float)
    q = np.asarray(q, dtype=float)

    mask = (~np.isnan(voltage_v)) & (~np.isnan(q))
    voltage_v, q = voltage_v[mask], q[mask]
    if len(voltage_v) < 2:
        return np.full_like(grid_v, np.nan, dtype=float)

    order = np.argsort(voltage_v)
    voltage_v = voltage_v[order]
    q = q[order]

    # Merge duplicate voltages by averaging q
    v_unique, inv = np.unique(voltage_v, return_inverse=True)
    q_sum = np.bincount(inv, weights=q)
    counts = np.bincount(inv)
    q_unique = q_sum / np.clip(counts, 1, None)

    out = np.interp(grid_v, v_unique, q_unique)
    out[(grid_v < v_unique[0]) | (grid_v > v_unique[-1])] = np.nan
    return out


@FEATURE_EXTRACTORS.register()
class NormalizedVoltageCapacityMatrixFeatureExtractor(BaseFeatureExtractor):
    """Capacity-normalized voltage-capacity matrix features for sparse RPT.

    Goal: make features more transferable across cell sizes (e.g. 3Ah vs 100Ah)
    by normalizing discharge capacity to the baseline-cycle capacity.

    For each cell:
    - pick a baseline cycle (default 0)
    - for each target cycle in `cycles_to_compare`, compute q_norm(v) where
      q_norm = Qd / Q0_end, interpolated on a voltage grid
    - output rows are diff curves: q_norm_target(v) - q_norm_base(v)
    - optionally append a scalar SOH (Q_end/Q0_end) as the last column
    """

    def __init__(
        self,
        interp_dim: int = 256,
        baseline_cycle_number: int = 0,
        cycles_to_compare: List[int] | int | None = None,
        min_voltage_in_V: float | None = None,
        max_voltage_in_V: float | None = None,
        discharge_current_sign: str = 'auto',
        allow_nearest_cycle: bool = True,
        max_cycle_number_gap: int = 2,
        append_soh_scalar: bool = True,
    ):
        self.interp_dim = int(interp_dim)
        self.baseline_cycle_number = int(baseline_cycle_number)
        if cycles_to_compare is None:
            cycles_to_compare = [100, 200]
        if isinstance(cycles_to_compare, int):
            cycles_to_compare = [cycles_to_compare]
        self.cycles_to_compare = [int(x) for x in cycles_to_compare]

        self.min_voltage_in_V = min_voltage_in_V
        self.max_voltage_in_V = max_voltage_in_V

        self.discharge_current_sign = discharge_current_sign
        self.allow_nearest_cycle = bool(allow_nearest_cycle)
        self.max_cycle_number_gap = int(max_cycle_number_gap)
        self.append_soh_scalar = bool(append_soh_scalar)

    def _get_voltage_range(self, cell_data: BatteryData) -> tuple[float, float]:
        if self.min_voltage_in_V is not None and self.max_voltage_in_V is not None:
            return float(self.min_voltage_in_V), float(self.max_voltage_in_V)
        min_v = getattr(cell_data, 'min_voltage_limit_in_V', None)
        max_v = getattr(cell_data, 'max_voltage_limit_in_V', None)
        if min_v is not None and max_v is not None:
            return float(min_v), float(max_v)
        raise ValueError(
            'Voltage range not specified. Provide min_voltage_in_V/max_voltage_in_V '
            'or ensure BatteryData has min/max voltage limits.'
        )

    def _discharge_mask(self, cycle: CycleData) -> np.ndarray:
        I = np.asarray(cycle.current_in_A if cycle.current_in_A is not None else [], dtype=float)
        if len(I) == 0:
            return np.ones(0, dtype=bool)

        sign = self.discharge_current_sign
        if sign == 'auto':
            sign = _infer_discharge_sign(I)
        if sign == 'negative':
            return I < -1e-6
        if sign == 'positive':
            return I > 1e-6
        # Fallback: treat all points as discharge
        return np.ones_like(I, dtype=bool)

    def _qd_curve(self, cycle: CycleData) -> tuple[np.ndarray, np.ndarray]:
        V = np.asarray(cycle.voltage_in_V if cycle.voltage_in_V is not None else [], dtype=float)
        Qd = np.asarray(
            cycle.discharge_capacity_in_Ah if cycle.discharge_capacity_in_Ah is not None else [],
            dtype=float,
        )
        if len(V) == 0 or len(Qd) == 0 or len(V) != len(Qd):
            return np.array([]), np.array([])

        # Prefer a Qd-increasing mask (robust across sign conventions)
        dq = np.diff(Qd, prepend=Qd[0])
        inc_mask = dq > 1e-9
        if inc_mask.sum() >= 2:
            return V[inc_mask], Qd[inc_mask]

        # Fallback to current-sign discharge mask
        if cycle.current_in_A is not None:
            mask = self._discharge_mask(cycle)
            if len(mask) == len(V) and mask.any():
                return V[mask], Qd[mask]

        # Last resort: return full arrays
        return V, Qd

    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        base_cycle = _pick_cycle(
            cell_data,
            self.baseline_cycle_number,
            allow_nearest=self.allow_nearest_cycle,
            max_abs_diff=self.max_cycle_number_gap,
        )
        if base_cycle is None:
            raise ValueError(
                f'Baseline cycle not found: {self.baseline_cycle_number} for {cell_data.cell_id}')

        base_V, base_Qd = self._qd_curve(base_cycle)
        if len(base_Qd) == 0:
            raise ValueError(f'Empty baseline discharge curve: {cell_data.cell_id}')
        q0_end = float(np.nanmax(base_Qd))
        if not np.isfinite(q0_end) or q0_end <= 0:
            raise ValueError(f'Invalid baseline capacity (Q0_end): {cell_data.cell_id}')

        min_v, max_v = self._get_voltage_range(cell_data)
        grid_v = np.linspace(min_v, max_v, self.interp_dim)

        base_qnorm = base_Qd / q0_end
        base_qnorm_lin = _interp_q_on_voltage(base_V, base_qnorm, grid_v)

        rows = []
        for cyc in self.cycles_to_compare:
            c = _pick_cycle(
                cell_data,
                cyc,
                allow_nearest=self.allow_nearest_cycle,
                max_abs_diff=self.max_cycle_number_gap,
            )
            if c is None:
                raise ValueError(
                    f'Required cycle not found: {cyc} for {cell_data.cell_id}')

            V, Qd = self._qd_curve(c)
            if len(Qd) == 0:
                raise ValueError(
                    f'Empty discharge curve at cycle {cyc} for {cell_data.cell_id}')
            qnorm = Qd / q0_end
            qnorm_lin = _interp_q_on_voltage(V, qnorm, grid_v)
            diff = qnorm_lin - base_qnorm_lin

            if self.append_soh_scalar:
                soh = float(np.nanmax(Qd) / q0_end)
                diff = np.concatenate([diff, [soh]])
            rows.append(torch.from_numpy(diff.astype(np.float32)))

        feat = torch.stack(rows)
        feat[torch.isnan(feat) | torch.isinf(feat)] = 0.0
        return feat
