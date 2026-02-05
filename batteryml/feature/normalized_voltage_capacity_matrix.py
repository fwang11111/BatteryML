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

    If `include_cv` is False, Q is computed by trimming the *tail* CV region:
    points are excluded only when current drops below a fraction of the CC median
    and the voltage is near the discharge Vmin.
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
        include_cv: bool = True,
        cc_current_fraction: float = 0.98,
        cv_voltage_window_in_V: float = 0.05,
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
        self.include_cv = bool(include_cv)
        self.cc_current_fraction = float(cc_current_fraction)
        if not (0.0 < self.cc_current_fraction <= 1.0):
            raise ValueError('cc_current_fraction must be in (0, 1].')
        self.cv_voltage_window_in_V = float(cv_voltage_window_in_V)
        if self.cv_voltage_window_in_V <= 0:
            raise ValueError('cv_voltage_window_in_V must be > 0.')

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

    def _discharge_direction(self, cycle: CycleData) -> float | None:
        I = np.asarray(cycle.current_in_A if cycle.current_in_A is not None else [], dtype=float)
        if len(I) == 0:
            return None
        sign = self.discharge_current_sign
        if sign == 'auto':
            sign = _infer_discharge_sign(I)
        if sign == 'negative':
            return -1.0
        if sign == 'positive':
            return 1.0
        return None

    def _cc_mask(self, cycle: CycleData) -> np.ndarray:
        V = np.asarray(cycle.voltage_in_V if cycle.voltage_in_V is not None else [], dtype=float)
        I = np.asarray(cycle.current_in_A if cycle.current_in_A is not None else [], dtype=float)
        T = np.asarray(cycle.time_in_s if cycle.time_in_s is not None else [], dtype=float)
        if len(V) == 0 or len(I) == 0 or len(T) == 0:
            return np.ones(0, dtype=bool)
        if len(V) != len(I) or len(V) != len(T):
            return np.ones(0, dtype=bool)

        valid = np.isfinite(V) & np.isfinite(I) & np.isfinite(T)
        if valid.sum() < 2:
            return np.zeros_like(I, dtype=bool)

        Vv, Iv, Tv = V[valid], I[valid], T[valid]
        direction = self._discharge_direction(cycle)
        if direction is None:
            return np.zeros_like(I, dtype=bool)

        discharge_mask = Iv < -1e-6 if direction < 0 else Iv > 1e-6
        abs_current = np.abs(Iv[discharge_mask])
        if abs_current.size < 2:
            return np.zeros_like(I, dtype=bool)

        median_abs = float(np.nanmedian(abs_current))
        if not np.isfinite(median_abs) or median_abs <= 0:
            return np.zeros_like(I, dtype=bool)

        v_min = float(np.nanmin(Vv[discharge_mask]))
        if not np.isfinite(v_min):
            return np.zeros_like(I, dtype=bool)

        low_current = np.abs(Iv) < self.cc_current_fraction * median_abs
        near_vmin = Vv <= (v_min + self.cv_voltage_window_in_V)
        tail_candidate = discharge_mask & low_current & near_vmin

        order = np.argsort(Tv)
        tail_candidate = tail_candidate[order]
        discharge_sorted = discharge_mask[order]
        tail_sorted = np.zeros_like(tail_candidate, dtype=bool)
        idx = np.where(discharge_sorted)[0]
        if idx.size > 0:
            i = idx[-1]
            while i >= 0 and tail_candidate[i]:
                tail_sorted[i] = True
                i -= 1

        tail_mask = np.zeros_like(discharge_mask, dtype=bool)
        tail_mask[order] = tail_sorted
        cc_mask_valid = discharge_mask & ~tail_mask

        cc_mask = np.zeros_like(I, dtype=bool)
        cc_mask[valid] = cc_mask_valid
        return cc_mask

    def _cc_qd_curve(self, cycle: CycleData) -> tuple[np.ndarray, np.ndarray]:
        V = np.asarray(cycle.voltage_in_V if cycle.voltage_in_V is not None else [], dtype=float)
        I = np.asarray(cycle.current_in_A if cycle.current_in_A is not None else [], dtype=float)
        T = np.asarray(cycle.time_in_s if cycle.time_in_s is not None else [], dtype=float)

        if len(V) == 0 or len(I) == 0 or len(T) == 0:
            return np.array([]), np.array([])
        if len(V) != len(I) or len(V) != len(T):
            return np.array([]), np.array([])

        valid = np.isfinite(V) & np.isfinite(I) & np.isfinite(T)
        V, I, T = V[valid], I[valid], T[valid]
        if len(V) < 2:
            return np.array([]), np.array([])

        cc_mask = self._cc_mask(cycle)
        if len(cc_mask) != len(V):
            cc_mask = cc_mask[valid]
        if cc_mask.sum() < 2:
            return np.array([]), np.array([])

        direction = self._discharge_direction(cycle)
        if direction is None:
            return np.array([]), np.array([])

        order = np.argsort(T)
        V, I, T, cc_mask = V[order], I[order], T[order], cc_mask[order]
        dt = np.diff(T, prepend=T[0])
        dt[0] = 0.0

        interval_mask = cc_mask[:-1] & cc_mask[1:]
        current_interval = 0.5 * (I[1:] + I[:-1])

        dq = np.zeros_like(T)
        dq[1:] = np.where(interval_mask, direction * current_interval * dt[1:], 0.0)
        q_cc = np.cumsum(dq) / 3600.0

        return V[cc_mask], q_cc[cc_mask]

    def _qd_curve(self, cycle: CycleData) -> tuple[np.ndarray, np.ndarray]:
        if not self.include_cv:
            return self._cc_qd_curve(cycle)

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
        out_w = self.interp_dim + (1 if self.append_soh_scalar else 0)
        out_h = len(self.cycles_to_compare)

        base_cycle = _pick_cycle(
            cell_data,
            self.baseline_cycle_number,
            allow_nearest=self.allow_nearest_cycle,
            max_abs_diff=self.max_cycle_number_gap,
        )
        if base_cycle is None and (cell_data.cycle_data is not None) and len(cell_data.cycle_data) > 0:
            # Some processed datasets may drop early cycles, so the smallest available
            # cycle_number can be far from the requested baseline. For transfer-style
            # sparse checkpoints, falling back to the first available cycle is usually
            # better than failing the entire dataset build.
            base_cycle = min(cell_data.cycle_data, key=lambda c: int(c.cycle_number))
        if base_cycle is None:
            return torch.full((out_h, out_w), float('nan'))

        base_V, base_Qd = self._qd_curve(base_cycle)
        if len(base_Qd) == 0:
            return torch.full((out_h, out_w), float('nan'))
        q0_end = float(np.nanmax(base_Qd))
        if not np.isfinite(q0_end) or q0_end <= 0:
            return torch.full((out_h, out_w), float('nan'))

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
                return torch.full((out_h, out_w), float('nan'))

            V, Qd = self._qd_curve(c)
            if len(Qd) == 0:
                return torch.full((out_h, out_w), float('nan'))
            qnorm = Qd / q0_end
            qnorm_lin = _interp_q_on_voltage(V, qnorm, grid_v)
            diff = qnorm_lin - base_qnorm_lin

            if self.append_soh_scalar:
                soh = float(np.nanmax(Qd) / q0_end)
                diff = np.concatenate([diff, [soh]])
            rows.append(torch.from_numpy(diff.astype(np.float32)))

        feat = torch.stack(rows)
        return feat
