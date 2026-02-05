# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from __future__ import annotations

from typing import List

import numpy as np
import torch

from batteryml.builders import FEATURE_EXTRACTORS
from batteryml.data.battery_data import BatteryData, CycleData
from batteryml.feature.base import BaseFeatureExtractor


def _interp_q_on_voltage(voltage_v: np.ndarray,
                         q: np.ndarray,
                         grid_v: np.ndarray) -> np.ndarray:
    voltage_v = np.asarray(voltage_v, dtype=float)
    q = np.asarray(q, dtype=float)

    mask = (~np.isnan(voltage_v)) & (~np.isnan(q))
    voltage_v, q = voltage_v[mask], q[mask]
    if len(voltage_v) < 2:
        return np.full_like(grid_v, np.nan, dtype=float)

    order = np.argsort(voltage_v)
    voltage_v = voltage_v[order]
    q = q[order]

    v_unique, inv = np.unique(voltage_v, return_inverse=True)
    q_sum = np.bincount(inv, weights=q)
    counts = np.bincount(inv)
    q_unique = q_sum / np.clip(counts, 1, None)

    out = np.interp(grid_v, v_unique, q_unique)
    out[(grid_v < v_unique[0]) | (grid_v > v_unique[-1])] = np.nan
    return out


def _scale_q_to_unit_interval(
    voltage_v: np.ndarray,
    q: np.ndarray,
    *,
    min_v: float,
    max_v: float,
) -> np.ndarray:
    voltage_v = np.asarray(voltage_v, dtype=float)
    q = np.asarray(q, dtype=float)

    mask = (
        np.isfinite(voltage_v)
        & np.isfinite(q)
        & (voltage_v >= min_v)
        & (voltage_v <= max_v)
    )
    if mask.sum() < 2:
        return np.full_like(q, np.nan, dtype=float)

    q_min = float(np.nanmin(q[mask]))
    q_max = float(np.nanmax(q[mask]))
    if not np.isfinite(q_min) or not np.isfinite(q_max) or q_max <= q_min:
        return np.full_like(q, np.nan, dtype=float)

    return (q - q_min) / (q_max - q_min)


def _infer_discharge_sign(current: np.ndarray) -> str:
    nonzero = current[np.abs(current) > 1e-12]
    if len(nonzero) == 0:
        return 'unknown'
    return 'negative' if np.nanmedian(nonzero) < 0 else 'positive'


def _tail_cv_mask(
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    discharge_mask: np.ndarray,
    cc_current_fraction: float,
    cv_voltage_window_in_V: float,
) -> np.ndarray:
    if discharge_mask.sum() < 2:
        return np.zeros_like(discharge_mask, dtype=bool)

    abs_current = np.abs(current_a[discharge_mask])
    if abs_current.size < 2:
        return np.zeros_like(discharge_mask, dtype=bool)

    median_abs = float(np.nanmedian(abs_current))
    if not np.isfinite(median_abs) or median_abs <= 0:
        return np.zeros_like(discharge_mask, dtype=bool)

    v_min = float(np.nanmin(voltage_v[discharge_mask]))
    if not np.isfinite(v_min):
        return np.zeros_like(discharge_mask, dtype=bool)

    low_current = np.abs(current_a) < cc_current_fraction * median_abs
    near_vmin = voltage_v <= (v_min + cv_voltage_window_in_V)
    tail_candidate = discharge_mask & low_current & near_vmin

    tail_mask = np.zeros_like(discharge_mask, dtype=bool)
    idx = np.where(discharge_mask)[0]
    if idx.size == 0:
        return tail_mask

    i = idx[-1]
    while i >= 0 and tail_candidate[i]:
        tail_mask[i] = True
        i -= 1

    return tail_mask


def _cc_qd_curve(
    cycle: CycleData,
    cc_current_fraction: float,
    cv_voltage_window_in_V: float,
) -> tuple[np.ndarray, np.ndarray]:
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

    sign = _infer_discharge_sign(I)
    if sign == 'unknown':
        return np.array([]), np.array([])

    order = np.argsort(T)
    V, I, T = V[order], I[order], T[order]

    discharge_mask = I < -1e-6 if sign == 'negative' else I > 1e-6
    tail_mask = _tail_cv_mask(
        V,
        I,
        discharge_mask,
        cc_current_fraction,
        cv_voltage_window_in_V,
    )
    cc_mask = discharge_mask & ~tail_mask
    if cc_mask.sum() < 2:
        return np.array([]), np.array([])
    dt = np.diff(T, prepend=T[0])
    dt[0] = 0.0

    interval_mask = cc_mask[:-1] & cc_mask[1:]
    current_interval = 0.5 * (I[1:] + I[:-1])
    direction = -1.0 if sign == 'negative' else 1.0

    dq = np.zeros_like(T)
    dq[1:] = np.where(interval_mask, direction * current_interval * dt[1:], 0.0)
    q_cc = np.cumsum(dq) / 3600.0

    return V[cc_mask], q_cc[cc_mask]


def _cycle_msas_efc(cycle: CycleData) -> float | None:
    efc = getattr(cycle, 'additional_data', {}).get('msas_efc')
    if efc is None:
        return None
    try:
        return float(efc)
    except (TypeError, ValueError):
        return None


def _cycle_tag(cycle: CycleData) -> str | None:
    tag = getattr(cycle, 'additional_data', {}).get('msas_tag')
    return str(tag) if tag is not None else None


def _pick_cycle_by_efc(
    cycles: List[CycleData],
    target_efc: float,
    *,
    max_abs_error: float,
    allowed_tags: set[str] | None,
) -> CycleData | None:
    best = None
    best_err = float('inf')
    for c in cycles:
        efc = _cycle_msas_efc(c)
        if efc is None:
            continue
        if allowed_tags is not None:
            tag = _cycle_tag(c)
            if tag is None or tag not in allowed_tags:
                continue

        err = abs(efc - float(target_efc))
        if best is None or err < best_err:
            best = c
            best_err = err

    if best is None:
        return None
    return best if best_err <= float(max_abs_error) else None


def _qd_curve(
    cycle: CycleData,
    *,
    include_cv: bool,
    cc_current_fraction: float,
    cv_voltage_window_in_V: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not include_cv:
        return _cc_qd_curve(cycle, cc_current_fraction, cv_voltage_window_in_V)

    V = np.asarray(cycle.voltage_in_V if cycle.voltage_in_V is not None else [], dtype=float)
    Qd = np.asarray(
        cycle.discharge_capacity_in_Ah if cycle.discharge_capacity_in_Ah is not None else [],
        dtype=float,
    )
    if len(V) == 0 or len(Qd) == 0 or len(V) != len(Qd):
        return np.array([]), np.array([])

    # Prefer a Qd-increasing mask (robust to sign conventions)
    dq = np.diff(Qd, prepend=Qd[0])
    inc_mask = dq > 1e-9
    if inc_mask.sum() >= 2:
        return V[inc_mask], Qd[inc_mask]
    return V, Qd


@FEATURE_EXTRACTORS.register()
class EFCNormalizedVoltageCapacityMatrixFeatureExtractor(BaseFeatureExtractor):
    """Capacity-normalized delta curves aligned by MSAS EFC metadata.

    This is designed for Multi-Stage Aging Study preprocessed cells where each
    check-up discharge curve has `msas_efc` in `CycleData.additional_data`.

    Output per checkpoint:
    - delta_retention(v) = Q(v)/Q0_end - Q0(v)/Q0_end
    - delta_shape(v) = Q_scaled(v) - Q0_scaled(v)
      where Q_scaled is Q scaled to [0, 1] using Q range within [Vmin, Vmax]
    - optional scalar retention = Q_end / Q0_end

    Final output is a flat vector concatenating checkpoints in order.

    If `include_cv` is False, Q is computed by trimming the *tail* CV region:
    points are excluded only when current drops below a fraction of the CC median
    and the voltage is near the discharge Vmin.
    """

    def __init__(
        self,
        interp_dim: int = 256,
        baseline_efc: float = 0.0,
        efc_checkpoints: List[float] | float | None = None,
        min_voltage_in_V: float | None = None,
        max_voltage_in_V: float | None = None,
        max_abs_efc_error: float = 100.0,
        allowed_tags: List[str] | None = None,
        append_soh_scalar: bool = True,
        include_shape_delta: bool = True,
        include_cv: bool = True,
        cc_current_fraction: float = 0.98,
        cv_voltage_window_in_V: float = 0.05,
    ):
        self.interp_dim = int(interp_dim)
        self.baseline_efc = float(baseline_efc)
        if efc_checkpoints is None:
            efc_checkpoints = [100.0, 200.0, 300.0]
        if isinstance(efc_checkpoints, (int, float)):
            efc_checkpoints = [float(efc_checkpoints)]
        self.efc_checkpoints = [float(x) for x in efc_checkpoints]
        self.min_voltage_in_V = min_voltage_in_V
        self.max_voltage_in_V = max_voltage_in_V
        self.max_abs_efc_error = float(max_abs_efc_error)
        self.allowed_tags = set(str(x) for x in allowed_tags) if allowed_tags else None
        self.append_soh_scalar = bool(append_soh_scalar)
        self.include_shape_delta = bool(include_shape_delta)
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
        # MSAS check-up curves may span different voltage windows; enforce explicit range.
        raise ValueError(
            'Voltage range not specified. Provide min_voltage_in_V/max_voltage_in_V '
            'or ensure BatteryData has min/max voltage limits.'
        )

    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        cycles = list(cell_data.cycle_data or [])
        if not cycles:
            return torch.full((self._out_dim(),), float('nan'))

        base_cycle = _pick_cycle_by_efc(
            cycles,
            self.baseline_efc,
            max_abs_error=self.max_abs_efc_error,
            allowed_tags=self.allowed_tags,
        )
        if base_cycle is None:
            return torch.full((self._out_dim(),), float('nan'))

        base_V, base_Qd = _qd_curve(
            base_cycle,
            include_cv=self.include_cv,
            cc_current_fraction=self.cc_current_fraction,
            cv_voltage_window_in_V=self.cv_voltage_window_in_V,
        )
        if len(base_Qd) == 0:
            return torch.full((self._out_dim(),), float('nan'))

        q0_end = float(np.nanmax(base_Qd))
        if not np.isfinite(q0_end) or q0_end <= 0:
            return torch.full((self._out_dim(),), float('nan'))

        min_v, max_v = self._get_voltage_range(cell_data)
        grid_v = np.linspace(min_v, max_v, self.interp_dim)

        base_qret = base_Qd / q0_end
        base_qret_lin = _interp_q_on_voltage(base_V, base_qret, grid_v)

        # Shape baseline uses its own end capacity.
        base_qshape = _scale_q_to_unit_interval(
            base_V,
            base_Qd,
            min_v=min_v,
            max_v=max_v,
        )
        base_qshape_lin = _interp_q_on_voltage(base_V, base_qshape, grid_v)

        parts: list[np.ndarray] = []
        for efc_t in self.efc_checkpoints:
            c = _pick_cycle_by_efc(
                cycles,
                efc_t,
                max_abs_error=self.max_abs_efc_error,
                allowed_tags=self.allowed_tags,
            )
            if c is None:
                return torch.full((self._out_dim(),), float('nan'))

            V, Qd = _qd_curve(
                c,
                include_cv=self.include_cv,
                cc_current_fraction=self.cc_current_fraction,
                cv_voltage_window_in_V=self.cv_voltage_window_in_V,
            )
            if len(Qd) == 0:
                return torch.full((self._out_dim(),), float('nan'))

            q_end = float(np.nanmax(Qd))
            if not np.isfinite(q_end) or q_end <= 0:
                return torch.full((self._out_dim(),), float('nan'))

            qret = Qd / q0_end
            qret_lin = _interp_q_on_voltage(V, qret, grid_v)
            delta_ret = qret_lin - base_qret_lin
            parts.append(delta_ret)

            if self.include_shape_delta:
                qshape = _scale_q_to_unit_interval(
                    V,
                    Qd,
                    min_v=min_v,
                    max_v=max_v,
                )
                qshape_lin = _interp_q_on_voltage(V, qshape, grid_v)
                delta_shape = qshape_lin - base_qshape_lin
                parts.append(delta_shape)

            if self.append_soh_scalar:
                parts.append(np.array([q_end / q0_end], dtype=float))

        out = np.concatenate(parts, axis=0)
        if out.size != self._out_dim():
            return torch.full((self._out_dim(),), float('nan'))
        return torch.tensor(out, dtype=torch.float)

    def _out_dim(self) -> int:
        per = self.interp_dim
        if self.include_shape_delta:
            per += self.interp_dim
        if self.append_soh_scalar:
            per += 1
        return len(self.efc_checkpoints) * per
