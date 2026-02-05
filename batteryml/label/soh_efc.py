# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from __future__ import annotations

from typing import List

import numpy as np
import torch

from batteryml.builders import LABEL_ANNOTATORS
from batteryml.data.battery_data import BatteryData, CycleData
from batteryml.label.base import BaseLabelAnnotator


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


def _infer_discharge_sign(current: np.ndarray) -> str:
    nonzero = current[np.abs(current) > 1e-12]
    if len(nonzero) == 0:
        return 'unknown'
    return 'negative' if np.nanmedian(nonzero) < 0 else 'positive'


def _cc_q_end_ah(
    cycle: CycleData,
    cc_current_fraction: float,
    cv_voltage_window_in_V: float,
) -> float | None:
    V = np.asarray(cycle.voltage_in_V if cycle.voltage_in_V is not None else [], dtype=float)
    I = np.asarray(cycle.current_in_A if cycle.current_in_A is not None else [], dtype=float)
    T = np.asarray(cycle.time_in_s if cycle.time_in_s is not None else [], dtype=float)
    if len(V) == 0 or len(I) == 0 or len(T) == 0:
        return None
    if len(V) != len(I) or len(V) != len(T):
        return None

    valid = np.isfinite(V) & np.isfinite(I) & np.isfinite(T)
    V, I, T = V[valid], I[valid], T[valid]
    if len(I) < 2:
        return None

    order = np.argsort(T)
    V, I, T = V[order], I[order], T[order]

    sign = _infer_discharge_sign(I)
    if sign == 'unknown':
        return None

    discharge_mask = I < -1e-6 if sign == 'negative' else I > 1e-6
    abs_current = np.abs(I[discharge_mask])
    if abs_current.size < 2:
        return None

    median_abs = float(np.nanmedian(abs_current))
    if not np.isfinite(median_abs) or median_abs <= 0:
        return None

    v_min = float(np.nanmin(V[discharge_mask]))
    if not np.isfinite(v_min):
        return None

    low_current = np.abs(I) < cc_current_fraction * median_abs
    near_vmin = V <= (v_min + cv_voltage_window_in_V)
    tail_candidate = discharge_mask & low_current & near_vmin

    tail_mask = np.zeros_like(discharge_mask, dtype=bool)
    idx = np.where(discharge_mask)[0]
    if idx.size > 0:
        i = idx[-1]
        while i >= 0 and tail_candidate[i]:
            tail_mask[i] = True
            i -= 1

    cc_mask = discharge_mask & ~tail_mask
    if cc_mask.sum() < 2:
        return None

    dt = np.diff(T)
    current_interval = 0.5 * (I[1:] + I[:-1])
    interval_mask = cc_mask[:-1] & cc_mask[1:]
    direction = -1.0 if sign == 'negative' else 1.0

    q_total = float(np.sum(direction * current_interval[interval_mask] * dt[interval_mask]) / 3600.0)
    if not np.isfinite(q_total) or q_total <= 0:
        return None
    return q_total


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


def _cycle_q_end_ah(
    cycle: CycleData,
    *,
    include_cv: bool,
    cc_current_fraction: float,
    cv_voltage_window_in_V: float,
) -> float | None:
    if not include_cv:
        return _cc_q_end_ah(cycle, cc_current_fraction, cv_voltage_window_in_V)

    q = getattr(cycle, 'discharge_capacity_in_Ah', None)
    if q is None:
        return None
    q = np.asarray(q, dtype=float)
    if q.size == 0:
        return None
    val = float(np.nanmax(q))
    if not np.isfinite(val) or val <= 0:
        return None
    return val


@LABEL_ANNOTATORS.register()
class SOHAtEFCLabelAnnotator(BaseLabelAnnotator):
    """Label = SOH (capacity retention) at a target EFC for MSAS-style cells.

    If `include_cv` is False, Q is computed by trimming the *tail* CV region:
    points are excluded only when current drops below a fraction of the CC median
    and the voltage is near the discharge Vmin.
    """

    def __init__(
        self,
        target_efc: float = 600.0,
        baseline_efc: float = 0.0,
        max_abs_efc_error: float = 10.0,
        allowed_tags: List[str] | None = None,
        mode: str = 'relative',
        include_cv: bool = True,
        cc_current_fraction: float = 0.98,
        cv_voltage_window_in_V: float = 0.05,
    ):
        self.target_efc = float(target_efc)
        self.baseline_efc = float(baseline_efc)
        self.max_abs_efc_error = float(max_abs_efc_error)
        self.allowed_tags = set(str(x) for x in allowed_tags) if allowed_tags else None
        if mode not in {'relative', 'absolute'}:
            raise ValueError(f'Invalid mode: {mode}. Use "relative" or "absolute".')
        self.mode = mode
        self.include_cv = bool(include_cv)
        self.cc_current_fraction = float(cc_current_fraction)
        if not (0.0 < self.cc_current_fraction <= 1.0):
            raise ValueError('cc_current_fraction must be in (0, 1].')
        self.cv_voltage_window_in_V = float(cv_voltage_window_in_V)
        if self.cv_voltage_window_in_V <= 0:
            raise ValueError('cv_voltage_window_in_V must be > 0.')

    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        cycles = list(cell_data.cycle_data or [])
        if not cycles:
            return torch.tensor(float('nan'))

        base = _pick_cycle_by_efc(
            cycles,
            self.baseline_efc,
            max_abs_error=self.max_abs_efc_error,
            allowed_tags=self.allowed_tags,
        )
        tgt = _pick_cycle_by_efc(
            cycles,
            self.target_efc,
            max_abs_error=self.max_abs_efc_error,
            allowed_tags=self.allowed_tags,
        )
        if base is None or tgt is None:
            return torch.tensor(float('nan'))

        q0 = _cycle_q_end_ah(
            base,
            include_cv=self.include_cv,
            cc_current_fraction=self.cc_current_fraction,
            cv_voltage_window_in_V=self.cv_voltage_window_in_V,
        )
        qt = _cycle_q_end_ah(
            tgt,
            include_cv=self.include_cv,
            cc_current_fraction=self.cc_current_fraction,
            cv_voltage_window_in_V=self.cv_voltage_window_in_V,
        )
        if q0 is None or qt is None:
            return torch.tensor(float('nan'))

        if self.mode == 'absolute':
            return torch.tensor(float(qt))
        return torch.tensor(float(qt / q0))
