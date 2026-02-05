# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from __future__ import annotations

from typing import List

import numpy as np
import torch

from batteryml.builders import LABEL_ANNOTATORS
from batteryml.data.battery_data import BatteryData
from batteryml.label.base import BaseLabelAnnotator
from batteryml.label.soh_efc import (
    _cycle_msas_efc,
    _cycle_q_end_ah,
    _cycle_tag,
    _pick_cycle_by_efc,
)


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    if len(x) < 2:
        return None
    try:
        a, b = np.polyfit(x, y, 1)
    except Exception:
        return None
    y_hat = a * x + b
    sse = float(np.sum((y - y_hat) ** 2))
    return float(a), float(b), sse


def _prepare_soh_points(
    cell_data: BatteryData,
    *,
    baseline_efc: float,
    max_abs_efc_error: float,
    allowed_tags: set[str] | None,
    include_cv: bool,
    cc_current_fraction: float,
    cv_voltage_window_in_V: float,
    allow_baseline_fallback: bool,
    monotonic: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    cycles = list(cell_data.cycle_data or [])
    if not cycles:
        return None

    points: list[tuple[float, float]] = []
    for c in cycles:
        efc = _cycle_msas_efc(c)
        if efc is None or not np.isfinite(efc):
            continue
        if allowed_tags is not None:
            tag = _cycle_tag(c)
            if tag is None or tag not in allowed_tags:
                continue
        q_end = _cycle_q_end_ah(
            c,
            include_cv=include_cv,
            cc_current_fraction=cc_current_fraction,
            cv_voltage_window_in_V=cv_voltage_window_in_V,
        )
        if q_end is None or not np.isfinite(q_end) or q_end <= 0:
            continue
        points.append((float(efc), float(q_end)))

    if not points:
        return None

    base_cycle = _pick_cycle_by_efc(
        cycles,
        baseline_efc,
        max_abs_error=max_abs_efc_error,
        allowed_tags=allowed_tags,
    )
    if base_cycle is None and allow_baseline_fallback:
        base_efc, base_q = min(points, key=lambda t: t[0])
        q0 = base_q
    else:
        q0 = _cycle_q_end_ah(
            base_cycle,
            include_cv=include_cv,
            cc_current_fraction=cc_current_fraction,
            cv_voltage_window_in_V=cv_voltage_window_in_V,
        ) if base_cycle is not None else None

    if q0 is None or not np.isfinite(q0) or q0 <= 0:
        return None

    # Group by EFC (average SOH if duplicate EFCs exist)
    grouped: dict[float, list[float]] = {}
    for efc, q_end in points:
        soh = q_end / q0
        if not np.isfinite(soh):
            continue
        grouped.setdefault(efc, []).append(float(soh))

    if len(grouped) < 2:
        return None

    xs = np.array(sorted(grouped.keys()), dtype=float)
    ys = np.array([float(np.mean(grouped[x])) for x in xs], dtype=float)

    if monotonic:
        ys = np.minimum.accumulate(ys)

    return xs, ys


def _compute_knee_intersection(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_points: int,
    min_slope_drop: float,
) -> float | None:
    n = len(x)
    if n < 2 * min_points:
        return None

    best = None
    for k in range(min_points - 1, n - min_points):
        x1, y1 = x[:k + 1], y[:k + 1]
        x2, y2 = x[k:], y[k:]

        fit1 = _fit_line(x1, y1)
        fit2 = _fit_line(x2, y2)
        if fit1 is None or fit2 is None:
            continue

        a1, b1, sse1 = fit1
        a2, b2, sse2 = fit2

        if not (np.isfinite(a1) and np.isfinite(a2)):
            continue
        if a2 >= a1 - min_slope_drop:
            continue

        denom = a1 - a2
        if abs(denom) < 1e-12:
            continue

        x_star = (b2 - b1) / denom
        if not np.isfinite(x_star):
            continue

        x_left = x[k]
        x_right = x[k + 1] if (k + 1) < n else x[k]
        if x_star < x_left:
            x_star = x_left
        if x_star > x_right:
            x_star = x_right

        sse = sse1 + sse2
        if best is None or sse < best[0]:
            best = (sse, float(x_star))

    return None if best is None else best[1]


@LABEL_ANNOTATORS.register()
class KneeEFCLabelAnnotator(BaseLabelAnnotator):
    """Knee point (EFC) via piecewise linear intersection.

    Returns the intersection point of two fitted line segments as the knee EFC.
    """

    def __init__(
        self,
        baseline_efc: float = 0.0,
        max_abs_efc_error: float = 400.0,
        allowed_tags: List[str] | None = None,
        include_cv: bool = True,
        cc_current_fraction: float = 0.98,
        cv_voltage_window_in_V: float = 0.05,
        min_points: int = 3,
        min_slope_drop: float = 0.0,
        allow_baseline_fallback: bool = True,
        monotonic: bool = True,
    ):
        self.baseline_efc = float(baseline_efc)
        self.max_abs_efc_error = float(max_abs_efc_error)
        self.allowed_tags = set(str(x) for x in allowed_tags) if allowed_tags else None
        self.include_cv = bool(include_cv)
        self.cc_current_fraction = float(cc_current_fraction)
        if not (0.0 < self.cc_current_fraction <= 1.0):
            raise ValueError('cc_current_fraction must be in (0, 1].')
        self.cv_voltage_window_in_V = float(cv_voltage_window_in_V)
        if self.cv_voltage_window_in_V <= 0:
            raise ValueError('cv_voltage_window_in_V must be > 0.')
        self.min_points = int(min_points)
        if self.min_points < 2:
            raise ValueError('min_points must be >= 2.')
        self.min_slope_drop = float(min_slope_drop)
        self.allow_baseline_fallback = bool(allow_baseline_fallback)
        self.monotonic = bool(monotonic)

    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        points = _prepare_soh_points(
            cell_data,
            baseline_efc=self.baseline_efc,
            max_abs_efc_error=self.max_abs_efc_error,
            allowed_tags=self.allowed_tags,
            include_cv=self.include_cv,
            cc_current_fraction=self.cc_current_fraction,
            cv_voltage_window_in_V=self.cv_voltage_window_in_V,
            allow_baseline_fallback=self.allow_baseline_fallback,
            monotonic=self.monotonic,
        )
        if points is None:
            return torch.tensor(float('nan'))

        x, y = points
        knee = _compute_knee_intersection(
            x,
            y,
            min_points=self.min_points,
            min_slope_drop=self.min_slope_drop,
        )
        if knee is None or not np.isfinite(knee):
            return torch.tensor(float('nan'))
        return torch.tensor(float(knee))


@LABEL_ANNOTATORS.register()
class KneeBeforeEFCLabelAnnotator(BaseLabelAnnotator):
    """Binary label for whether knee happens before a threshold EFC."""

    def __init__(
        self,
        threshold_efc: float = 600.0,
        baseline_efc: float = 0.0,
        max_abs_efc_error: float = 400.0,
        allowed_tags: List[str] | None = None,
        include_cv: bool = True,
        cc_current_fraction: float = 0.98,
        cv_voltage_window_in_V: float = 0.05,
        min_points: int = 3,
        min_slope_drop: float = 0.0,
        allow_baseline_fallback: bool = True,
        monotonic: bool = True,
        no_knee_as_negative_if_max_efc_ge_threshold: bool = False,
    ):
        self.threshold_efc = float(threshold_efc)
        self.no_knee_as_negative_if_max_efc_ge_threshold = bool(
            no_knee_as_negative_if_max_efc_ge_threshold
        )
        self.knee_label = KneeEFCLabelAnnotator(
            baseline_efc=baseline_efc,
            max_abs_efc_error=max_abs_efc_error,
            allowed_tags=allowed_tags,
            include_cv=include_cv,
            cc_current_fraction=cc_current_fraction,
            cv_voltage_window_in_V=cv_voltage_window_in_V,
            min_points=min_points,
            min_slope_drop=min_slope_drop,
            allow_baseline_fallback=allow_baseline_fallback,
            monotonic=monotonic,
        )

    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        knee = self.knee_label.process_cell(cell_data)
        if not torch.isnan(knee):
            return torch.tensor(float(knee <= self.threshold_efc))

        if not self.no_knee_as_negative_if_max_efc_ge_threshold:
            return torch.tensor(float('nan'))

        points = _prepare_soh_points(
            cell_data,
            baseline_efc=self.knee_label.baseline_efc,
            max_abs_efc_error=self.knee_label.max_abs_efc_error,
            allowed_tags=self.knee_label.allowed_tags,
            include_cv=self.knee_label.include_cv,
            cc_current_fraction=self.knee_label.cc_current_fraction,
            cv_voltage_window_in_V=self.knee_label.cv_voltage_window_in_V,
            allow_baseline_fallback=self.knee_label.allow_baseline_fallback,
            monotonic=self.knee_label.monotonic,
        )
        if points is None:
            return torch.tensor(float('nan'))
        x, _ = points
        if len(x) == 0:
            return torch.tensor(float('nan'))

        max_efc = float(np.nanmax(x))
        if np.isfinite(max_efc) and max_efc >= self.threshold_efc:
            return torch.tensor(0.0)

        return torch.tensor(float('nan'))
