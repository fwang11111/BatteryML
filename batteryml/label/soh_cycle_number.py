# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from __future__ import annotations

import torch

from batteryml.builders import LABEL_ANNOTATORS
from batteryml.data.battery_data import BatteryData, CycleData

from .base import BaseLabelAnnotator


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

    best = None
    best_diff = None
    for c in cycles:
        diff = abs(int(c.cycle_number) - int(target_cycle_number))
        if best is None or diff < best_diff:
            best = c
            best_diff = diff

    if best is None or best_diff is None or best_diff > max_abs_diff:
        return None
    return best


@LABEL_ANNOTATORS.register()
class SOHAtCycleNumberLabelAnnotator(BaseLabelAnnotator):
    """SOH label at a specific *cycle_number*.

    Unlike `SOHLabelAnnotator`, this does NOT assume dense cycle indices.
    """

    def __init__(
        self,
        cycle_number: int = 600,
        baseline_cycle_number: int = 0,
        mode: str = 'relative',
        allow_nearest_cycle: bool = True,
        max_cycle_number_gap: int = 2,
    ):
        self.cycle_number = int(cycle_number)
        self.baseline_cycle_number = int(baseline_cycle_number)
        self.mode = mode
        self.allow_nearest_cycle = bool(allow_nearest_cycle)
        self.max_cycle_number_gap = int(max_cycle_number_gap)

    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        target = _pick_cycle(
            cell_data,
            self.cycle_number,
            allow_nearest=self.allow_nearest_cycle,
            max_abs_diff=self.max_cycle_number_gap,
        )
        if target is None or target.discharge_capacity_in_Ah is None:
            return torch.tensor(float('nan'))

        q_t = max(target.discharge_capacity_in_Ah)

        if self.mode != 'relative':
            return torch.tensor(float(q_t))

        baseline = _pick_cycle(
            cell_data,
            self.baseline_cycle_number,
            allow_nearest=self.allow_nearest_cycle,
            max_abs_diff=self.max_cycle_number_gap,
        )
        if baseline is None or baseline.discharge_capacity_in_Ah is None:
            # Fallback to nominal capacity if baseline is missing
            q0 = getattr(cell_data, 'nominal_capacity_in_Ah', None)
            if q0 is None or q0 == 0:
                return torch.tensor(float('nan'))
            return torch.tensor(float(q_t) / float(q0))

        q0 = max(baseline.discharge_capacity_in_Ah)
        if q0 == 0:
            return torch.tensor(float('nan'))
        return torch.tensor(float(q_t) / float(q0))
