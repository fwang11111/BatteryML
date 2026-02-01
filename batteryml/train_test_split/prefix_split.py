# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from batteryml.builders import TRAIN_TEST_SPLITTERS
from batteryml.train_test_split.base import BaseTrainTestSplitter


@TRAIN_TEST_SPLITTERS.register()
class CellIdPrefixTrainTestSplitter(BaseTrainTestSplitter):
    """Split by explicit cell IDs or by cell-id prefix.

    Useful for: train on public cells, test on custom cells.
    """

    def __init__(
        self,
        cell_data_path: str | List[str],
        test_cell_ids: List[str] | None = None,
        test_prefixes: List[str] | None = None,
    ):
        BaseTrainTestSplitter.__init__(self, cell_data_path)
        test_cell_ids = set(test_cell_ids or [])
        test_prefixes = list(test_prefixes or [])
        if not test_cell_ids and not test_prefixes:
            raise ValueError('Provide test_cell_ids and/or test_prefixes.')

        self.train_cells, self.test_cells = [], []
        for filename in self._file_list:
            stem = Path(filename).stem
            is_test = False
            if stem in test_cell_ids:
                is_test = True
            if not is_test and test_prefixes:
                is_test = any(stem.startswith(p) for p in test_prefixes)
            if is_test:
                self.test_cells.append(filename)
            else:
                self.train_cells.append(filename)

    def split(self) -> Tuple[List, List]:
        return self.train_cells, self.test_cells
