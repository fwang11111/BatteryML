# BatteryML Agent Guide

This repository is a Python package plus a CLI (`batteryml`) for battery
degradation ML workflows (download/preprocess/run).

Agent priorities:
- small, focused diffs
- follow conventions in `batteryml/`
- keep workflows config-driven (`configs/**/*.yaml`)

No Cursor rules (`.cursor/rules/`, `.cursorrules`) or Copilot rules
(`.github/copilot-instructions.md`) were found in this repo.

## Quick Orientation

Key entry points:
- `bin/batteryml.py`: CLI implementation (install exposes `batteryml`)
- `batteryml/pipeline.py`: training/evaluation orchestration
- `batteryml/task.py`: builds a `DataBundle` via split -> feature -> label
- `configs/`: YAML configs describing model + data split + feature + label

Core data format:
- Preprocessed battery cells are serialized `BatteryData` objects as `.pkl`.

## Setup / Build
Install (venv recommended):

```bash
python -m venv .venv
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Notes:
- Deep models require PyTorch, but it is NOT pinned in `requirements.txt`.
  Install PyTorch separately per https://pytorch.org/get-started/locally/.
- The code uses PEP 604 unions (`Path | str`), so Python 3.10+ is implied.

Notebook setup:
- `requirements.txt` includes `ipykernel`, but does NOT include Jupyter itself.
  Install `jupyterlab` (or `notebook`) in your environment.

## Run Commands (CLI)

Common CLI invocations:

```bash
# Download raw public dataset
batteryml download MATR /path/to/raw

# Preprocess raw files into `.pkl` `BatteryData`
batteryml preprocess MATR /path/to/raw /path/to/processed

# Train/eval via a YAML config
batteryml run configs/baselines/sklearn/variance_model/matr_1.yaml --workspace ./workspaces/test --train --eval
```

## Test / Single-Test Equivalents

This repo does not include a conventional unit test suite (`tests/`, `pytest.ini`,
`tox.ini` not present). The practical equivalent of a single test is a single
config execution (smoke/regression run):

```bash
# "single test" = run one config end-to-end
batteryml run configs/baselines/sklearn/variance_model/matr_1.yaml \
  --workspace ./workspaces/smoke/matr_1 \
  --train --eval \
  --seed 0 \
  --skip_if_executed false
```

Run many baselines:
- `run_all_rul_baseline.sh` loops through configs and calls `batteryml run ...`.
  It is written for a Unix shell environment.

## Lint / Formatting

This repo includes a `/.flake8` config; `flake8` is not pinned in `requirements.txt`.

```bash
python -m pip install flake8
flake8 .
```

Notes:
- `/.flake8` ignores `F401`/`F403` in `__init__.py`.

## Code Style Conventions (Observed)

### Licensing Header
Most Python modules start with:

```python
# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.
```

When creating new source files under `batteryml/`, include the same header.

### Imports / Typing
- standard lib -> third-party -> local imports
- keep type hints consistent; avoid type suppression

### Naming
- Classes: `PascalCase` (e.g., `Pipeline`, `DataBundle`)
- Functions/vars: `snake_case` (e.g., `build_dataset`, `set_seed`)
- Registry names: `SomethingPredictor`, `SomethingFeatureExtractor`,
  `SomethingLabelAnnotator` (matching config `name:` strings)

### Error Handling
- prefer explicit exceptions for user-facing failures; no empty `except`

## Notebooks

Notebooks live at repo root:
- `baseline.ipynb`
- `soh_example.ipynb`
- `result.ipynb`

When using notebooks, prefer installing BatteryML in editable mode (`pip install -e .`)
so notebook imports reflect your local changes.

## Key Architectural Patterns

### Registry + Config-Driven Construction
`batteryml/builders.py` defines registries:
- `MODELS`, `PREPROCESSORS`, `FEATURE_EXTRACTORS`, `LABEL_ANNOTATORS`,
  `TRAIN_TEST_SPLITTERS`, `DATA_TRANSFORMATIONS`

Pattern for adding a new component:
- Implement the class and decorate with `@<REGISTRY>.register()`.
- Ensure the module is imported somewhere so registration happens at runtime.
- Reference it in YAML configs via `name: <ClassName>`.

### Pipeline / Workspace / Caching
- `batteryml/pipeline.py` derives a default workspace under `workspaces/` based
  on the config path (relative to `configs/`).
- Dataset build results are cached in `cache/battery_cache_<hash>.pkl` based on
  config fields (split/feature/label/transforms).

Be careful when changing config hashing inputs: it affects cache behavior.

## Data Preparation Doc Note

`dataprepare.md` mentions `python scripts/preprocess.py`, but this repository does
not include a `scripts/` directory. Prefer the CLI-based flow:

```bash
batteryml download <DATASET> <raw_dir>
batteryml preprocess <DATASET> <raw_dir> <processed_dir>
```
