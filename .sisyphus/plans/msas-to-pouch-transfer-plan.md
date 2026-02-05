# Plan: MSAS -> Pouch Sparse-RPT Transfer (SOH@EFC600 + Knee Risk)

## TL;DR
Use the public Multi-Stage Aging Study (MSAS) check-up discharge curves to train models that map early sparse checkpoints (EFC=0/100/200/300) to:
- SOH at EFC=600 (regression)
- Knee-related risk (knee_EFC regression and knee_before_600 classification)

Alignment axis: use EFC/throughput (not cycle index). Build fixed-checkpoint features by curve alignment + interpolation between check-ups.

---

## Background And Objective

- Target user data: NMC/graphite pouch, sparse RPT checkpoints (e.g., 0/100/200/300), want SOH@600 and knee/plating risk.
- Public data: MSAS (Samsung INR21700-50E). Contains ZYK cycling segments and CU/exCU/ET/AT check-ups.
- Key challenge: MSAS check-ups are time-scheduled; do EFC-based alignment and interpolate to fixed checkpoints.

---

## Current Repo Starting Point (If Present)
These items can be treated as baseline to validate/extend:

- MSAS preprocessor entry: `batteryml/preprocess/preprocess_MSAS.py`
- CLI: `batteryml preprocess MSAS ...` (registered via `batteryml/preprocess/__init__.py`)
- MSAS check-up discharge extraction: step_type 22 (fallback 32) -> `CycleData`
- EFC accumulation: from ZYK discharge (step_type 42) -> stored as `msas_efc` on each check-up `CycleData.additional_data`
- EFC-aligned feature + label (initial):
  - `batteryml/feature/efc_normalized_voltage_capacity_matrix.py`
  - `batteryml/label/soh_efc.py`
- Example config: `configs/soh/transfer/msas_soh_efc600_efcnorm_ridge.yaml`
- Dataset notes: `docs/datasets/multi-stage-aging-study.md`

---

## Scope Boundaries

IN:
- MSAS raw -> BatteryML `BatteryData` reproducible preprocessing
- EFC-aligned feature engineering (including interpolation to fixed checkpoints)
- SOH@EFC600 label generation (prefer interpolation)
- Knee label definition and implementation (knee_EFC / knee_before_600)
- Group-aware splits to avoid leakage (tp/lab/stage)
- Baseline models (ridge/xgb/rf) + evaluation
- Apply the same feature interface to pouch sparse RPT and run inference

OUT (deferred):
- Full HPPC/pOCV/DCIR parsing and electrochem-specific modeling
- Heavy domain adaptation methods as first requirement
- Production deployment

---

## Canonical Definitions

- EFC: cumulative discharge throughput / nominal_capacity
- Check-up curve: CU/exCU/ET/AT discharge capacity measurement segment
- Fixed checkpoints: default `EFC=[0,100,200,300]`
- Two normalizations (keep both):
  - Retention-normalized: `Q(V) / Q0_end`
  - Shape-only normalized: `Q(V) / Q_end`
- Knee:
  - `knee_EFC`: change-point in SOH(EFC)
  - `knee_before_600`: `knee_EFC < 600`

---

## Deliverables

- Data:
  - `data/processed/MSAS/*.pkl`
  - Coverage/QA report: counts, check-up density, EFC coverage, missing rates
- Code:
  - Interpolated EFC checkpoint feature extractor (retention + shape + optional temperature)
  - Interpolated SOH@EFC label annotator
  - Knee label annotator(s)
  - Group splitters for MSAS evaluation
- Configs:
  - MSAS ridge/xgb/rf baseline configs (group split)
  - MSAS->pouch transfer/inference configs
- Docs/notebooks:
  - End-to-end quickstart + common pitfalls (torch, cache, missing check-ups)
  - Sanity-check plots for SOH(EFC) and knee

---

## Verification Strategy

BatteryML-style verification is config-driven end-to-end runs.

- Preprocess:
  - `batteryml preprocess MSAS data/raw/Multi-Stage_Aging_Study data/processed/MSAS --config configs/datasets/msas_preprocess.yaml`
- Train/Eval:
  - `batteryml run configs/soh/transfer/msas_soh_efc600_efcnorm_ridge.yaml --workspace workspaces/msas/soh600 --train --eval`

Artifacts to confirm:
- `workspaces/.../predictions_seed_0_*.pkl` exists and loads
- metrics printed (RMSE/MAE) and dataset size is non-trivial

Important:
- `cache/battery_cache_*.pkl` can mask changes; clear cache after feature/label/split changes.

---

## Execution Strategy: Parallel Waves

Wave 1 (Data + QA):
- Validate MSAS preprocessing correctness, metadata attachment, EFC monotonicity
- Decide comparable tag strategy (recommended default: T23-only)
- Produce MSAS coverage report (EFC range, check-up counts)

Wave 2 (Features/Labels):
- Implement EFC-interpolated fixed-checkpoint curve features
- Implement SOH@EFC600 interpolated label
- Implement knee labels + visualization validation

Wave 3 (Splits + Baselines):
- Implement MSAS group split (tp as group; optional stage/lab stratification)
- Run ridge/xgb/rf baselines and collect results table

Wave 4 (Transfer To Pouch):
- Define pouch EFC/checkpoint metadata input contract
- Run MSAS-trained model inference on pouch processed pkls
- Output per-cell predictions + summary CSV

Wave 5 (Docs/Examples):
- Write end-to-end docs + common gotchas
- Provide a notebook for result browsing and sanity checks

---

## TODOs (Distributable Task List)

### 1) MSAS preprocessing QA and metadata attachment
Goal:
- Ensure each extracted check-up `CycleData` carries `msas_efc` and `msas_tag`
- Attach `msas_stage/msas_lab/msas_type/msas_tp/msas_rep` to `BatteryData` (or an equivalent accessible location) for splits

References:
- `batteryml/preprocess/preprocess_MSAS.py`
- `docs/datasets/multi-stage-aging-study.md`

Acceptance criteria:
- Preprocess completes (or completes on a configured subset)
- Random sample of pkls show monotonic `msas_efc` across check-ups for cycling cells
- Coverage report exists (counts and missing rates)

### 2) Decide comparable tag strategy (recommend: T23-only)
Default recommendation:
- Use only `ET_T23`, `CU`, `exCU`, `AT_T23` to minimize temperature/protocol confounding
- Focus first on cycling cells (`type=z`) for EFC-based alignment

Acceptance criteria:
- Strategy documented and represented in preprocessing/training configs
- Feature/label missing rate decreases relative to mixed-tag baseline

### 3) Choose a robust voltage window and encode in configs
Goal:
- Pick `min_voltage_in_V/max_voltage_in_V` to minimize out-of-range NaNs on the chosen tags

Acceptance criteria:
- With window applied, >~80% of intended cells yield finite features for all checkpoints

### 4) Implement EFC-interpolated fixed-checkpoint curve features
Goal:
- Synthesize curves at exact checkpoints (100/200/300 EFC) by interpolating between adjacent check-ups

Implementation notes:
- Interpolate `qret(V)` and `qshape(V)` on the EFC axis for each voltage-grid point
- No extrapolation by default (out-of-range -> NaN)

References:
- `batteryml/feature/efc_normalized_voltage_capacity_matrix.py`

Acceptance criteria:
- End-to-end `batteryml run ... --train --eval` works without collapsing dataset to near-zero samples
- Feature dimension is fixed and independent of nearest-checkpoint availability

### 5) Add temperature features (T and deltaT)
Goal:
- Include discharge-segment temperature statistics at early checkpoints

Implementation notes:
- Per checkpoint: mean/min/max/std of `c_surf_temp`; endpoint delta
- deltaT uses ambient temp; store needed stats during preprocessing if not present

References:
- `batteryml/preprocess/preprocess_MSAS.py`

Acceptance criteria:
- Feature extraction returns finite values when temperature columns exist

### 6) Implement SOH@EFC600 interpolated label
Goal:
- Prefer interpolated SOH at EFC600 rather than nearest-only selection

References:
- `batteryml/label/soh_efc.py`

Acceptance criteria:
- Cells spanning EFC600 have finite labels
- Cells not spanning EFC600 produce NaN and are filtered cleanly

### 7) Knee labels (knee_EFC and knee_before_600)
Goal:
- Define and compute knee change-point from SOH(EFC)

Default algorithm:
- Piecewise linear change-point search minimizing SSE with minimum points per segment

Acceptance criteria:
- Visual sanity checks on random cells show reasonable knee placement
- knee_EFC distribution not degenerate (not all boundaries)

### 8) Group-aware MSAS splitters to avoid leakage
Goal:
- Hold out by `tp` (test point) as group; keep replicates together
- Optional: stratify by stage/lab

Acceptance criteria:
- No tp appears in both train and test
- Config-driven runs work end-to-end

### 9) Baseline model sweep (ridge/xgb/rf) with results table
Goal:
- Produce comparable RMSE/MAE and sample counts under group split

Acceptance criteria:
- Each config produces predictions artifact
- Results summarized in a table with seeds and workspace paths

### 10) Pouch data contract for EFC/checkpoints
Goal:
- Ensure pouch RPT cycles carry EFC metadata compatible with MSAS feature extractor

Options:
- Extend `RPTPreprocessor` to read a sidecar mapping (filename->EFC)
- Or make extractor read generic `additional_data['efc']` key

Acceptance criteria:
- Pouch sample pkls yield finite early-checkpoint features

### 11) MSAS -> Pouch inference pipeline (config-driven)
Goal:
- Train on MSAS, run inference on pouch cells, output per-cell predictions + summary CSV

Acceptance criteria:
- `batteryml run <transfer-config> --train --eval` produces pouch predictions

### 12) Documentation and reproducibility
Goal:
- End-to-end doc: raw -> processed -> train/eval -> inference
- Common pitfalls: torch install, cache invalidation, missing check-ups

Acceptance criteria:
- A new user/agent can reproduce the full pipeline from the doc

---

## Risks And Mitigations

- Insufficient EFC coverage for some cells -> interpolate; drop uncovered; report missing rate
- Mixed temperature/protocol confounding -> default T23-only; optionally per-tag models
- Voltage window mismatch -> compute coverage-driven window before committing to grid
- Leakage via replicate/tp -> enforce tp-group split and add automatic checks
- Cache hides changes -> clear `cache/battery_cache_*.pkl` after key changes

---

## Definition Of Done

- MSAS preprocessing produces stable pkls with EFC metadata
- Fixed checkpoints features + SOH@EFC600 + knee labels are available with acceptable missing rates
- Group-split baselines (ridge/xgb/rf) run and results are recorded
- Same feature interface works on pouch sparse RPT and produces per-cell predictions
- Docs allow someone else to reproduce the pipeline end-to-end
