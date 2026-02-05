# BatteryML Feature Extractors

This document lists the currently available feature extractors in this repo
(usable via YAML `feature: { name: ... }`), their parameters, and their
input/output formats.

Source of truth:
- Registry: `batteryml/builders.py` (`FEATURE_EXTRACTORS`)
- Feature modules: `batteryml/feature/*.py`

---

## Common Input / Output Contract

All feature extractors implement `BaseFeatureExtractor.__call__(cells)`
(`batteryml/feature/base.py`).

Input:
- `cells`: `list[batteryml.data.battery_data.BatteryData]`

Output:
- `torch.Tensor` (float32)
- Returned tensor is `torch.stack([process_cell(cell) ...])`
  so **every cell must return the same shape**.

### BatteryData / CycleData fields used

Depending on the extractor, each `BatteryData` must contain:
- `cycle_data`: list of `CycleData`
- `min_voltage_limit_in_V`, `max_voltage_limit_in_V` (required by Severson
  `Qdlin` logic unless you override voltage range in the config)

Each `CycleData` may need (varies by extractor):
- Always needed for Q(V) / Qdlin-based features:
  - `voltage_in_V`: list[float]
  - `current_in_A`: list[float]
  - `discharge_capacity_in_Ah`: list[float]
- Sometimes needed:
  - `charge_capacity_in_Ah` (for Coulombic efficiency in plotting only)
  - `time_in_s` (for charge-time features)
  - `temperature_in_C` (for temperature features)
  - `internal_resistance_in_ohm` (for IR features)

Notes:
- Severson `Qdlin` computation treats discharge points as `I < -eps` with
  `eps=1e-1` (`batteryml/feature/severson.py:_get_Qdlin`).
- Some extractors can optionally use precomputed `Qdlin` stored in
  `cycle_data.additional_data['Qdlin']`.

---

## 1) VarianceModelFeatureExtractor

Name:
- `VarianceModelFeatureExtractor` (`batteryml/feature/variance_model.py`)

Base:
- Inherits `SeversonFeatureExtractor` (`batteryml/feature/severson.py`)

What it computes:
- A single scalar: `Variance = log10(var(ΔQdlin) + eps)`
- `ΔQdlin = Qdlin(late_cycle) - Qdlin(early_cycle)`

Parameters (inherited from SeversonFeatureExtractor):
- `interp_dims: int = 1000`
  - Note: current implementation of `_get_Qdlin` is hardcoded to 1000 points.
- `critical_cycles: list[int] = [1, 9, 99]`
  - Uses `early = critical_cycles[1]`, `late = critical_cycles[2]` (cycle *index*
    into `cell_data.cycle_data`, not cycle_number).
- `smooth_diff_qdlin: bool = True`
- `use_precalculated_qdlin: bool = False`

Input requirements:
- `cell_data.cycle_data` must be long enough to index `critical_cycles`.
- Each selected cycle must have `current_in_A`, `voltage_in_V`,
  `discharge_capacity_in_Ah`.

Output:
- Per-cell: shape `(1,)` (1 scalar)
- Batch: shape `(N, 1)`

---

## 2) DischargeModelFeatureExtractor

Name:
- `DischargeModelFeatureExtractor` (`batteryml/feature/discharge_model.py`)

Base:
- Inherits `SeversonFeatureExtractor`

What it computes (6 scalars):
- From `ΔQdlin` (all log-scaled):
  - `Minimum`
  - `Variance`
  - `Skewness`
  - `Kurtosis`
- From discharge capacity fade curve (raw scale):
  - `Early discharge capacity`
  - `Difference between max discharge capacity and early discharge capacity`

Parameters:
- Same as SeversonFeatureExtractor (see above)

Input requirements:
- Same as VarianceModelFeatureExtractor
- Additionally, `Early discharge capacity` and `Difference...` require
  `max(cycle.discharge_capacity_in_Ah)` to be meaningful for cycles in range.

Output:
- Per-cell: shape `(6,)`
- Batch: shape `(N, 6)`

---

## 3) FullModelFeatureExtractor

Name:
- `FullModelFeatureExtractor` (`batteryml/feature/full_model.py`)

Base:
- Inherits `SeversonFeatureExtractor`

What it computes (9 scalars):
- From `ΔQdlin` (log-scaled):
  - `Minimum`
  - `Variance`
- From discharge capacity fade curve (linear regression over early cycles):
  - `Slope of linear fit to the capacity curve`
  - `Intercept of linear fit to the capacity curve`
- Additional early-cycle features:
  - `Early discharge capacity`
  - `Average early charge time` (log-scaled)
  - `Integral of temperature over time` (log-scaled)
  - `Minimum internal resistance`
  - `Internal resistance change`

Parameters:
- Same as SeversonFeatureExtractor

Input requirements:
- Same as DischargeModelFeatureExtractor
- For `Average early charge time`: needs `time_in_s` for early cycles.
- For `Integral of temperature over time`: needs `temperature_in_C`.
- For IR features: needs `internal_resistance_in_ohm`.

Output:
- Per-cell: shape `(9,)`
- Batch: shape `(N, 9)`

---

## 4) VoltageCapacityMatrixFeatureExtractor

Name:
- `VoltageCapacityMatrixFeatureExtractor` (`batteryml/feature/voltage_capacity_matrix.py`)

What it computes:
- For each selected cycle (by **cycle index**), compute:
  - `diff_qdlin = Qdlin(cycle) - Qdlin(diff_base_cycle)`
- Stack selected cycles into a matrix.

Parameters:
- `interp_dim: int = 1000`
  - Note: `get_Qdlin` currently returns 1000 points regardless; this arg is
    retained for API consistency.
- `diff_base: int = 9`
  - Baseline cycle index used for differencing.
- `cycles_to_keep: list[int] | int | None = None`
  - If set, only keep these cycle indices.
- `min_cycle_index: int = 0`
- `max_cycle_index: int = 99`
- `use_precalculated_qdlin: bool = False`
- `smooth: bool = True`
  - Applies robust median smoothing (`batteryml/feature/severson.py:smooth`).
- `cycle_average: int | None = None`
  - Downsample `Qdlin` by slicing `[..., ::cycle_average]`.

Input requirements:
- Needs `cell_data.min_voltage_limit_in_V/max_voltage_limit_in_V` for Qdlin.
- Assumes discharge is `I < -0.1` when computing Qdlin.
- Requires cycles by **index** to exist up to `max_cycle_index` (or the
  indices in `cycles_to_keep`).

Output:
- Let:
  - `H = number of kept cycles` (depends on `min/max` and `cycles_to_keep`)
  - `W = 1000` if `cycle_average is None`, else `ceil(1000 / cycle_average)`
- Per-cell: shape `(H, W)`
- Batch: shape `(N, H, W)`

---

## 5) NormalizedVoltageCapacityMatrixFeatureExtractor

Name:
- `NormalizedVoltageCapacityMatrixFeatureExtractor` (`batteryml/feature/normalized_voltage_capacity_matrix.py`)

What it computes (designed for sparse RPT + cross-capacity transfer):
- Baseline capacity: `Q0_end = max(Qd_baseline)`
- Normalize: `q_norm = Qd / Q0_end`
- Interpolate each cycle’s `q_norm(V)` onto a fixed `V_grid`
- For each target cycle number in `cycles_to_compare`:
  - `diff = q_norm_target(V_grid) - q_norm_baseline(V_grid)`
  - Optionally append a scalar `soh_scalar = max(Qd_target)/Q0_end`

Notes on Qd:
- If `include_cv=True`, Qd comes directly from `cycle.discharge_capacity_in_Ah`.
- If `include_cv=False`, Qd is recomputed by **trimming only the tail CV region**
  (low current near Vmin) and integrating current over time. This keeps the
  high-voltage portion of the discharge curve intact while removing the CV tail.

Parameters:
- `interp_dim: int = 256`
- `baseline_cycle_number: int = 0`
  - Selected by **cycle_number** (not list index). If not found, falls back to
    the smallest available cycle_number.
- `cycles_to_compare: list[int] | int = [100, 200]`
  - Selected by **cycle_number**.
- `min_voltage_in_V: float | None = None`
- `max_voltage_in_V: float | None = None`
  - If not provided, uses `BatteryData.min_voltage_limit_in_V/max_voltage_limit_in_V`.
- `discharge_current_sign: str = 'auto'`
  - `auto|negative|positive` (used only as a fallback; extractor prefers Qd
    monotonic increase to select discharge points).
- `allow_nearest_cycle: bool = True`
- `max_cycle_number_gap: int = 2`
  - If the exact cycle number is missing, allow nearest within this gap.
- `append_soh_scalar: bool = True`
- `include_cv: bool = True`
  - If False, drop only the tail CV region (low current near Vmin) and recompute
    Qd from current/time.
- `cc_current_fraction: float = 0.98`
  - A point is considered low-current if `|I| < cc_current_fraction * median(|I|)`
    over discharge points.
- `cv_voltage_window_in_V: float = 0.05`
  - Tail CV candidates are restricted to voltages within `Vmin + window`.

Input requirements:
- Each cell must have a usable baseline cycle and each cycle in
  `cycles_to_compare` (exact or nearest).
- Each required cycle must have `voltage_in_V` and `discharge_capacity_in_Ah`.

Output:
- Let:
  - `H = len(cycles_to_compare)`
  - `W = interp_dim + (1 if append_soh_scalar else 0)`
- Per-cell: shape `(H, W)`
- Batch: shape `(N, H, W)`

---

## 6) EFCNormalizedVoltageCapacityMatrixFeatureExtractor

Name:
- `EFCNormalizedVoltageCapacityMatrixFeatureExtractor` (`batteryml/feature/efc_normalized_voltage_capacity_matrix.py`)

Purpose:
- Designed for MSAS-style data where each check-up curve has
  `CycleData.additional_data['msas_efc']` and `msas_tag`.
- Uses EFC checkpoints (e.g. 100/200/300) instead of cycle numbers.

Inputs used:
- `CycleData.voltage_in_V`
- `CycleData.current_in_A`
- `CycleData.time_in_s`
- `CycleData.discharge_capacity_in_Ah`
- `CycleData.additional_data['msas_efc']` (required)
- `CycleData.additional_data['msas_tag']` (optional; used by `allowed_tags`)

Key parameters:
- `interp_dim: int = 256`
- `baseline_efc: float = 0.0`
- `efc_checkpoints: list[float] = [100.0, 200.0, 300.0]`
- `min_voltage_in_V: float | None`
- `max_voltage_in_V: float | None`
- `max_abs_efc_error: float = 10.0`
- `allowed_tags: list[str] | None = None`
- `append_soh_scalar: bool = True`
- `include_shape_delta: bool = True`
- `include_cv: bool = True`
- `cc_current_fraction: float = 0.98`
- `cv_voltage_window_in_V: float = 0.05`

Detailed computation (step-by-step):

1) **Select baseline and checkpoints**
   - Baseline cycle = closest cycle with `msas_efc` to `baseline_efc`
     within `max_abs_efc_error`.
   - For each `efc_checkpoint`, select the closest cycle within the same
     tolerance.
   - If any required cycle is missing, return all-NaN features for the cell.
   - If `allowed_tags` is provided, only cycles whose `msas_tag` is in the list
     are considered.

2) **Build Q(V)**
   - If `include_cv=True`: use `cycle.discharge_capacity_in_Ah` directly.
   - If `include_cv=False`: recompute Q from current/time while trimming only
     the **tail CV region**:
     - Determine discharge sign from median of nonzero current.
     - `discharge_mask` = `I < -eps` (negative) or `I > eps` (positive).
     - `median_abs = median(|I|)` over discharge points.
     - `v_min = min(V)` over discharge points.
     - `low_current = |I| < cc_current_fraction * median_abs`.
     - `near_vmin = V <= v_min + cv_voltage_window_in_V`.
     - `tail_candidate = discharge_mask & low_current & near_vmin`.
     - **Trim only the trailing contiguous tail** where `tail_candidate` is true.
     - Integrate current over time (trapezoid, using adjacent kept points) to
       produce cumulative Q in Ah.

3) **Retention normalization (delta_retention)**
   - `Q_ref = Q0_end = max(Q_baseline)`.
   - For each cycle, compute `q_ret = Q / Q_ref`.
   - Interpolate `q_ret(V)` onto `V_grid = linspace(min_voltage_in_V, max_voltage_in_V, interp_dim)`.
   - `delta_retention = q_ret_checkpoint(V_grid) - q_ret_baseline(V_grid)`.

4) **Shape normalization (delta_shape)**
   - Within the voltage window `[min_voltage_in_V, max_voltage_in_V]`, compute
     `Q_min` and `Q_max` from the cycle’s Q values.
   - Scale to unit interval: `q_shape = (Q - Q_min) / (Q_max - Q_min)`.
   - Interpolate `q_shape(V)` onto the same `V_grid`.
   - `delta_shape = q_shape_checkpoint(V_grid) - q_shape_baseline(V_grid)`.

5) **Optional scalar SOH**
   - If `append_soh_scalar=True`, append `soh_scalar = Q_end_checkpoint / Q_ref`.

6) **Output shape**
   - Per checkpoint length:
     - `interp_dim` for `delta_retention`
     - `interp_dim` for `delta_shape` (if enabled)
     - `1` for `soh_scalar` (if enabled)
   - Final vector = concatenation across checkpoints in order.
   - Shape: `(len(efc_checkpoints) * per_checkpoint_dim,)`.

NaN behavior:
- If baseline or any checkpoint is missing (or Q cannot be computed), the
  extractor returns an all-NaN vector for that cell.
