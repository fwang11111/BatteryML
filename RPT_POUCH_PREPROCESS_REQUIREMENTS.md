# RPT Pouch Preprocess Requirements

This document specifies the **minimum requirements** to preprocess sparse
RPT-style discharge data (CSV/Excel) into BatteryML `BatteryData` `.pkl` files.

Goal: Convert your 3 pouch cells (NMC/graphite, ~70–100Ah) with RPT checkpoints
at cycles `0, 100, 200, 300, 400, 500, 600` into:

```
data/processed/MY_POUCH/
  RPT_<CELL_A>.pkl
  RPT_<CELL_B>.pkl
  RPT_<CELL_C>.pkl
```

Each `.pkl` must load via `BatteryData.load()` and contain a `cycle_data` list
where each element is one RPT discharge curve.

---

## 1) Input Data Structure

Recommended folder layout (simplest to parse):

```
<raw_dir>/
  <cell_id_1>/
    rpt_000.csv
    rpt_100.csv
    rpt_200.csv
    ...
  <cell_id_2>/
    rpt_000.xlsx
    rpt_100.xlsx
    ...
```

Rules:
- One file = one RPT checkpoint (one discharge curve).
- The **parent folder name** is the `cell_id` (unless you explicitly choose a
  filename-based cell id strategy).
- The file name must contain the **cycle number** (0/100/200/...) as digits.

Example filename patterns that work:
- `rpt_000.csv` / `rpt_100.csv`
- `cycle-200.xlsx`
- `CellA_cycle600.csv`

---

## 2) Required Columns (per file)

Each file must contain a table with at least these columns:

- `time` (time axis)
- `I` (current)
- `V` (voltage)

Optional but strongly recommended:
- `T` (temperature)
- `Q` (cumulative discharge capacity)

BatteryML target fields per `CycleData`:
- `time_in_s`
- `current_in_A`
- `voltage_in_V`
- `temperature_in_C` (optional)
- `discharge_capacity_in_Ah` (required; can be computed from I/t if Q missing)

Data constraints:
- Time must be monotonic increasing (or at least sortable).
- Voltage/current arrays must be the same length as time.

---

## 3) Units + Sign Conventions

Preferred units:
- time: seconds
- current: A
- voltage: V
- temperature: degC
- capacity: Ah

If your files use other units (ms, hours, mA, mV, mAh), you must provide
`scales` to convert to the preferred units.

Discharge current sign:
- BatteryML datasets commonly use **negative current for discharge**.
- Your cycler may use positive current for discharge.

Preprocessor must support:
- `discharge_current_sign = positive | negative | auto`

---

## 4) Cycle Number Semantics

Each RPT file corresponds to a real cycle count (0/100/200/.../600).
The processed `CycleData.cycle_number` must be set to that real cycle count.

This is critical because downstream feature/label logic will select
`cycle_number` directly (not list index).

---

## 5) Cell-Level Metadata (required for good transfer)

Each output `BatteryData` must include:
- `form_factor = "pouch"`
- `anode_material = "graphite"`
- `cathode_material = "NMC"`
- `nominal_capacity_in_Ah` (70–100)
- `min_voltage_limit_in_V`, `max_voltage_limit_in_V` (your RPT bounds)

If you plan to train on RWTH (3.5–3.9V window), you must also be able to
restrict your curves to that voltage window for feature extraction.

---

## 6) Expected Output + Validation

For each cell output `.pkl`:
- `BatteryData.load(pkl).cell_id` is stable and unique
- `len(cycle_data)` equals number of available RPT checkpoints
- The set of `cycle_number` contains at least: 0, 100, 200, 600
- `max(discharge_capacity_in_Ah)` at cycle 0 is close to nominal capacity
- No NaNs/empty arrays in V/I/t/Q

Suggested quick sanity checks:
- Plot V vs Q for each checkpoint and confirm curves look consistent.
- Compute SOH_600 = Q_600 / Q_0 and confirm it is reasonable.
