# RPTdata preprocess guide (BatteryML/MSAS style)

This note documents the minimum preprocessing requirements for RPT-style data so other agents can ingest your data into BatteryML and run SOH@600 + knee-before-600 workflows.

## 1) What RPTdata means here

RPTdata = sparse discharge-only check-up points (Relative Performance Tests). Each file represents a single discharge curve at a known checkpoint (cycle/EFC index such as 0/100/200/600).

BatteryML expects each checkpoint as one `CycleData` entry inside a `BatteryData` cell. The cycle number stored in `CycleData.cycle_number` must match the real checkpoint (0/100/200/...).

## 2) Input directory layout (recommended)

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
- Parent folder name is the cell id (default behavior).
- Filename must contain the checkpoint number (digits) so the preprocessor can parse `cycle_number`.

## 3) Required columns and units

Minimum columns per file:
- `time` (time axis)
- `I` (current)
- `V` (voltage)

Optional but strongly recommended:
- `T` (temperature)
- `Q` (cumulative discharge capacity)

Preferred units:
- time: seconds
- current: A
- voltage: V
- temperature: degC
- capacity: Ah

If your raw units are different (ms, hours, mA, mV, mAh), set scaling factors in the config.

Discharge current sign:
- BatteryML assumes **negative current for discharge** in many datasets, but RPT preprocessing can handle either.
- Set `discharge_current_sign` to `negative`, `positive`, or `auto`.

If `Q` is missing, the preprocessor will integrate current over time to compute capacity (requires correct discharge sign).

## 4) Config knobs you must set

Use `configs/cyclers/rpt_pouch_example.yaml` as a template. Key fields:

- `cell_id_prefix`: prefix for output cell ids (default `RPT`)
- `cell_id_from`: `parent_dir` (default) or `filename_regex`
- `cell_id_regex`: regex to extract cell id if using filename parsing
- `cycle_number_regex`: regex to extract checkpoint number from filenames
- `columns`: map raw column names to BatteryML fields
- `scales`: unit conversion multipliers
- Cell metadata (important for transfer and voltage windowing):
  - `form_factor`
  - `anode_material`
  - `cathode_material`
  - `nominal_capacity_in_Ah`
  - `min_voltage_limit_in_V`
  - `max_voltage_limit_in_V`

## 5) Preprocess command

```bash
batteryml preprocess RPT <raw_dir> <processed_dir> --config configs/cyclers/rpt_pouch_example.yaml
```

Example output:

```
<processed_dir>/
  RPT_<CELL_A>.pkl
  RPT_<CELL_B>.pkl
  RPT_<CELL_C>.pkl
```

Each file should load via `BatteryData.load()` and contain one `cycle_data` entry per checkpoint.

## 6) Validation checklist

For each output `.pkl`:
- `BatteryData.load(pkl).cell_id` is stable and unique.
- `len(cycle_data)` equals number of RPT checkpoints.
- `cycle_number` values include at least `0` and `600` (or your target horizon).
- Capacity at cycle 0 is close to nominal capacity.
- No NaNs or empty arrays in `V/I/t/Q`.

Quick sanity plots:
- Plot Q(V) curves per checkpoint to confirm shapes are consistent.
- Compute `SOH_600 = Q_600 / Q_0` to verify a reasonable retention value.

## 7) Notes for SOH@600 and knee-before-600

- SOH@600 uses the label `SOHAtEFCLabelAnnotator` (capacity retention at target EFC).
- Knee-before-600 uses `KneeEFCLabelAnnotator` (piecewise linear knee in capacity vs EFC).
- Both labels require a meaningful checkpoint axis, so **cycle numbers must match the real checkpoint** (0/100/200/...).

## 8) MSAS reference (if needed)

MSAS check-ups use tags like `ET_T23`, `CU`, `exCU`, `AT_T23` and are stored as `msas_tag`/`msas_efc` metadata after preprocessing. This guide is for RPT-style data (sparse checkpoints), not full MSAS raw zips.
