# Multi-Stage Aging Study (Stroebl & Palm, 2024) - Dataset Notes

Source paper (local): `data/raw/Multi-Stage_Aging_Study/2024_SciData_Stroebl_Palm.pdf`

These notes capture:
- how the dataset was generated (calendar vs cycling aging, multi-stage DoE)
- what parameters are controlled
- how RPT/check-ups are scheduled and encoded
- how the raw data is organized and how to parse it

This file is intended as a durable reference for future analysis.

---

## TL;DR

- Cell type: Samsung INR21700-50E (21700 format), nominal ~4.9Ah.
- Scale: 279 cells, 93 distinct aging conditions, ~250 years of accumulated aging time (per paper).
- Two aging modes:
  - `k` = calendar aging (storage at SOC + temperature, cells disconnected between check-ups)
  - `z` = cycling aging (CC charge/discharge, constrained SOC window)
- Two stages:
  - `Stage_1`: baseline DoE (calendar: full-factorial; cycling: conditioned Latin hypercube)
  - `Stage_2`: refinement (pi-OED + additional exploratory points)
- Measurements are stored per cell as a zip. Inside each zip are time-series CSVs and matching `*_meta.txt`.
- Time-series CSV schema is consistent: `run_time,c_vol,c_cur,c_surf_temp,amb_temp,step_type`.

BatteryML integration:

```bash
# Convert raw MSAS zips into BatteryML `BatteryData` pkls
batteryml preprocess MSAS data/raw/Multi-Stage_Aging_Study data/processed/MSAS \
  --config configs/datasets/msas_preprocess.yaml
```

---

## 1) Directory Layout

Top-level:

```
data/raw/Multi-Stage_Aging_Study/
  2024_SciData_Stroebl_Palm.pdf
  experiments_meta.csv
  Stage_1/
    TP_k01_01.zip
    TP_k01_02.zip
    ...
    TP_z01_01.zip
    ...
  Stage_2/
    TP_k01_01.zip
    ...
```

Key points:
- Each `TP_*.zip` corresponds to a single physical cell (one row in `experiments_meta.csv`).
- The filename encodes aging mode + test-point id + replicate:
  - `TP_kNN_RR` = calendar (k), test point NN, replicate RR
  - `TP_zNN_RR` = cycling (z), test point NN, replicate RR

Metadata mapping:
- `experiments_meta.csv` maps `serial` (e.g., `TP_k01_01`) to:
  - lab, stage, sampling strategy
  - aging parameters (temperature/SOC/C-rates)
  - initial mass, voltage, 1kHz resistance

---

## 2) `experiments_meta.csv` schema (global metadata)

Header (as stored):

`serial_internal,serial,lab,type,tp,cell,amb_temp_tp,soc_max_tp,dod_tp,c_ch_tp,c_dch_tp,sampling,stage,m_0,scale,R_1khz_0,resistance_meter,U_0, volt_meter`

Interpretation (high level):
- `serial_internal`: internal unique cell serial (e.g., `S001`, `INT001`, ...)
- `serial`: dataset cell id (matches zip name without `.zip`, e.g. `TP_k01_01`)
- `lab`: lab identifier (e.g. `SIE`, `INT`, `HM`)
- `type`: `k` (calendar) or `z` (cycling)
- `tp`: test-point index
- `cell`: replicate index within that test point
- `amb_temp_tp`: ambient/chamber temperature setpoint for aging condition (degC)
- `soc_max_tp`: calendar storage SOC (for `k`) or upper SOC bound parameter (for `z`)
- `dod_tp`: depth-of-discharge parameter for cycling (`z`); calendar uses 0
- `c_ch_tp`, `c_dch_tp`: charge/discharge C-rates for cycling (`z`); calendar uses 0
- `sampling`: DoE method (e.g. FF, cLH, pi-OED, manual)
- `stage`: 1 or 2
- `m_0`, `scale`: initial cell mass and measuring device
- `R_1khz_0`, `resistance_meter`: initial impedance measurement at 1kHz and instrument
- `U_0`, `volt_meter`: initial OCV and instrument

---

## 3) Aging Protocols

### 3.1 Calendar aging (`type = k`)

Concept:
- Cells are stored at a target SOC and temperature.
- Between check-ups, cells are physically disconnected (open circuit), so there is no
  continuous time-series during storage.

SOC setpoint:
- Target storage SOC is set using CCCV charge/discharge as needed.

Storage conditions:
- Temperature setpoints include 10, 23, 35, 45 degC.

RPT / check-up cadence:
- Check-ups occur every several weeks (paper reports intervals of 5, 4, 8, 8 weeks).
- Some cells only have initial (ET) and final (AT) check-ups to quantify the aging impact
  of check-ups themselves.

### 3.2 Cycling aging (`type = z`)

Concept:
- Repeated CC charge/discharge cycles, designed to avoid CV segments.
- SOC window constrained to 20-80% SOC region to keep CC profiles.

Controlled parameters:
- Ambient temperature `T_amb` (degC)
- Upper SOC parameter `soc_max_tp`
- DOD parameter `dod_tp`
- Charge C-rate `c_ch_tp` (0.05C to 1C)
- Discharge C-rate `c_dch_tp` (0.05C to 2C)

Endpoint logic (paper summary):
- Discharge ends at a fixed lower voltage bound `U_lb` determined after initial discharge.
- Charge termination uses DOD-based logic to avoid SOC drift due to current integration errors.

RPT cadence:
- Check-ups after cycling occur frequently early and less frequently later
  (paper reports 1/1/1/2/2/2/4/4/4/4 week intervals).

---

## 4) Check-ups / RPT definitions

The dataset uses multiple check-up types (filenames include these tags):

- `ET_T10`, `ET_T23`, `ET_T45`: initial check-up at 10/23/45 degC
- `AT_T10`, `AT_T23`, `AT_T45`: final check-up at 10/23/45 degC
- `exCU`: extended check-up (performed at 23 degC)
- `CU`: short check-up (performed at 23 degC)
- `ZYK`: cycling segment (only present for cycling aging cells)

Break-in:
- All cells receive a break-in of three full cycles at 23 degC before the initial ET.

RPT content (paper summary + validated by step_type codes observed in sample files):

- `CU`:
  - capacity measurement
  - HPPC at a single SOC point (50% SOC)

- `exCU`:
  - capacity measurement
  - HPPC at multiple SOC levels (5% to 95%)
  - (no pOCV)

- `ET`/`AT`:
  - capacity measurement
  - pOCV curves at C/20 (charge + discharge)
  - HPPC across SOC levels (5% to 95%)
  - performed at 10, 23, and 45 degC

Observed in sample files:
- `CU` step_type includes `21/22` and `5011..5034` only.
- `exCU` includes `21/22` and `511..9534` (many SOC levels).
- `ET_T23` includes `21/22`, `31/32`, and `511..9534`.
- `ZYK` includes `41/42` plus rest (`0`).

---

## 5) Per-cell zip content and naming convention

Example (calendar, Stage 1): `Stage_1/TP_k01_01.zip`

Inside:

```
TP_k01_01/
  TP_k01_01_01_ET_T10.csv
  TP_k01_01_01_ET_T10_meta.txt
  TP_k01_01_01_ET_T23.csv
  ...
  TP_k01_01_02_CU.csv
  TP_k01_01_02_CU_meta.txt
  TP_k01_01_03_exCU.csv
  ...
  TP_k01_01_XX_AT_T23.csv
  ...
```

Example (cycling, Stage 1): `Stage_1/TP_z01_01.zip`

Adds cycling segments (`ZYK`) interleaved with check-ups:

```
TP_z01_01/
  TP_z01_01_01_ET_T10.csv
  ...
  TP_z01_01_02_ZYK.csv
  TP_z01_01_03_CU.csv
  TP_z01_01_04_ZYK.csv
  ...
  TP_z01_01_XX_AT_T45.csv
```

The `*_meta.txt` file includes:
- lab, internal serial, testpoint
- measurement date
- device/channel/chamber/setpoint
- column definitions (units + sign conventions)

---

## 6) Time-series CSV schema

All observed CSV files share the same header:

`run_time,c_vol,c_cur,c_surf_temp,amb_temp,step_type`

Column meanings (from `*_meta.txt` + paper):
- `run_time`: time since start of the measurement, format `HH:MM:SS.sss`
- `c_vol`: cell voltage [V]
- `c_cur`: cell current [A]
  - negative during discharging (verified in meta text)
- `c_surf_temp`: surface temperature (degC)
- `amb_temp`: ambient/chamber temperature (degC)
  - can be NaN in some files
- `step_type`: integer code identifying the test section

Practical parsing notes:
- `run_time` is relative; convert to seconds for analysis.
- Files use CRLF; robust CSV readers should handle it.

---

## 7) `step_type` decoding (critical)

Basic steps:
- `0`: rest / idle
- `21`: capacity measurement charge (CCCV)
  - paper specifies a CCCV charge (e.g., C/2 to 4.2V, C/20 cutoff)
- `22`: capacity measurement discharge (CCCV)
  - paper specifies a CCCV discharge (e.g., 1C to 2.5V, C/20 cutoff)
- `31`: pOCV CC charge (C/20)
- `32`: pOCV CC discharge (C/20)
- `41`: cycling charge phase
- `42`: cycling discharge phase

HPPC multi-digit codes:
- Paper definition: `100 * SOC + 10 * i_pulse + suffix`
  - `SOC`: SOC level (e.g., 5, 10, 20, ..., 95)
  - `i_pulse`: pulse index (1, 2, or 3)
  - `suffix`:
    - `1`: charge pulse (30s)
    - `2`: discharge pulse (30s)
    - `3`: pause after charge
    - `4`: pause after discharge

HPPC coverage by check-up type:
- `CU`: only 50% SOC (codes `5011..5034`)
- `exCU` and `ET/AT`: SOC sweep (codes `511..9534`, representing 5%..95% SOC)

Examples:
- `5011`..`5034`: HPPC at 50% SOC (used in `CU`)
- `511`..`534`: HPPC at 5% SOC
- `1011`..`1034`: HPPC at 10% SOC
- ...
- `9511`..`9534`: HPPC at 95% SOC

---

## 8) Notable caveats (paper summary)

- Calendar storage periods have no time-series (cells disconnected).
- Cycling data can have gaps due to tester malfunctions or blackouts (multi-day gaps reported).
- Some Stage 1 cells had handling/mounting issues causing corrupted phases or incorrect SOC.
- Final AT at 23 degC may be influenced by stress from surrounding AT tests at 10/45 degC.

---

## 9) How to inspect a cell without extracting zips

Python example (read header from a CSV inside a zip):

```python
import zipfile

zp = 'data/raw/Multi-Stage_Aging_Study/Stage_1/TP_k01_01.zip'
name = 'TP_k01_01/TP_k01_01_02_CU.csv'
with zipfile.ZipFile(zp) as z:
    with z.open(name) as f:
        for _ in range(5):
            print(f.readline().decode('utf-8', errors='replace').rstrip())
```
