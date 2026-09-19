# Clinic Data Schema

The intake schema for clinic data (`validate_clinic_data.py`, the research
calibration diagnostics in `calibrate_for_clinic.py`, and `fit_befe_ood.py`).
In 7.1 none of these changes the L7 headline; see
[CLINIC_CALIBRATION_GUIDE.md](CLINIC_CALIBRATION_GUIDE.md).

**Blank and 0 are different.** A blank cell means *unknown*; 0 means *observed
absence*. Do not fill gaps with zeros or averages.

Documented here as text rather than shipped as a spreadsheet. Spreadsheets are
how patient rows reach a repository by accident, so this project blanket-blocks
`.xlsx` in `.gitignore`, in the pre-commit hook, and in the public export
filter. A filled `.xlsx` template is provided directly with commercial
onboarding; for research use, a CSV matching the columns below works with both
scripts.

## Columns

One row per **cycle**, not per patient. 30 columns in five groups.

### Identification

| Column | Type | Notes |
|---|---|---|
| `cycle_id` | string | Your internal cycle reference |
| `patient_id` | string | **Pseudonymous.** Never a name, DOB, or medical record number. A stable hash is fine and is what lets repeat attempts link. |

> Do not add a name, date of birth, address, or free-text clinical note column.
> None is used by any model, and each turns a counts table into a personal data
> record with the compliance burden that implies.

### Demographics and baseline

| Column | Type | Unit |
|---|---|---|
| `age` | float | years at cycle start |
| `amh` | float | ng/mL |
| `afc` | int | antral follicle count |
| `bmi` | float | kg/m² |
| `attempt_number` | int | 1-based; drives the NVSA decay correction |
| `sperm_source` | category | Esteves stratum: `ejaculate`, `testicular_NOA`, `testicular_OA`, `epididymal`; accepted aliases `donor`, `partner` (→ ejaculate), `tese_noa`, `tese_oa`, `pesa`, `mesa`. Other values are rejected, not treated as ejaculate |

### Stimulation protocol

| Column | Type | Unit |
|---|---|---|
| `diagnosis` | category | primary indication |
| `protocol_type` | category | antagonist / agonist long / short / other |
| `fsh_start_iu` | float | IU/day starting gonadotrophin dose |
| `rlh_used` | bool | recombinant LH supplementation |
| `rlh_dose_iu` | float | IU/day, blank if `rlh_used` is false |
| `stim_days` | int | days of stimulation |
| `follicles_14mm` | int | follicles ≥14 mm at trigger |
| `e2_trigger_pmol` | float | estradiol at trigger, pmol/L |

### Laboratory results

| Column | Type | Notes |
|---|---|---|
| `okk` | int | cumulus-oocyte complexes retrieved |
| `mii` | int | mature oocytes. MII > OCC can be legitimate (warmed oocytes added) |
| `pn2` | int | two-pronuclear zygotes |
| `cleavage_d3` | int | day-3 cleavage-stage embryos |
| `blasts_total` | int | blastocysts |
| `blasts_good` | int | good-quality blastocysts |
| `emb_frozen` | int | embryos cryopreserved |
| `emb_transferred` | int | embryos transferred |
| `euploid` | int | euploid embryos; blank when PGT-A not performed |
| `ohss_grade` | category | none / mild / moderate / severe |
| `cycle_cancelled` | bool | cancelled before retrieval |

### Cycle outcome

| Column | Type | Notes |
|---|---|---|
| `outcome` | category | clinical pregnancy confirmed by ultrasound (ectopic counts as positive); not live birth |
| `outcome_date` | date | leave blank if it would narrow identification |
| `outcome_known` | bool | **false for cycles still in follow-up.** Rows with `outcome_known = false` are excluded from calibration rather than counted as failures — treating pending cycles as negatives is the most common way clinic recalibration goes wrong. |

## Volume

| Rows | What you get |
|---|---|
| < 100 | Validation runs; the clinic batches barely move the L3 Beta-Binomial posterior. Expected — the Bayesian design is doing its job. |
| 100–300 | The L3 posterior (`bayes_mean`) shifts meaningfully toward your population. |
| > 500 | Calibration diagnostics per phenotype become informative. |

The clinic build's centre calibration needs at least 100 pregnancies, 100
negative outcomes and 100 patients among eligible single blastocyst transfers;
it is not part of this repository.

## Workflow

```bash
python validate_clinic_data.py --input your_cycles.csv --output validated.csv   # schema, ranges, missingness
python calibrate_for_clinic.py --data validated.csv --clinic "Clinic name"       # research diagnostics
```

`validate_clinic_data.py` reports funnel violations (`blasts_good > blasts_total`;
`mii > okk`, which is legitimate when warmed oocytes were added — check before
correcting), out-of-range values, and missingness per column.
Fix what it reports before calibrating — a funnel violation usually means a
column mapping error, and calibrating on it will quietly bias every downstream
layer.

Then copy `clinic_config.template.json` to `clinic_config.json` and fill
`batches` with `[successes, transfers]` per period. The batches enter the L3
Beta-Binomial posterior only; they do not change the L7 headline or its range.

## Privacy

This file stays on the clinic's own machine. Nothing in this pipeline
transmits it anywhere: the models run locally and the narrative layer talks
only to a local Ollama instance. See [SECURITY.md](../SECURITY.md).
