# Clinic Data Schema

The intake schema for clinic-specific calibration
(`validate_clinic_data.py` → `calibrate_for_clinic.py`).

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
| `sperm_source` | category | e.g. ejaculate / TESA / donor |

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
| `mii` | int | mature oocytes |
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
| `outcome` | category | the endpoint you are calibrating against |
| `outcome_date` | date | leave blank if it would narrow identification |
| `outcome_known` | bool | **false for cycles still in follow-up.** Rows with `outcome_known = false` are excluded from calibration rather than counted as failures — treating pending cycles as negatives is the most common way clinic recalibration goes wrong. |

## Volume

| Rows | What you get |
|---|---|
| < 100 | Validation runs; calibration barely moves the literature prior. Expected — the Bayesian design is doing its job. |
| 100–300 | Beta-Binomial prior shifts meaningfully toward your population. |
| > 500 | Reliable clinic-specific recalibration, including per-phenotype. |

## Workflow

```bash
python validate_clinic_data.py --input your_cycles.csv    # schema, ranges, missingness
python calibrate_for_clinic.py --input your_cycles.csv    # refit priors
```

`validate_clinic_data.py` reports impossible funnels (`mii > okk`,
`blasts_good > blasts_total`), out-of-range values, and missingness per column.
Fix what it reports before calibrating — a funnel violation usually means a
column mapping error, and calibrating on it will quietly bias every downstream
layer.

Then copy `clinic_config.template.json` to `clinic_config.json` and fill
`batches` with `[successes, transfers]` per period.

## Privacy

This file stays on the clinic's own machine. Nothing in this pipeline
transmits it anywhere: the models run locally and the narrative layer talks
only to a local Ollama instance. See [SECURITY.md](../SECURITY.md).
