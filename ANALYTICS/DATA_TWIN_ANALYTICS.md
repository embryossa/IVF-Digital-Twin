# IVF Digital Twin — Prediction vs. Reality Analytics

> **Repository module:** `Data_Twin_analytics.ipynb`
> **Purpose:** Systematic, multi-layer validation of the IVF Digital Twin (DT) predictive pipeline against real clinical cycle outcomes.

---

## Table of Contents

1. [Overview](#overview)
2. [Input Files](#input-files)
3. [OPU Table — Data Entry Guide](#opu-table--data-entry-guide)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Section-by-Section Reference](#section-by-section-reference)
   - [Section 1 — Data Loading & Merge](#section-1--data-loading--merge)
   - [Section 2 — Cohort Input Parameters](#section-2--cohort-input-parameters)
   - [Section 3 — Funnel: MC-Medians vs. Reality](#section-3--funnel-mc-medians-vs-reality)
   - [Section 4 — Pregnancy Predictions: All Models](#section-4--pregnancy-predictions-all-models)
   - [Section 5 — KAT vs. PRAI Comparison](#section-5--kat-vs-prai-comparison)
   - [Section 6 — Calibration & Quality Metrics](#section-6--calibration--quality-metrics)
   - [Section 7 — Model Concordance Heatmaps](#section-7--model-concordance-heatmaps)
   - [Section 8 — Cluster Analysis](#section-8--cluster-analysis)
   - [Section 9 — Risk Metrics: OHSS, Cancellation, Banking](#section-9--risk-metrics-ohss-cancellation-banking)
   - [Section 10 — Accuracy by Age Group & Conversion Rates](#section-10--accuracy-by-age-group--conversion-rates)
   - [Section 11 — Auto-Interpretation & Medical Review List](#section-11--auto-interpretation--medical-review-list)
   - [Section 12 — Patient Similarity Graph](#section-12--patient-similarity-graph)
6. [Output Report: DT\_Analytics\_Report.xlsx](#output-report-dt_analytics_reportxlsx)
7. [Metrics Glossary](#metrics-glossary)
8. [Visualization Style Guide](#visualization-style-guide)
9. [Dependencies](#dependencies)

---

## Overview

This notebook is the **validation and analytics layer** of the IVF Digital Twin system. After the DT pipeline generates per-patient predictions (Monte Carlo simulation, Bayesian layers, and ML classifiers), this notebook compares those predictions against actual clinical outcomes recorded in the OPU table.

The analysis covers two complementary dimensions:

| Dimension | What it answers |
|---|---|
| **Embryological funnel accuracy** | Did the model predict the right number of oocytes, blastocysts, and top-quality embryos? |
| **Pregnancy probability calibration** | Are the model's probability estimates well-calibrated — does "60%" actually correspond to ~60% real-world implantation rates? |

The notebook is **not** a generic statistics report. Every metric, chart, and flag is designed specifically for IVF outcome analysis, where small sample sizes (often 10–40 patients per cohort) require robust non-parametric methods and careful uncertainty quantification via Monte Carlo simulation intervals.

---

## Input Files

| File | Description |
|---|---|
| `dt_predictions.csv` | CSV export from the Digital Twin pipeline. Contains all per-patient model predictions: MC medians, confidence intervals, cluster assignments, and all model probability scores. |
| `OPU table.xlsx` | Clinical records table filled manually after each retrieval cycle. Contains the ground-truth outcomes to compare against DT predictions. See the [OPU Table Guide](#opu-table--data-entry-guide) below. |

The two files are joined on `patient_id` (from the DT CSV) ↔ `ID` (from the OPU table), with both keys normalized to uppercase stripped strings to prevent merge failures.

---

## OPU Table — Data Entry Guide

The `OPU table.xlsx` is the **source of ground truth** for all validation analyses. It must be filled in after each patient's cycle is complete. Below is a column-by-column guide.

### Required Columns

| Column | Type | Description | How to fill |
|---|---|---|---|
| `Patient Full Name` | Text | Full name (Surname, Given name) | As in the clinic's EMR. Used for display only. |
| `DOB` | Date | Date of birth | `DD.MM.YYYY` format. Used to verify the `Age` field. |
| `ID` | Text | Unique patient identifier | **Must match exactly** the `patient_id` field in `dt_predictions.csv`. This is the join key. Case-insensitive. |
| `Date OPU` | Date | Date of the oocyte pick-up procedure | `DD.MM.YYYY`. Used for cohort timeline tracking. |
| `Date ET` | Date | Date of embryo transfer | `DD.MM.YYYY`. Leave blank if no transfer occurred. |
| `BMI` | Numeric | Body mass index at cycle start | kg/m². One decimal place sufficient. |
| `AMH` | Numeric | Anti-Müllerian hormone level | pmol/L or ng/mL — be consistent with what was used as DT model input. |
| `Attempt` | Integer | Sequential IVF attempt number for this patient | `1` = first cycle, `2` = second, etc. |
| `AFC` | Integer | Antral follicle count from the baseline ultrasound | Total bilateral count. |
| `Age` | Integer | Patient age at OPU date | Calculated from DOB at the time of OPU. |

### Embryological Outcome Columns

These are the core measurements from the cycle laboratory, filled after the retrieval day and during embryo culture.

| Column | Type | Description | When to fill |
|---|---|---|---|
| `N folicules OPU` | Integer | Total number of follicles aspirated during OPU | Day of OPU. |
| `OCC` | Integer | Oocytes collected (all, including degenerate/immature) | Day of OPU. This is what the DT predicts as `med_okk`. |
| `MII` | Integer | Mature oocytes (metaphase II) — ready for insemination | Day of OPU / Day 0. This is what the DT predicts as `med_mii`. |
| `Inseminated` | Integer | Number of oocytes inseminated (ICSI or conventional) | Day 0. |
| `2pN` | Integer | Number of normally fertilized zygotes (two pronuclei) | Day 1. This is what the DT predicts as `med_pn2`. |
| `Cleavage` | Integer | Number of embryos at cleavage stage (Day 2–3) | Day 2 or 3. |
| `Bl` | Integer | Total number of blastocysts reached (any grade) | Day 5 or 6. This is what the DT predicts as `med_blasts`. |
| `Good Bl` | Integer | Blastocysts of good quality (e.g., AA, AB, BA by Gardner grading) | Day 5 or 6. This is what the DT predicts as `med_good`. |
| `Cryo` | Integer | Number of blastocysts vitrified (frozen) | Day 5–7. Relevant for the banking strategy analysis. |
| `ET` | Integer | Number of embryos transferred | Day of transfer. |
| `Day of ET` | Integer | Day of embryo transfer relative to fertilization | Typically `5` or `6` for blastocyst transfer, `3` for cleavage stage. |

### Clinical Outcome Columns

| Column | Type | Description | When to fill |
|---|---|---|---|
| `Preg` | Binary (0 / 1) | Pregnancy outcome after transfer | Fill **only after** the serum hCG test (approx. 2 weeks post-ET). `1` = positive hCG, `0` = negative. Leave **blank** if the outcome is not yet known — do not fill with 0 prematurely. |
| `PRAI` | Numeric (0–1) | Pregnancy probability from the PRAI model (external full-cycle scoring system) | Fill with the PRAI probability score for this patient's cycle, expressed as a decimal (e.g., `0.62`). This is the external reference used to validate the DT's own KAT model. |
| `DIGITAL TWIN` | Numeric (0–1) | Final DT pregnancy probability as displayed in the DT clinical interface | Cross-check column — fill from the DT interface to confirm the CSV export matches what was shown to the physician. |

### Important Notes

- **Do not pre-fill `Preg = 0`** for patients still waiting for their hCG result. The analysis separates patients with known outcomes (`df_preg`) from those without. Premature `0` entries will corrupt calibration metrics.
- **The `ID` column is case-insensitive** but must be free of leading/trailing spaces. Extra whitespace is the most common merge failure cause.
- **All numeric columns** should use period (`.`) as the decimal separator, not comma.
- If a cycle was **cancelled before OPU**, still add the row with the known parameters and leave embryological columns blank — the row will appear in input demographics but be excluded from funnel accuracy analysis automatically.

---

## Pipeline Architecture

The Digital Twin pipeline that this notebook validates consists of multiple layers. Understanding these layers is essential to interpreting the analytics:

```
Patient inputs (Age, AFC, AMH, ...)
        │
        ▼
┌─────────────────────────────┐
│  Monte Carlo (MC) Simulation │  → 10,000 stochastic cycle runs
│  Outputs: med_okk, med_mii,  │    p025/p975 confidence intervals
│  med_pn2, med_blasts,        │    for each embryological stage
│  med_good                    │
└──────────────┬──────────────┘
               │  p_per_transfer  (MC median pregnancy probability)
               ▼
┌─────────────────────────────┐
│  Bayesian Layer             │  → bayes_mean (Bayesian-adjusted probability)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  KAT Model                  │  → p_kat_raw + ci_kat_low/high
│  (simulated transfer model) │     Estimates implantation from
│                             │     embryo quality & patient profile
└──────────────┬──────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌──────────────┐  ┌──────────────┐
│  NVSA model  │  │  CSDI model  │
│  (p_nvsa)    │  │  (p_csdi)    │
└──────────────┘  └──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  DIGITAL TWIN (final score) │  → Ensemble / final clinical output
└─────────────────────────────┘

External reference:
┌─────────────────────────────┐
│  PRAI                       │  → Full-cycle real-world scoring
│  (from OPU table)           │     Used as ground-truth comparator
└─────────────────────────────┘

Clustering:
┌─────────────────────────────┐
│  Patient cluster assignment │  C0: Standard responders
│  (dominant_cluster)         │  C1: Poor responders
│                             │  C2: High responders
└─────────────────────────────┘
```

---

## Section-by-Section Reference

### Section 1 — Data Loading & Merge

**What it does:**
Loads `dt_predictions.csv` and `OPU table.xlsx`, normalizes the join keys, and produces three dataframes used throughout the notebook:

| Variable | Description |
|---|---|
| `df` | All merged rows, including duplicate entries for patients with multiple calculations |
| `df_u` | Deduplicated: one row per patient, keeping the most recent DT calculation (`timestamp` descending) |
| `df_preg` | Subset of `df_u` where `Preg` is not null — patients with known pregnancy outcomes |

**Console output:**

```
DT records:             [total rows in CSV]
OPU rows:               [rows in OPU table]
Matched by ID (all):    [rows successfully joined]
Unique patients:        [rows in df_u]
With Preg outcome:      [N] (+[positive] / -[negative])
```

**What to check:** If "Matched by ID" is significantly less than "OPU rows," there are ID mismatches. Verify leading/trailing spaces and confirm that the `patient_id` values in the CSV exactly correspond to the `ID` column in the OPU table.

---

### Section 2 — Cohort Input Parameters

**Charts produced:** Three side-by-side histograms with KDE overlay and mean line.

| Subplot | X-axis | What it shows |
|---|---|---|
| Age distribution | Patient age (years) | Age spread of the current cohort. Important context — older cohorts will have systematically lower real outcomes than the model may expect if it was calibrated on a younger population. |
| Follicle count (ТВП / AFC) | `N folicules OPU` | Distribution of ovarian reserve in the cohort. Bimodal distributions may indicate mixed poor/high-responder groups. |
| MII oocytes | `MII` | Mature oocyte yield distribution. |

Each histogram shows:
- **Gray bars**: patient distribution
- **Dashed KDE curve**: smoothed density estimate
- **Vertical dashed line + μ annotation**: cohort mean

**Why this matters:** These charts document what was *input* to the model. If the cohort is extreme (e.g., all patients > 40, or all AFC < 5), poor funnel accuracy is expected and contextualizes downstream errors.

---

### Section 3 — Funnel: MC-Medians vs. Reality

This is the core embryological validation section. It produces two independent chart groups.

#### Cell A — Oocytes: OCC and MII

**Charts:** Two scatter plots side by side (OCC, MII).

Each scatter plot shows:
- **X-axis**: Real outcome (from OPU table)
- **Y-axis**: MC model median prediction
- **Dashed gray diagonal**: perfect prediction line (predicted = real)
- **Shaded band around diagonal**: ±mean SD of the Monte Carlo interval, derived from the 2.5th–97.5th percentile CI of the MC simulation. This band represents the **model's inherent uncertainty**, not error — a point inside the band is within the simulation's expected variability.
- **Dotted trendline**: linear regression of predicted vs. real values
- **Error bars on each point**: SD derived from the MC confidence interval for that patient (`SD ≈ (p975 − p025) / (2 × 1.96)`)
- **Annotation box** (top-left of each plot):

```
r = [Pearson]  |  ρ = [Spearman]
MAE = [mean absolute error]  |  Bias = [mean signed error]
Mean SD = [average MC uncertainty]
```

**Metric interpretation:**

| Metric | Formula | Interpretation |
|---|---|---|
| Pearson r | Standard correlation | How linearly correlated predictions and reality are. r > 0.7 indicates strong tracking. |
| Spearman ρ | Rank correlation | More robust to outliers. Should be close to Pearson r. Large divergence suggests outliers drive the linear correlation. |
| MAE | Mean \|predicted − real\| | Average prediction error in absolute units (oocytes). |
| Bias | Mean (predicted − real) | Systematic over- (+) or under-prediction (−). A positive bias means the model consistently overestimates counts. |
| Mean SD | Average MC interval width ÷ 3.92 | Represents the average uncertainty the model reports. If MAE ≈ Mean SD, the model is well-calibrated in uncertainty. If MAE >> Mean SD, the model is overconfident. |

#### Cell B — Embryology: 2PN, Blastocysts, Good-Quality Blastocysts

**Charts:** Three scatter plots (2PN, total blastocysts, good-quality blastocysts).

Same structure as Cell A, with one additional feature:

**Color-coded points by ±30% accuracy threshold:**

| Color | Meaning |
|---|---|
| 🟢 Green | Prediction within ±30% of real value — clinically acceptable |
| 🔴 Red | Prediction error > 30% of real value — outlier, warrants review |
| ⚫ Gray | Real value = 0 (cannot compute relative error) |

**Additional annotation:** `✅ ±30%: X/N` — count of patients whose prediction fell within the ±30% corridor.

**The ±30% threshold** is a domain-specific convention for embryological count prediction. In IVF, a prediction of 3 blastocysts when 4 were obtained (25% error) is considered clinically acceptable. A prediction of 6 when 2 were obtained (200% error) is a significant miss.

In addition to the SD band, Cell B also shows a **dotted ±30% corridor** (lines at y = 1.3x and y = 0.7x), giving a visual reference for the acceptable accuracy zone.

---

### Section 4 — Pregnancy Predictions: All Models

**Chart:** Box plots with strip (jitter) overlay for each model.

**Models shown** (if present in the data):

| Model | Column | Description |
|---|---|---|
| MC | `p_per_transfer` | Raw Monte Carlo simulation probability of pregnancy per transfer |
| Bayes | `bayes_mean` | Bayesian-adjusted probability |
| KAT | `p_kat_raw` | Simulated transfer model based on embryo quality |
| NVSA | `p_nvsa` | NVSA sub-model output |
| CSDI | `p_csdi` | CSDI sub-model output |
| DT | `DIGITAL TWIN` | Final ensemble score shown in the DT interface |
| PRAI | `PRAI` | External full-cycle real-world reference score (from OPU table) |

Each box shows median, IQR, whiskers, mean (dot), and standard deviation. Individual patient values are overlaid as scatter points.

**Horizontal reference line at P = 0.5** — the clinical decision threshold.

**Descriptive statistics table** is printed below the chart: count, mean, std, min, 25%, 50%, 75%, max for each model.

**Purpose:** This chart reveals whether different models in the pipeline tend to be systematically higher or lower than each other, and whether the DT ensemble aligns with external PRAI scores.

---

### Section 5 — KAT vs. PRAI Comparison

This section performs a detailed head-to-head comparison between the **KAT model** (the DT's simulated transfer model) and **PRAI** (an external full-cycle scoring system, treated as an independent clinical benchmark).

**Two charts:**

#### Left: KAT vs. PRAI Scatter

- **X-axis**: PRAI score (external reference, 0–1)
- **Y-axis**: KAT score (DT pipeline, 0–1)
- **Points color-coded by cluster** (C0 Standard / C1 Poor / C2 High)
- **Error bars**: ±SD of KAT from its Monte Carlo confidence interval
- **Shaded band**: ±mean SD of KAT along the diagonal
- **Trendline**: linear regression
- **Annotation box**: r, ρ, MAE, Bias, N

#### Right: Distribution of Differences (KAT − PRAI)

- Histogram of `p_kat_raw − PRAI` per patient
- KDE curve (normal approximation) overlaid
- Vertical markers:
  - Red dashed at `0` (no difference)
  - Orange dotted at `bias` value
  - Gray dotted at `bias ± 1SD`
- Shaded zone: ±1 SD around bias

**Statistical test:** If N ≥ 10, the **Wilcoxon signed-rank test** is applied:
- H₀: KAT and PRAI are drawn from the same distribution (no systematic difference)
- p < 0.05 → ⚠️ Statistically significant systematic discrepancy
- p ≥ 0.05 → ✅ No significant discrepancy detected

**Why KAT vs. PRAI?** KAT is a simulated prediction derived from embryo quality parameters and patient profile alone. PRAI incorporates the full clinical picture of the actual transfer cycle. Agreement between them validates that the DT's pre-cycle estimate closely matches what a full post-cycle assessment would conclude. Divergence — especially systematic bias — may indicate that the DT over- or under-weights specific factors.

---

### Section 6 — Calibration & Quality Metrics

This section evaluates how well each pregnancy model actually performs on patients with known outcomes (`df_preg`).

**Requires:** Patients with `Preg` column filled (0 or 1).

#### Metrics Table

For each model, the following are computed:

| Metric | Direction | Description |
|---|---|---|
| Brier Score | ↓ lower is better | Mean squared error between predicted probability and binary outcome. A perfect model = 0; uninformative model = 0.25. |
| AUC-ROC | ↑ higher is better | Area under the ROC curve. Measures discrimination — how well the model separates patients who become pregnant from those who don't. AUC = 0.5 is random; AUC = 1.0 is perfect. |
| Acc@0.5 | ↑ higher is better | Accuracy when using 0.5 as the decision threshold (predict pregnancy if score ≥ 0.5). |
| Mean pred | — | Average predicted probability across the cohort. Compare to Real rate to detect systematic over/under-prediction. |
| Real rate | — | Observed pregnancy rate in the cohort with known outcomes. |

#### Violin Plots by Outcome

For the top 4 models, violin + box plots split by outcome (Pregnant / Not Pregnant):
- Shows whether higher predicted probabilities actually correspond to the pregnant group
- Good discrimination = two clearly separated distributions

---

### Section 7 — Model Concordance Heatmaps

**Charts:** Two correlation heatmaps (Pearson r, Spearman ρ) showing pairwise correlations between all probability models.

**Purpose:** Reveals the internal consistency of the DT pipeline. Questions answered:

- Do MC, KAT, NVSA, CSDI, and DT all agree on which patients have high/low probability?
- Does PRAI (external reference) correlate well with the DT's own models?
- If two models designed to measure the same thing show low correlation, there may be a pipeline bug or data mismatch.

**How to read:** Cell (X, Y) shows the correlation between model X and model Y. Values close to 1.0 = strong agreement; values close to 0 = independent estimates; negative values (rare) = models disagree on patient ranking.

---

### Section 8 — Cluster Analysis

The DT assigns each patient to one of three clusters based on their predicted response profile:

| Cluster | Label | Typical profile |
|---|---|---|
| C0 | Standard | Average ovarian reserve, typical response, moderate pregnancy probability |
| C1 | Poor | Low AFC/AMH, diminished reserve, low expected yield |
| C2 | High | High AFC, high yield, elevated OHSS risk |

This section validates whether cluster assignments correspond to meaningful differences in real outcomes.

**Charts include:**
- Real embryological outcomes (OCC, MII, blastocysts) by cluster — box plots with jitter
- Predicted vs. real comparisons stratified by cluster
- Cluster distribution in the cohort (bar chart)
- Pregnancy rate by cluster (if outcomes are available)

---

### Section 9 — Risk Metrics: OHSS, Cancellation, Banking

This section analyzes three DT-specific risk and strategy metrics:

#### OHSS Risk (`ohss_risk`)

Ovarian Hyperstimulation Syndrome probability — a serious complication of IVF stimulation. The DT provides a per-patient risk estimate. This section shows:
- Distribution of OHSS risk scores in the cohort
- Correlation with AFC and total follicle count (higher AFC → typically higher OHSS risk)
- Flag when `ohss_risk > 0.3` (threshold for clinical concern)

#### Cancellation Probability (`p_cancel`)

The DT estimates the probability that a cycle will be cancelled (e.g., due to poor response or OHSS risk). Validation here shows:
- Were patients flagged as high-cancellation risk actually cancelled?
- Distribution across clusters

#### Banking Strategy (`banking`)

For patients with `Good Bl > 0`, the DT may recommend a freeze-all banking strategy vs. fresh transfer. This section examines whether the `Cryo` column in the OPU table reflects the recommended strategy, and whether banking patients had higher cumulative success.

---

### Section 10 — Accuracy by Age Group & Conversion Rates

#### Age-Stratified Accuracy

MAE and Bias for embryological predictions (OCC, blastocysts, good-quality blastocysts) broken down by age group:

| Age group | Rationale |
|---|---|
| < 35 | Typically best prognosis; model should be most accurate here |
| 35–37 | Intermediate — beginning of age-related decline |
| 38–40 | Accelerated follicle pool decline |
| > 40 | Poorest prognosis; highest prediction uncertainty |

#### Conversion Rates

Observed conversion ratios at each embryological stage, compared to the DT's predicted conversion rates:

| Conversion | Formula | DT prediction |
|---|---|---|
| OCC → MII | MII / OCC | `med_mii / med_okk` |
| MII → 2PN | 2PN / MII | `med_pn2 / med_mii` |
| 2PN → Blastocyst | Bl / 2PN | `med_blasts / med_pn2` |
| Blastocyst → Good | Good Bl / Bl | `med_good / med_blasts` |

If the DT systematically overestimates a particular conversion step, this pinpoints where in the embryological chain the model has a bias.

---

### Section 11 — Auto-Interpretation & Medical Review List

This section auto-generates a structured clinical report and a **priority list of patients for medical case review**.

#### Auto-Interpretation Logic

Each patient is assigned a priority flag:

| Priority | Label | Criteria |
|---|---|---|
| 1 | 🔴 Critical review | One or more critical flags: false positive (predicted pregnant, outcome negative with MC > 60%), or predicted blastocyst yield > 3× real yield |
| 2 | 🟡 Review recommended | Moderate discrepancy: funnel accuracy > 30% off on ≥ 2 stages, or KAT−PRAI divergence > 20 percentage points |
| 3 | 🟢 Within expectation | All predictions within ±30%, no major discrepancies |

#### HTML Dashboard Output

Rendered inline in the notebook — a styled clinical report showing:
- Cohort summary statistics
- Per-patient summary rows (name, ID, age, all model probabilities, outcome, cluster, flags)
- Color-coded outcome badges
- Summary of flag types across the cohort

#### Review List Table

A filtered table containing only Priority 1 and 2 patients, with columns:

| Column | Description |
|---|---|
| Пациент | Patient name |
| ID | Patient identifier |
| Возраст | Age |
| Приоритет | Priority (1 = highest) |
| Основание | Reason for review (specific flag description) |

This table is suitable for direct clinical use — it is the "handoff document" from the analytics pipeline to the medical team.

---

### Section 12 — Patient Similarity Graph

This is the most structurally complex section. It builds a **patient similarity network** using cosine similarity on a combined feature vector of DT inputs, predictions, and (if available) real outcomes.

#### Block A — Graph Construction

**Feature vector per patient** (standardized with `StandardScaler`):
- Age, AMH, AFC
- MC predictions (OCC, MII, blastocysts, good blastocysts)
- Pregnancy model scores (MC, KAT, NVSA, CSDI, DT)
- Cluster assignment

**Edge creation:** A graph edge is drawn between two patients if their cosine similarity exceeds a threshold (typically 0.70). Edge weight = cosine similarity value.

**Node attributes stored:**
- Cluster membership
- Pregnancy outcome (if known)
- Degree centrality
- Betweenness centrality

#### Block B — Graph Visualization

Interactive Plotly network graph:
- **Node size**: proportional to degree centrality (more connections = more "typical")
- **Node color**: cluster (C0 blue / C1 red / C2 green)
- **Node border**: outcome (thick = known outcome, thin = unknown)
- **Edge color/opacity**: proportional to similarity weight
- **Isolated nodes**: patients with no neighbors above the similarity threshold — atypical cases

#### Block C — Similar Cases Lookup

For each **high-priority patient** (false positives and high model uncertainty), the system finds the top-5 most similar patients from the graph who have a known pregnancy outcome.

**Output per patient:**
- Header card: patient name, MC probability, model SD (uncertainty), outcome badge
- Count of similar patients with outcomes: "X pregnant / Y not pregnant"
- Table of 5 nearest neighbors with their key parameters and outcomes

**Clinical use:** If the model predicted 70% for a patient who did not become pregnant, and 4 out of 5 similar historical patients also did not become pregnant, this provides retrospective clinical context — the model may be over-optimistic for this patient profile.

#### Block D — Graph Structure Report

Three statistical charts:
1. **Degree distribution histogram** — how many connections each patient has. A heavy right tail indicates the cohort contains "typical" patients that are similar to many others.
2. **Betweenness centrality bar chart** (top 10) — patients who act as "bridges" between subgroups. High betweenness = clinically interesting intermediate profiles.
3. **Intra- vs. inter-cluster similarity box plots** — mean edge weight within clusters vs. between clusters. If intra-cluster similarity >> inter-cluster, the clustering is well-separated.

**Statistical test:** Mann-Whitney U test comparing intra-cluster vs. inter-cluster edge similarities:
- H₀: intra- and inter-cluster similarities are equal
- H₁: intra-cluster similarity > inter-cluster (one-tailed)
- Reports: U statistic, p-value, rank-biserial r (effect size), Cohen's d, bootstrap 95% CI for medians

**Auto-generated methods paragraph** in journal format, suitable for inclusion in a publication Methods section.

---

## Output Report: DT\_Analytics\_Report.xlsx

The notebook saves all analysis outputs to `DT_Analytics_Report.xlsx` with three sheets:

### Sheet 1: Summary

One row per unique patient. This is the primary clinical reference sheet.

| Column | Description |
|---|---|
| Пациент | Patient name (truncated to 25 chars) |
| ID | Normalized patient key |
| Возраст | Age |
| OCC (р/п) | OCC: real / predicted (e.g., "8 / 9") |
| OCC | Flag: ✅ within ±30% of real, ⚠️ outside ±30%, — if data missing |
| Bl (р/п) | Blastocysts: real / predicted |
| Bl | Flag for blastocyst accuracy |
| MC% | Monte Carlo pregnancy probability (as %) |
| KAT% | KAT model probability (as %) |
| PRAI% | External PRAI score (as %) |
| DT% | Final Digital Twin score (as %) |
| KAT−PRAI | Signed difference: KAT minus PRAI in percentage points (e.g., "+12.3 пп" or "−5.1 пп") |
| Кластер | Assigned cluster: C0 Standard / C1 Poor / C2 High |
| Исход | Pregnancy outcome: ✅ / ❌ / — (unknown) |

### Sheet 2: FullData

All columns from `df_u` (the merged, deduplicated dataframe), minus internal `_*` computation columns. This is the raw analytical dataset for anyone who wants to perform additional analyses in Excel or export to another system.

### Sheet 3: WithOutcomes

Subset of FullData containing only patients with a known pregnancy outcome (`Preg` is not null). This is the dataset used for all calibration and quality metric calculations. Useful for building historical calibration datasets over time.

---

## Metrics Glossary

| Term | Definition |
|---|---|
| **MC median** | The median outcome from 10,000 Monte Carlo simulation runs for a given patient. Reported as `med_okk`, `med_mii`, `med_pn2`, `med_blasts`, `med_good`. |
| **MC CI (p025 / p975)** | The 2.5th and 97.5th percentiles of the Monte Carlo distribution — the 95% simulation interval. Not a confidence interval in the frequentist sense; represents the stochastic range of plausible outcomes given model uncertainty. |
| **SD (from MC CI)** | Derived standard deviation: `(p975 − p025) / (2 × 1.96)`. Used as the error bar on scatter plots. |
| **Bias** | Mean signed error: `mean(predicted − real)`. Positive bias = systematic overestimation. Negative bias = systematic underestimation. |
| **MAE** | Mean Absolute Error: `mean(|predicted − real|)`. Unsigned average error — does not cancel over/under-prediction errors. |
| **Pearson r** | Linear correlation coefficient. Sensitive to outliers. |
| **Spearman ρ** | Rank correlation coefficient. Robust to outliers and non-linear monotonic relationships. |
| **Brier Score** | Proper scoring rule for probability predictions: `mean((p̂ − y)²)`. Range 0 (perfect) to 1 (worst). For a 50/50 prevalence cohort, a naive model that always predicts 0.5 scores 0.25. |
| **AUC-ROC** | Area Under the Receiver Operating Characteristic Curve. Measures discrimination ability. 0.5 = random; 1.0 = perfect; < 0.5 = systematically wrong direction. |
| **Calibration** | Whether predicted probabilities match observed frequencies — if a model predicts 70% for 10 patients, ~7 of them should actually become pregnant. |
| **±30% corridor** | Domain-specific accuracy threshold: a prediction is considered acceptable if |predicted − real| / real ≤ 0.30. Standard for embryological count validation in IVF prediction literature. |
| **KAT** | Simulated transfer model. Predicts pregnancy probability from embryo quality metrics and patient profile, without requiring an actual transfer to have occurred. |
| **PRAI** | External full-cycle pregnancy probability score calculated after the transfer cycle is complete. Used as the ground-truth reference for pregnancy probability calibration. |
| **Cosine similarity** | Similarity measure between two patient feature vectors: `cos(θ) = (A · B) / (‖A‖ × ‖B‖)`. Range 0 to 1. Value of 1 means identical normalized profiles. |
| **Betweenness centrality** | Graph metric measuring how often a node lies on the shortest path between two other nodes. High betweenness = clinical "bridge" patient whose profile connects different subgroups. |
| **Degree centrality** | Number of graph neighbors normalized by maximum possible connections. High degree = very "typical" patient, similar to many others in the cohort. |
| **Rank-biserial r** | Non-parametric effect size for Mann-Whitney U: `r = 1 − 2U/(n₁×n₂)`. |0.1| = small, |0.3| = medium, |0.5| = large effect. |

---

## Visualization Style Guide

All charts use a consistent visual language:

| Element | Style |
|---|---|
| Font | Inter / Arial, 12pt, color `#2c3e50` |
| Background | Plot area: `#F7F9FC`; Paper: white |
| Primary colors | Blue `#1B4F72`, Red `#C0392B`, Green `#1E8449`, Orange `#D68910`, Purple `#7D3C98`, Gray `#717D7E`, Teal `#148F77` |
| Transparency | All fill colors use alpha 0.08–0.70. Scatter markers: 0.65–0.70. Error bars: 0.40–0.45. |
| Error bars | Derived from MC confidence intervals (`±SD`). Color matches series but at 0.40–0.45 alpha. Width = 4px, thickness = 1.5px. |
| Diagonal reference | Dashed gray `rgba(150,150,150,0.7)`, width 1.5 |
| Trendline | Dotted, series color at 0.65 alpha, width 2.0–2.2 |
| Annotation boxes | White background 0.85 alpha, `#ccc` border, 10pt Inter font |
| Legend position | Horizontal, `y = 1.04` (above chart) or `y = −0.12` (below) |
| Hatch patterns | MC/model series bars use `/` hatch; real data bars use solid fill |
| Point color coding | Green = within ±30% threshold; Red = outside ±30%; Gray = zero denominator |

---

## Dependencies

```python
pandas
numpy
scipy
scikit-learn          # roc_auc_score, brier_score_loss, calibration_curve
plotly                # go, px, make_subplots
networkx              # patient similarity graph
IPython.display       # HTML inline rendering
openpyxl / calamine   # Excel I/O
xlsxwriter            # Excel output (preferred; falls back to openpyxl)
```

Install all at once:
```bash
pip install pandas numpy scipy scikit-learn plotly networkx openpyxl xlsxwriter
```
