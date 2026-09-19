<div align="center">

<img src="logo22.png" width="96" alt="IVF Digital Twin">

# IVF Digital Twin v7.1

**An Integrated Multi-Source Ensemble Platform for Stage-Stratified IVF Outcome Prediction**

*from in vitro to in silico*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: PolyForm NC 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue.svg)](LICENSE)
[![Commercial licence](https://img.shields.io/badge/commercial%20licence-available-green.svg)](COMMERCIAL-LICENSE.md)
[![CI](https://github.com/embryossa/IVF-Digital-Twin/actions/workflows/ci.yml/badge.svg)](https://github.com/embryossa/IVF-Digital-Twin/actions/workflows/ci.yml)
[![Status: research](https://img.shields.io/badge/status-research%20prototype-orange.svg)](DISCLAIMER.md)
[![Not a medical device](https://img.shields.io/badge/⚠-not%20a%20medical%20device-red.svg)](DISCLAIMER.md)

*Sergeev et al., 2026* · [embryossa@gmail.com](mailto:embryossa@gmail.com) · [LinkedIn](https://www.linkedin.com/in/serdj-sergeev-8b5893298/)

</div>

> [!WARNING]
> **This is a research tool, not a medical device.** It is not registered,
> cleared or approved as one in any jurisdiction. Its output must never replace
> clinical judgment, and a qualified clinician must review every prediction
> before it informs a decision or reaches a patient. Read
> [DISCLAIMER.md](DISCLAIMER.md) before any clinical use.

---

## What this is

Clinical decision-support tools for IVF typically address one endpoint and
return a point estimate. They do not quantify the biological variability that
dominates reproductive medicine, cannot incorporate information as the cycle
unfolds, do not benchmark a patient against documented protocol phenotypes,
offer no independent verification of laboratory outcome distributions, and
provide no principled way to reconcile several models that disagree.

IVF Digital Twin models the whole treatment trajectory as a sequential
probabilistic pipeline. Seven layers run in series, each contributing a
distinct epistemic perspective, and a final Bayesian arbiter (BEFE, L7) fuses
them into one posterior with an explicit account of how much each source was
trusted and why.

**Outcome:** clinical pregnancy confirmed by ultrasound (ectopic included), not
live birth. The headline probability is **per transfer**; a separate cycle
probability covers the whole stimulation cycle.

Coefficients are traceable to their source — the published literature or a
documented local refit. See [`docs/coefficients.md`](docs/coefficients.md) and
`src/embryology.py`.

### What makes it different

| | Point-estimate calculators | **IVF Digital Twin v7.1** |
|---|---|---|
| Output | A single number | Full distributions at every cycle stage |
| Uncertainty | Absent or nominal | Monte Carlo scenarios, conformal intervals, model uncertainty range |
| Mid-cycle information | Ignored | Every entered count conditions the whole stage chain |
| Model disagreement | Hidden | Named, quantified, and attributed |
| Out-of-distribution patients | Silently extrapolated | Flagged on two independent feature subspaces |
| Auditability | Black box | Every coefficient sourced; every weight reported |

---

## What's new in 7.1

The computation logic is that of the 7.1 clinic build; the Streamlit interface
keeps its 7.0 layout. **Predictions differ from 7.0.** Full list in
[CHANGELOG.md](CHANGELOG.md).

- **L1 refitted on local data.** Oocyte yield: negative binomial with a log
  link (the linear ART-ONE predictor under-estimated low responders).
  Blastocyst yield: zero-inflated beta-binomial; quality: beta-binomial.
- **Exact conditioning.** Entered laboratory counts condition the whole chain
  (forward filtering / backward sampling) instead of overwriting one stage.
- **Transfer scenario.** Without PGT-A every blastocyst is transferable, good
  quality first; with PGT-A only euploid embryos. Transfers from one retrieval
  share a random effect, so each additional embryo adds less than under
  independence.
- **Model inputs match training.** KAT, GAT and CSDI receive features with
  their training definitions; the KAT isotonic calibrator is interpolated
  between step centres (no plateaus or jumps); FT-Transformer encoding no
  longer depends on the other rows of the batch.
- **CSDI applicability.** L5 enters fusion only inside its training domain.
- **Uncertainty range.** The interval next to the L7 probability is now the
  model uncertainty range, which narrows as observations arrive — not the
  clinic's historical corridor.
- **One data flow.** Screen, PDF, BEFE tab and analytics read one result
  ([`dt_bridge.py`](dt_bridge.py)); the TRP plan starts from the L1–L7 cycle
  probability.

---

## Table of contents

- [Validation results](#validation-results)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Repository map](#repository-map)
- [Model weights](#model-weights)
- [Clinical interpretation](#clinical-interpretation)
- [Privacy and security](#privacy-and-security)
- [Licensing](#licensing)
- [Citation](#citation)
- [References](#references)

---

## Validation results

Reported for the 7.0 model layers:

| Layer | Component | Metric | Value |
|---|---|---|---|
| L1 | Stochastic pipeline (7.0) | Spearman ρ (oocyte count) | 0.73 |
| L1 | Stochastic pipeline (7.0) | Spearman ρ (blastocyst count) | 0.41 |
| L5 | CSDI Hybrid v3 | AUC | 0.661 |
| L5 | CSDI Hybrid v3 | Brier | 0.209 |
| L5 | CSDI Hybrid v3 | ECE | 0.029 |
| L5 | CSDI Hybrid v3 | Prevalence bias | −0.1 pp |
| L6 | GAT alone | AUC-ROC | 0.632 |
| L3+L6 | GAT + KAT ensemble | AUC-ROC | 0.658–0.665 |
| L3+L6 | GAT + KAT ensemble | Brier | ~0.229 |
| — | Overall pregnancy prediction (7.0) | AUC | 0.63 |
| — | Overall pregnancy prediction (7.0) | Brier | 0.22 |

7.1 refits and checks:

| Component | Data | Result |
|---|---|---|
| L1 stage 1 — oocyte yield | Stimulated cycles, n = 687 with OCC ≥ 1 | Zero-truncated NB, 5-fold patient cross-validation; replaces ART-ONE (AFC < 5: predicted 1.4 vs observed 2.7 oocytes) |
| L1 stage 4 — blastocyst yield | protocols_15k, culture-observed, n = 7,919 | Zero-inflated beta-binomial |
| Fair-quality transfer | Day-5 single blastocyst transfers, n = 2,975 | Adjusted OR 0.51 (95% CI 0.35–0.75) vs good quality |
| Transfer frailty | Clinic exports with the outcome of each transfer from one retrieval | σ = 0.746 on the logit scale |
| KAT interpolated calibration | protocols_15k | Same fit as the step calibrator (Brier within 0.0001, log loss within 0.0004); plateaus and jumps removed |
| KAT after the FT-Transformer encoding fix | 1,606 validation rows of protocols_15k | Brier 0.1594, ECE 0.0279 |

The 7.1 figures are re-evaluations on the development cohorts, **not
independent external validation**. The combined 7.1 stack has not yet been
externally validated.

**Reading these honestly.** An AUC around 0.63–0.66 is typical of IVF outcome
prediction and reflects a genuine ceiling: much of the variance in whether a
transfer implants is not captured by any pre-transfer variable currently
measured. The contribution here is not discrimination — it is calibration,
distributional output, and explicit uncertainty attribution. A well-calibrated
40% that knows when it is unreliable is more useful in counselling than a
sharper number that does not.

CSDI Hybrid v3 was trained on 15,193 cycles with a 3-way split
(85% diffusion / 7.5% LightGBM / 7.5% conformal calibration). The GAT graph
holds 1,172 clinical protocols. Cluster centroids derive from 1,556 cycles.

---

## Architecture

```
Patient inputs: age · AMH · AFC · BMI · attempt number · sperm source
Optional:       follicles at puncture · OCC · MII · 2PN · blastocysts · good · euploid
                          │
 L1  Stochastic Monte Carlo pipeline (N = 2,000 by default)
     NB oocyte model → binomial MII/2PN → ZIBB blastocysts → BB quality
     exact conditioning on entered counts · transfer scenario (PGT-A or not)
                          │
 L2  Per-transfer ensemble — FORTUNE + KPIScore
     per-transfer · if transferable · whole-cycle probability
                          │
 L3  KAT neural ensemble — KAN (B-spline) + FT-Transformer
     interpolated isotonic calibration · Beta-Binomial clinic posterior
                          │
 L4  Unsupervised cluster classifier
     nearest centroid in 18-dim space, k = 3 phenotypes
                          │
 L5  CSDI Hybrid v3 diffusion module
     CSDI Transformer + LightGBM + split conformal
     applicability check → consistency check of L1
                          │
 L6  GAT graph attention transformer
     patient-similarity graph · 1,172 protocols
                          │
 L7  BEFE — Bayesian Evidence Fusion Engine
     prior (L1) → evidence (L3·L6) → posterior, on scenarios with a transfer
     trust-weighted logit pooling · reliability · dual OOD
                          │
 OUTPUT  per-transfer probability · model uncertainty range · reliability
         cycle probability · TRP plan · PDF clinical report
```

<details>
<summary><b>L1 — Stochastic Monte Carlo pipeline</b></summary>

Current coefficients live in `src/embryology.py`.

| Stage | Model (7.1) | Data / source |
|---|---|---|
| 1. Retrieved oocytes | Negative binomial, log link on ln(AMH+0.1), ln(AFC+1), age; θ = 5.77; no separate structural-zero component | Stimulated cycles, n = 687, 5-fold patient CV |
| 2. Mature (MII) | Binomial, logistic maturity rate | — |
| 3. Two-pronuclear (2PN) | Binomial, logistic fertilisation rate | — |
| 4. Blastocysts | Zero-inflated beta-binomial, mean depends on age and 2PN count | protocols_15k, n = 7,919 |
| 5. Good-quality blastocysts | Beta-binomial, conditional on the blastocyst count | protocols_15k |
| 6. Euploid embryos | Age-stratified euploidy, Beta-distributed | Franasiak et al. 2014 |
| 6b. Warming survival | 95% per blastocyst | Coello et al. 2021 |

**Observations.** An entered count is not simply substituted for its stage:
unknown earlier stages are sampled from their conditional distribution given
it, and later stages stay stochastic. A blank field means *unknown*; zero means
*observed absence*.

**Transfer scenario.** Without PGT-A all blastocysts are transferable — good
quality first, then fair quality with the fair-vs-good odds ratio. Entering a
euploid count switches to PGT-A, where only euploid embryos are transferable.
Transfers from one retrieval share a random effect (σ = 0.746 on the logit
scale): the per-transfer probability is unchanged, but each additional embryo
adds less to the cycle probability than under independence.

OHSS probability and empty-cycle risk are computed per iteration, not
post-hoc.
</details>

<details>
<summary><b>L2 — Per-transfer ensemble</b></summary>

Two independent probability sources combined on the logit scale:

- **FORTUNE** — logistic regression on clinical predictors (Carrasquillo et al. 2025)
- **KPIScore** — laboratory performance metric, integer score 5–25, with
  Beta-distributed 95% CIs per level

Produces a three-level decomposition: per transfer, cumulative when
transferable embryos exist, and whole-cycle probability (including cycles
without a transfer).
</details>

<details>
<summary><b>L3 — KAT neural ensemble + Bayesian posterior</b></summary>

- **KAN** — 3-layer Kolmogorov-Arnold Network, B-spline activations (Liu et al. 2024)
- **FT-Transformer** — (Gorishniy et al., NeurIPS 2021)
- **Isotonic calibration, interpolated** between the centres of the isotonic
  steps, so KAT responds continuously to embryology
- **NVSA** correction for attempt-number-dependent probability decay
- **Beta-Binomial conjugate posterior** fusing the network output, a
  covariate-dependent Beta regression prior, and the clinic's retrospective
  data (reported for analytics; see [Clinic data](#clinic-data))

Inputs follow the training definitions: follicles at puncture (entered, or
OCC / 0.846), day-5 embryos = round((blastocysts + 2PN) / 2), frozen =
max(good − 1, 0), one embryo transferred. KAT enters L7 as the mean over
scenarios with a transfer; without KAT weights it does not enter L7 at all.

**FT-Transformer encoding.** mambular 0.2.2 replaces the last fitted
piecewise-linear encoding threshold with the maximum of the batch, which makes
one patient's output depend on the other rows. `src/mambular_ple_fix.py`
restores the fitted thresholds at start-up, exactly as in the 7.1 clinic build.
</details>

<details>
<summary><b>L4 — Unsupervised cluster classifier</b></summary>

Nearest-centroid assignment to three protocol phenotypes (poor / standard /
high responder) in an 18-dimensional z-standardized feature space. Centroids
from k-means (k = 3) over 1,556 cycles. 7.1 reports the standardized distance
to each centroid, which feeds cluster certainty in L7. This is a phenotype
comparison, not a formal POSEIDON assignment and not a separate pregnancy
probability.
</details>

<details>
<summary><b>L5 — CSDI Hybrid v3 diffusion module</b></summary>

A two-stage generative model that reconstructs embryological outcome
distributions **without parametric rate assumptions** — a consistency check
on L1's embryology, not another independent vote on pregnancy.

```
CONDITIONING (7): follicle count · OCC · inseminated · 2PN
                  OCC retrieval rate · fertilization rate · KPIScore

STAGE 1  CSDI Transformer (generative)
         QuantileNormalizer → CSDIDenoiser
         4 layers × 4 heads × hidden 128
         DDIM sampling, 50 steps, T = 1000 cosine schedule
         generates blastocyst counts (≤ 2PN); derives rates analytically

STAGE 2  LightGBM + Platt scaling
         7 conditioning + 2 count medians → P(pregnancy)
         DART boosting · ECE = 0.029

STAGE 3  Split conformal prediction
         distribution-free coverage; 90% PI → 91–93% observed
```

In 7.1 CSDI is conditioned on the medians of scenarios with a transfer and on
the training KPIScore formula. It is not run without 2PN. Before it enters
fusion, a Mahalanobis check against its training cohort
(`models/csdi_ood_stats.npz`) must place the case inside the training domain;
outside it, or without that file, the estimate may be shown but is excluded
from L7. The interval around P(pregnancy) is the 2.5–97.5% quantile of
per-scenario predictions, replacing a Wilson interval that assumed 1,000
Bernoulli observations.

**Why this architecture.** The previous TabDDPM v3 (FiLM-ResNet) carried a
+15.8 pp prevalence bias at AUROC 0.578. Separating count generation
(diffusion) from binary prediction (discriminative classifier) cut ECE from
0.158 to 0.029 and removed the bias.
</details>

<details>
<summary><b>L6 — GAT graph attention transformer</b></summary>

Formalizes "I have seen patients like this before." 1,172 clinical protocols
as nodes; edges weighted by cosine similarity in the 18-dimensional feature
space. For each new patient the k = 10 nearest neighbours are retrieved and
attention propagates across the subgraph. In 7.1 the feature contract was
checked row by row against training, and the k-NN graph is exact and
deterministic for ties.

Trust features passed to L7:

- `N_eff = 1/Σ(attention_weight²)` — effective neighbour count. High N_eff
  means many genuinely similar patients support the prediction; low N_eff
  flags an isolated case.
- **Attention entropy** — breadth of neighbourhood support
- **Neighbour outcome variance** — stability of the graph signal

A well-supported neighbourhood gets full weight in L7. An isolated patient has
its GAT contribution suppressed and the posterior falls back toward the
mechanistic prior.

The model can display the ten most similar training-cohort patients with their
documented outcomes — a contextualisation no scalar score provides.
</details>

<details>
<summary><b>L7 — BEFE, the Bayesian arbiter</b></summary>

L7 treats every upstream layer as a **named expert** and learns how much to
trust each one. It does not learn a new model of pregnancy. Because the
headline is per transfer, BEFE fuses the layers on the Monte Carlo scenarios in
which a transfer is possible.

Fusion is precision-weighted pooling in logit space — a conjugate Gaussian
approximation to Bayesian model averaging:

```
l_post  = (τ_prior · logit(P_L1) + τ_emp · logit(P_predictive)) / (τ_prior + τ_emp)
P_post  = sigmoid(l_post)
```

| Expert | Trust features determining τ |
|---|---|
| Prior (L1) | Spread of the per-scenario probabilities; L5 agreement |
| KAT (L3) | Width of the KAT bootstrap interval |
| GAT (L6) | N_eff; attention entropy; neighbour outcome variance |

When both neural models are absent or OOD, `τ_emp → 0` and the posterior
collapses to the mechanistic prior — the correct Bayesian fallback. **The
fusion pull ratio is always reported**, so the clinician sees which source
drove the number.

**Uncertainty range.** The interval shown with the posterior adds two logit
variances: the spread of the cycle's remaining scenarios (it vanishes once the
embryology is known), and the variance of the fused estimate under a
random-effects (DerSimonian–Laird) pooling of L1, KAT and GAT with the
precisions the fusion actually used. Disagreement between sources widens it
only beyond what their own precision explains; OOD widens it through the same
precisions. It is a model uncertainty range, not a confidence interval with
verified coverage.

**Reliability Index (0–100)** — 40% expert consensus + 30% diffusion agreement
+ 20% graph stability + 10% cluster certainty. Engine bands: High ≥ 70,
Moderate 45–69, Low < 45 (the Streamlit sidebar can change the displayed band,
not the score). An OOD flag scales the score down in proportion to how far
outside the patient is (to at most −35%).

**Dual OOD detection** — two independent Mahalanobis detectors with fitted
thresholds: `OOD_clinical` over age/AFC and `OOD_embryology` over OCC and the
empirical-Bayes MII/OCC, 2PN/MII and blastocyst/2PN ratios, referenced to the
GAT/KAT training cohort. AMH, BMI and KPIScore are reported as not assessed.
`OOD_final = max(...)`. The split lets the report say *"clinically typical,
embryologically atypical"*.

Physician-facing output (standard patient: age 35, AMH 2.5, AFC 15, before
stimulation):

```
====================================================
BEFE — BAYESIAN EVIDENCE FUSION  (Digital Twin L7)
====================================================

Mechanistic prior  (L1):        61%
Empirical evidence (L3/L6):     59%   [P_predictive]
----------------------------------------------------
Posterior probability:          59%
95% CI:                         29%-84%
Fusion pull:                    evidence 87% / prior 13%

Reliability:                    76/100  (High)
Consensus (empirical):          High
Patient similarity:             Moderate (N_eff = 10)
Diffusion agreement (L5):       Good
Cluster:                        0

OOD clinical:                   No
OOD embryology:                 No
OOD final (max):                No
====================================================
```

The "95% CI" line of the engine report is the uncertainty range described
above. It narrows as the cycle is observed — the same patient:

| Entered | P(pregnancy) per transfer | Range | Cycle probability |
|---|---|---|---|
| Nothing (before stimulation) | 59.1% | 29.0–83.6% | 82.4% |
| OCC 10, MII 8, 2PN 6 | 58.3% | 30.4–81.7% | 84.9% |
| + 3 blastocysts, 2 good quality | 48.8% | 26.3–71.9% | 76.2% |

When reliability drops, BEFE names the disagreement rather than hiding it:

```
Source of uncertainty: KAT (L3) and GAT (L6) differ by 20 pp
  KAT (neural network): 50%
  GAT (patient graph):  70%
KAT receives higher weight as the better-calibrated model.
Possible cause: non-standard clinical/embryological ratio.
```
</details>

<details>
<summary><b>Headline, cycle probability and TRP</b></summary>

`presentation.clinical_summary` defines one headline for the screen, the PDF
and the analytics row:

- **Per transfer** — the L7 posterior, in the usual case.
- **0 for the current cycle** — when the entered results exclude a transfer
  (zero OCC, MII, 2PN or blastocysts, or zero euploid with PGT-A). Zero
  good-quality blastocysts alone do not close the cycle.

The **cycle probability** is made consistent with the headline: scenario
probabilities are shifted by one logit constant so that their mean over
scenarios with a transfer equals the headline, then combined over all
transfers of the cycle with the shared random effect.

**TRP** (total reproductive potential) extrapolates the cycle probability over
future attempts with age, AMH decline and attempt number. It starts from the
L1–L7 cycle probability of the case; if the current cycle has no possible
transfer, it starts from a fresh cycle without the current observations, so
one closed cycle does not zero the whole plan. Outcomes are integrated rather
than sampled, so cycle 1 equals the anchor exactly. TRP estimates time to the
first pregnancy.
</details>

---

## Installation

### Windows — one click

1. Install **Python 3.11** from [python.org](https://www.python.org/downloads/), ticking *Add Python to PATH*.
2. Double-click **`INSTALL.bat`** — creates `.venv`, installs everything including PyTorch and PyTorch Geometric.
3. Double-click **`Start_IVF_Twin.bat`** — opens `http://localhost:8501`.

### Core only — no PyTorch

To read the code, run the test suite, or exercise the mechanistic pipeline
(L1 + L2 + L4 and BEFE on the L1 prior) without a 200 MB PyTorch download:

```bash
pip install -r requirements-core.txt
pytest tests -v
```

This is what CI runs. The neural layers report "model not loaded" and
everything else works.

### Any platform — manual, full stack

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate.bat

pip install -r requirements.txt
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_nn.txt
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
pip install torch-geometric

streamlit run app.py
```

> PyTorch Geometric wheels must match the exact PyTorch build (2.5.1+cpu).
> `INSTALL.bat` handles this; the manual path does not, which is why the
> `-f` index is explicit above.

> mambular must stay at **0.2.2**. The FT-Transformer encoding fix
> (`src/mambular_ple_fix.py`) is applied automatically by `app.py`,
> `dt_bridge.py` and the batch scripts; `tests/test_mambular_ple_fix.py`
> checks it. Code that imports `ivf_core` directly should call
> `mambular_ple_fix.apply()` first.

### Research Mode

This repository ships **without** the licence engine, so the app starts
directly in Research Mode with a banner to that effect. Every layer for which
you supply weights is available. Clinical deployment builds add the licence
engine — see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).

### Graceful degradation

If a neural component cannot load — missing weights, PyTorch version mismatch,
insufficient memory — the system falls back to L1 + L2 + L4, which remain
fully functional, and the UI states which layers are active. A layer that is
unavailable, or outside its applicability domain, does not enter L7; nothing
is silently substituted in its place.

---

## Quickstart

### The application

```bash
streamlit run app.py
```

The app opens on a **clinical summary** (headline, cycle probability,
reliability, banking, TRP). *Detailed report* shows the layer tabs:

| Tab | Content |
|---|---|
| L1 Pipeline | Stage medians, distributions, 95% intervals |
| L2 Pregnancy | Per transfer, if transferable, whole cycle |
| Risks | OHSS, empty cycle, oocyte distribution |
| Banking | Esteves euploid-per-MII model, oocytes to bank |
| TRP | Reproductive horizon over future attempts |
| L4 Cluster | Phenotype classification |
| L5 Diffusion | CSDI outputs, applicability, conformal intervals |
| L6 GAT Graph | Similarity graph and nearest neighbours |
| L7 BEFE | Fusion report — the headline number |
| LLM | Local narrative (Ollama) |

The PDF report is generated from the same result.

Inputs: age, AMH (ng/mL), AFC, height and weight (BMI), attempt number,
follicles at puncture, sperm source. Optional mid-cycle observations: OCC,
MII, 2PN, blastocysts, good-quality blastocysts, euploid (PGT-A). In this
interface 0 means "not observed"; the Python API below distinguishes
`None` (unknown) from `0` (observed absence).

### The full stack from Python

```python
import dt_bridge   # run from the repository root

result = dt_bridge.compute({
    "age": 35.0, "amh": 2.5, "afc": 15, "bmi": 23.0, "attempt": 1,
    "follicles": None, "sperm_source": "ejaculate",
    "known_okk": 10, "known_mii": 8, "known_pn2": 6,
    "known_blasts": None, "known_good": None, "known_euploid": None,
    "n_sim": 2000, "seed": 42,
})

result["clinical_summary"]   # {'probability': ..., 'kind': 'per_transfer', ...}
result["fusion"]             # BEFEResult: posterior, ci_low/ci_high, reliability
result["cycle_probability"]  # consistent with the headline
dt_bridge.trp_anchor(result) # (probability, source) for the TRP plan
```

Sperm sources are Esteves strata: `ejaculate`, `testicular_NOA`,
`testicular_OA`, `epididymal` (aliases such as `donor`, `tese_noa`, `pesa` are
resolved; unknown values are rejected rather than treated as ejaculate).

### The CSDI module directly

```python
from src.embryo_csdi_v3 import EmbryoHybridV3

model = EmbryoHybridV3.load("models/embryo_v3_model")

patient = {
    "Количество фолликулов":  12.0,
    "Число ОКК":               9.0,
    "Число инсеминированных":  8.0,
    "2 pN":                    6.0,
    "Частота получения ОКК":   0.75,
    "Частота оплодотворения":  0.75,
    "KPIScore":               18.0,
}

result = model.mc_sample(patient, n_samples=2000)
result["P_pregnancy"]         # calibrated probability
result["CI_95"]               # (lo, hi), 2.5–97.5% of per-scenario predictions
result["blast_total_median"]  # median blastocyst count
result["PI_90_counts"]        # {'Число Bl': (lo, hi), ...} conformal, ≤ 2PN
result["samples"]             # DataFrame, 2000 × 4
```

Calling the module directly skips the applicability check that decides
whether CSDI enters L7 (`befe_batch_utils.assess_csdi`).

### Sample cohort

`data/sample/sample_patients.csv` — five synthetic reference patients:

| Age | AMH | AFC | Phenotype |
|---|---|---|---|
| 32 | 2.5 | 18 | Standard responder |
| 42 | 0.5 | 6 | Poor responder |
| 28 | 4.5 | 28 | High responder |
| 35 | 2.5 | 15 | Typical |
| 38 | 1.5 | 10 | Mid-range borderline |

### Clinic data

Copy `clinic_config.template.json` to `clinic_config.json` and populate
`batches` with `[successes, transfers]` pairs from your own history. They
enter the L3 Beta-Binomial clinic posterior (`bayes_mean` in analytics); they
do not change the L7 headline or its range. Real clinic data is never
committed — see [SECURITY.md](SECURITY.md).

```bash
python validate_clinic_data.py  --input your_cycles.xlsx    # schema + sanity check
python calibrate_for_clinic.py  --input your_cycles.xlsx    # research recalibration
```

`calibrate_for_clinic.py` fits research layer temperatures; in 7.1 they are
not part of the clinical calculation. The clinic build's centre calibration (a
validated intercept correction bound to the model fingerprint) is not part of
this repository.

---

## Repository map

```
IVF-Digital-Twin/
├── app.py                     Streamlit clinical application
├── dt_ui.py  i18n.py          UI components, RU/EN localization
├── patient_brief.py           clinical summary view (default)
│
├── dt_bridge.py               7.1 data flow: L1–L6 → L7 → headline → TRP anchor
├── presentation.py            headline contract (per transfer / closed cycle)
│
├── src/
│   ├── ivf_core.py            one case, L1–L6, transfer view, model caches
│   ├── ivf_digital_twin.py    core L1–L4 pipeline, KAT, Esteves banking
│   ├── embryology.py          L1 coefficients, exact conditioning, transfers
│   ├── embryo_csdi_v3.py      CSDI Hybrid v3 diffusion (L5)
│   ├── embryo_tabddpm.py      TabDDPM v3, superseded (L5)
│   ├── gnn_predictor.py       GAT graph transformer (L6)
│   ├── modelio.py             one loader for plain and encrypted model files
│   ├── mambular_ple_fix.py    FT-Transformer encoding fix (mambular 0.2.2)
│   ├── local_network.py       Ollama host must be loopback
│   └── pdf_report.py          clinical PDF generator
│
├── befe.py befe_app.py        BEFE fusion engine (L7) + UI
├── befe_batch_utils.py        L7 for batches, OOD loading, CSDI applicability
├── fit_befe_ood.py            fits the dual OOD detectors
│
├── batch_analysis.py          cohort-level batch prediction
├── dt_postprocess.py          post-processing and control charts
├── trp_engine.py              total reproductive potential (TRP)
├── stim_protocol.py           deterministic stimulation guidance
│
├── calibrate_for_clinic.py    research recalibration
├── validate_clinic_data.py    intake schema validation
│
├── llm_consultant.py          narrative layer (local Ollama only)
├── guideline_rag.py           retrieval over guidelines_pack.json
├── faithfulness.py            grounding score for generated text
├── eval_retrieval.py          retrieval evaluation harness
├── protocol_guidance.py       protocol recommendation text
│
├── models/                    architecture docs + config, no weights
├── data/sample/               synthetic reference patients
├── docs/                      architecture, coefficients, QA protocols
├── narrator_qa/               narrative QA harness (no results)
├── tests/                     pytest suite
└── scripts/                   batch prediction, SPDX, public export
```

**Not in this repository, by design:** the offline licence engine, trained
neural network weights and fitted OOD statistics, any clinic's real outcome
data, and any patient-level record. The filter is enforced by
`scripts/export_public_repo.py`, which applies an allow-list, a denylist and a
secret scan before copying anything.

---

## Model weights

Trained weights are **not distributed here** — they encode proprietary
training data. All files go in `models/` (7.1 no longer searches `src/` or the
repository root).

| Artifact | Size | Layer |
|---|---|---|
| `models/Prediction_KAN.pth` | ~13 KB | KAN (L3) |
| `models/FTTransformer.joblib` | ~40 MB | FT-Transformer (L3) |
| `models/KAT_ensemble_raw_weights.pth` | ~2 KB | KAT ensemble weights (L3) |
| `models/isotonic_ensemble.pkl` | ~1 KB | KAT isotonic calibrator (L3) |
| `models/gnn_ivf_model.pt` | ~2 MB | GAT (L6) |
| `models/embryo_v3_model/csdi_weights.pt` | ~6 MB | CSDI denoiser (L5) |
| `models/embryo_v3_model/normalizer.pt` | ~100 KB | QuantileNormalizer |
| `models/embryo_v3_model/lgb_state.pt` | ~2 MB | LightGBM classifier |
| `models/embryo_v3_model/platt_calibrator.pt` | ~2 KB | Platt scaling |
| `models/embryo_v3_model/conformal.pt` | ~2 KB | Conformal radii |
| `models/befe_ood_stats.npz` | ~4 KB | L7 dual OOD detector (schema 3) |
| `models/csdi_ood_stats.npz` | ~2 KB | CSDI applicability (L5) |

Architecture definitions, hyperparameters, configuration and training history
**are** included, so the models are reproducible from your own data. The
methods are documented in [`models/`](models/) —
`CSDI_Hybrid_v3_Technical_Description.md` and `GAT_method_description.md`.

Weights are available for **research collaboration and external validation**:
email [embryossa@gmail.com](mailto:embryossa@gmail.com). Loading them executes
pickled code — read the trust boundary in [SECURITY.md](SECURITY.md) first.

Without weights, L1 + L2 + L4 run fully; L3, L5 and L6 report "model not
loaded". Without `befe_ood_stats.npz` the OOD detector is off; without
`csdi_ood_stats.npz` CSDI never enters L7.

---

## Clinical interpretation

### Reading the BEFE output

| Field | Clinical meaning |
|---|---|
| **P(pregnancy) posterior** | Clinical pregnancy per transfer — the number to use in counselling |
| **Uncertainty range** | Model uncertainty around it; narrows as the cycle is observed |
| **Reliability (0–100)** | How much to trust it: ≥70 high, 45–69 moderate, <45 low (engine default) |
| **Fusion pull** | Which source dominated — high evidence pull = data-driven; high prior pull = mechanistic fallback |
| **Source of uncertainty** | When models disagree, the specific pair and the gap |
| **OOD status** | Whether the patient sits outside the training distribution, clinically or embryologically; which features were not assessed |

Read the probability together with its range, reliability and OOD status; do
not average the layers — the result is L7. The cycle probability answers a
different question (at least one pregnancy from this stimulation, over all its
transfers) and is shown separately.

**High reliability.** Models agree, the neighbourhood is well populated,
diffusion confirms the Monte Carlo prediction. Present the posterior with
confidence.

**Moderate.** Some KAT/GAT disagreement, sparse neighbours, or weak diffusion
agreement. Use the posterior but present the range explicitly.

**Low.** Substantial disagreement or an OOD flag. BEFE still gives the best
available synthesis, but weight clinical judgment more heavily and read the
divergence source.

### Equifinality verification

When L1 (parametric) and L5 (data-driven, 15,193 cycles) agree on blastocyst
distributions — low KS statistic, p > 0.05 — two genuinely independent
epistemic sources have reached the same conclusion. The BEFE prior gains
precision and the diffusion component of the Reliability Index scores near
maximum.

When they diverge, prior precision drops, the range widens, and reliability
falls. **The divergence is itself diagnostic**: it points at either an unusual
patient or a laboratory process that has drifted from the training data.

### CSDI thresholds (L5)

The upper threshold is `best_threshold` from `models/embryo_v3_model/config.json`
(0.355 in the shipped configuration); the app reads it from the model.

| P(pregnancy) | Interpretation |
|---|---|
| ≥ 0.355 | Favourable — expected laboratory outcomes support transfer |
| 0.25–0.355 | Moderate — consider additional cycles or PGT-A |
| < 0.25 | Cautious — low blastocyst yield warrants counselling |

### Conformal intervals (L5)

The 90% conformal PI for blastocyst counts carries finite-sample coverage
without distributional assumptions, and in 7.1 is limited to the physically
possible range (0 to 2PN). Typical output (age 35, AFC 12, KPI 18):

```
P(pregnancy) = 46.7%   95% prediction interval over scenarios
Blastocysts total (median): 2   PI_90: [0, 5]
Blastocysts good  (median): 1   PI_90: [0, 3]
```

### Laboratory quality management

Because the system produces full predicted distributions at every stage, the
predicted−observed difference at each stage can be charted as a Shewhart
control chart with a patient-adjusted centre line and Monte Carlo-derived
control limits. That gives stage-resolved discrepancy attribution and early
detection of process variability.

The L7 OOD detectors add a second surveillance layer: systematic embryological
OOD flags (OCC, 2PN, blastocyst) can indicate equipment drift or a protocol
change before it becomes visible in outcome statistics.

---

## Privacy and security

- **Everything runs locally.** The app binds to `localhost`; the narrative
  layer talks to a local Ollama instance, and 7.1 refuses any `OLLAMA_HOST`
  that is not a loopback address. **No patient data is sent to any cloud LLM
  provider**, and there are no third-party API keys anywhere in this codebase.
- **Do not expose the app to the internet.** It has no authentication or
  multi-tenant isolation and was never designed for it.
- **Model files are trusted input.** `torch.load` / `joblib.load` execute code
  from the file they read. Load weights only from a source you trust.
- **No patient record is in this repository.** `data/sample/` is synthetic.
- Report vulnerabilities privately per [SECURITY.md](SECURITY.md).

Users deploying this remain responsible for GDPR / HIPAA / national health data
compliance, including lawful basis, minimization and retention.

---

## Licensing

**Source-available, not open source.**

The source is public so the methods, coefficients, architecture and calibration
can be inspected, reproduced and cited — what a clinical prediction tool should
allow. It is not public so it can be commercialized by third parties.

| You are | Terms |
|---|---|
| Researcher, student, university, public research institute, public hospital, government body, charity | **Free** — [PolyForm Noncommercial 1.0.0](LICENSE) |
| Anyone reading, auditing or reproducing results | **Free** — same licence |
| Private clinic, laboratory or company using it in paid services or a product | **Commercial licence required** — [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) |

Versions up to tag **`v6.2-apache`** were released under Apache-2.0 and remain
available under those terms — see [LICENSE-HISTORY.md](LICENSE-HISTORY.md).

Third-party dependencies keep their own licences
([THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)). Contributions require a CLA
([CONTRIBUTING.md](CONTRIBUTING.md)) — the reasoning is explained there.

---

## Citation

If you use this software in research, cite the software and the associated
publication. Machine-readable metadata is in [CITATION.cff](CITATION.cff).

```bibtex
@software{sergeev2026ivfdigitaltwin,
  author  = {Sergeev, Sergei},
  title   = {{IVF Digital Twin v7.1}: An Integrated Multi-Source Ensemble
             Platform for Stage-Stratified {IVF} Outcome Prediction},
  year    = {2026},
  version = {7.1.0},
  url     = {https://github.com/embryossa/IVF-Digital-Twin},
  note    = {Research prototype; not a medical device}
}
```

Scientific co-authors of the methodology are credited in
[AUTHORS.md](AUTHORS.md).

---

## References

1. Craig A et al. Stage-Structured, Distributional Prediction of IVF Outcomes with Conditional Updating. *medRxiv* 2025.09.27.25336680.
2. Carrasquillo R et al. FORTUNE: A clinically validated prediction model for IVF live birth rate. *Hum Reprod.* 2025. PMID: 40889782.
3. Liu Z et al. KAN: Kolmogorov-Arnold Networks. *arXiv:2404.19756*, 2024.
4. Gorishniy Y et al. Revisiting Deep Learning Models for Tabular Data. *NeurIPS*, 2021.
5. Vovk V, Petej I. Venn-Abers predictors. *arXiv:1211.0025*, 2012.
6. Tashiro Y et al. CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation. *NeurIPS*, 2021.
7. Nichol A, Dhariwal P. Improved Denoising Diffusion Probabilistic Models. *ICML*, 2021.
8. Song J et al. Denoising Diffusion Implicit Models. *ICLR*, 2021.
9. Ke G et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*, 2017.
10. Angelopoulos A, Bates S. A Gentle Introduction to Conformal Prediction. *arXiv:2107.07511*, 2022.
11. Franasiak JM et al. The nature of aneuploidy with increasing age of the female partner: 15,169 biopsies. *Fertil Steril.* 2014;101(3):656–663.
12. Romanski PA et al. Age-specific blastocyst conversion rates in embryo cryopreservation cycles. *Reprod Biomed Online.* 2022;45(3):432–439.
13. Coello A et al. Prediction of embryo survival and live birth rates after cryotransfers. *Reprod Biomed Online.* 2021;42(5):881–891.
14. Sergeev S et al. Decoding IVF Laboratory Performance through Dimensionality Reduction and Cluster Analysis. *(manuscript under review).*
15. Esteves SC et al. *Front Endocrinol.* 2019;10:99 — POSEIDON ART calculator; Table 2 logistic model of euploid blastocysts per MII (banking module). doi:10.3389/fendo.2019.00099.
16. DerSimonian R, Laird N. Meta-analysis in clinical trials. *Control Clin Trials.* 1986;7(3):177–188.

---

<div align="center">

**Sergei Sergeev** · [embryossa@gmail.com](mailto:embryossa@gmail.com) · [LinkedIn](https://www.linkedin.com/in/serdj-sergeev-8b5893298/)

Research collaboration · model weight access · clinic deployment · external validation

[Licence](LICENSE) · [Commercial](COMMERCIAL-LICENSE.md) · [Security](SECURITY.md) · [Disclaimer](DISCLAIMER.md) · [Changelog](CHANGELOG.md)

</div>
