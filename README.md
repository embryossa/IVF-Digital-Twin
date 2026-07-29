<div align="center">

<img src="logo22.png" width="96" alt="IVF Digital Twin">

# IVF Digital Twin v7.0

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
them into one calibrated posterior with an explicit account of how much each
source was trusted and why.

Every coefficient is traceable to a peer-reviewed source. See
[`docs/coefficients.md`](docs/coefficients.md).

### What makes it different

| | Point-estimate calculators | **IVF Digital Twin v7.0** |
|---|---|---|
| Output | A single number | Full distributions at every cycle stage |
| Uncertainty | Absent or nominal | Monte Carlo, conformal, and Bayesian intervals |
| Mid-cycle information | Ignored | Posterior re-conditions on each observation |
| Model disagreement | Hidden | Named, quantified, and attributed |
| Out-of-distribution patients | Silently extrapolated | Flagged on two independent feature subspaces |
| Auditability | Black box | Every coefficient cited; every weight reported |

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

| Layer | Component | Metric | Value |
|---|---|---|---|
| L1 | Stochastic pipeline | Spearman ρ (oocyte count) | 0.73 |
| L1 | Stochastic pipeline | Spearman ρ (blastocyst count) | 0.41 |
| L5 | CSDI Hybrid v3 | AUC | 0.661 |
| L5 | CSDI Hybrid v3 | Brier | 0.209 |
| L5 | CSDI Hybrid v3 | ECE | 0.029 |
| L5 | CSDI Hybrid v3 | Prevalence bias | −0.1 pp |
| L6 | GAT alone | AUC-ROC | 0.632 |
| L3+L6 | GAT + KAT ensemble | AUC-ROC | 0.658–0.665 |
| L3+L6 | GAT + KAT ensemble | Brier | ~0.229 |
| — | Overall pregnancy prediction | AUC | 0.63 |
| — | Overall pregnancy prediction | Brier | 0.22 |

**Reading these honestly.** An AUC around 0.63–0.66 is typical of IVF outcome
prediction and reflects a genuine ceiling: much of the variance in whether a
transfer implants is not captured by any pre-transfer variable currently
measured. The contribution here is not discrimination — it is **calibration**
(ECE 0.029), distributional output, and explicit uncertainty attribution. A
well-calibrated 40% that knows when it is unreliable is more useful in
counselling than a sharper number that does not.

CSDI Hybrid v3 was trained on 15,193 cycles with a 3-way split
(85% diffusion / 7.5% LightGBM / 7.5% conformal calibration). The GAT graph
holds 1,172 clinical protocols. Cluster centroids derive from 1,556 cycles.

---

## Architecture

```
Patient inputs: age · AMH · AFC · BMI · attempt number
                          │
 L1  Stochastic Monte Carlo pipeline (N = 5,000)
     ZINB oocyte model → 7 sequential biological stages
     → mechanistic prior P(pregnancy | physiology)
                          │
 L2  Per-transfer ensemble — FORTUNE + KPIScore
     three-level pregnancy decomposition
                          │
 L3  KAT neural ensemble — KAN (B-spline) + FT-Transformer
     Venn-Abers calibration · Beta-Binomial Bayesian posterior
                          │
 L4  Unsupervised cluster classifier
     nearest-centroid in 18-dim space, k = 3 phenotypes
                          │
 L5  CSDI Hybrid v3 diffusion module
     CSDI Transformer + LightGBM + split conformal
     → equifinality verification of L1
                          │
 L6  GAT graph attention transformer
     patient-similarity graph · 1,172 protocols
                          │
 L7  BEFE — Bayesian Evidence Fusion Engine
     prior (L1·L5) → evidence (L3·L6) → posterior
     trust-weighted logit pooling · reliability · dual OOD
                          │
 OUTPUT  one posterior · 95% CI · reliability score
         uncertainty source · PDF clinical report
```

<details>
<summary><b>L1 — Stochastic Monte Carlo pipeline</b></summary>

Seven probabilistic stages, from antral follicles to post-thaw survival.

| Stage | Model | Source |
|---|---|---|
| 1. Retrieved oocytes | Zero-Inflated Negative Binomial, θ = 5.0, logistic zero-inflation | Craig et al. 2025; HFEA registry cancellation rates |
| 2. Mature (MII) | Logistic-binomial filter | — |
| 3. Two-pronuclear (2PN) | Binomial filter | — |
| 4. Blastocysts | Age-stratified conversion | Romanski et al. 2022 |
| 5. Good-quality blastocysts | Beta-binomial | — |
| 6. Euploid embryos | Age-stratified aneuploidy table | Franasiak et al. 2014 |
| 6b. Post-thaw survival | Age-stratified survival | Coello et al. 2021 |

OHSS probability and empty-cycle risk are computed per iteration, not
post-hoc.
</details>

<details>
<summary><b>L2 — Per-transfer ensemble</b></summary>

Two independent probability sources combined on the logit scale:

- **FORTUNE** — logistic regression on clinical predictors (Carrasquillo et al. 2025)
- **KPIScore** — laboratory performance metric, integer score 5–25, with
  Beta-distributed 95% CIs per level

Produces a three-level decomposition: per-transfer, cumulative-if-viable, and
overall cycle probability.
</details>

<details>
<summary><b>L3 — KAT neural ensemble + Bayesian posterior</b></summary>

- **KAN** — 3-layer Kolmogorov-Arnold Network, B-spline activations (Liu et al. 2024)
- **FT-Transformer** — (Gorishniy et al., NeurIPS 2021)
- **Venn-Abers** conformal calibration (Vovk & Petej 2012)
- **NVSA** correction for attempt-number-dependent probability decay
- **Beta-Binomial conjugate posterior** fusing the network output, a
  covariate-dependent Beta regression prior, and the clinic's retrospective data

Mid-cycle conditional updating re-conditions the posterior as each stage
observation arrives (known OCC, 2PN, blastocyst counts).
</details>

<details>
<summary><b>L4 — Unsupervised cluster classifier</b></summary>

Nearest-centroid assignment to three protocol phenotypes (poor / standard /
high responder) in an 18-dimensional z-standardized feature space. Centroids
from k-means (k = 3) over 1,556 cycles. PCA visualization with synthetic
cohort clouds.
</details>

<details>
<summary><b>L5 — CSDI Hybrid v3 diffusion module</b></summary>

A two-stage generative model that reconstructs embryological outcome
distributions **without parametric rate assumptions** — an epistemically
independent check on L1's literature-derived coefficients.

```
CONDITIONING (7): follicle count · OCC · inseminated · 2PN
                  OCC retrieval rate · fertilization rate · KPIScore

STAGE 1  CSDI Transformer (generative)
         QuantileNormalizer → CSDIDenoiser
         4 layers × 4 heads × hidden 128
         DDIM sampling, 50 steps, T = 1000 cosine schedule
         generates blastocyst counts; derives rates analytically

STAGE 2  LightGBM + Platt scaling
         7 conditioning + 2 count medians → P(pregnancy)
         DART boosting · ECE = 0.029

STAGE 3  Split conformal prediction
         distribution-free coverage; 90% PI → 91–93% observed
```

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
attention propagates across the subgraph.

Trust features passed to L7:

- `N_eff = 1/Σ(attention_weight²)` — effective neighbour count. High N_eff
  means many genuinely similar patients support the prediction; low N_eff
  flags an isolated case.
- **Attention entropy** — breadth of neighbourhood support
- **Neighbour outcome variance** — stability of the graph signal

A well-supported neighbourhood (N_eff ≥ 20, uniform attention) gets full
weight in L7. An isolated patient (N_eff ≤ 4) has its GAT contribution
suppressed and the posterior falls back toward the mechanistic prior.

The model can display the ten most similar training-cohort patients with their
documented outcomes — a contextualisation no scalar score provides.
</details>

<details>
<summary><b>L7 — BEFE, the Bayesian arbiter</b></summary>

L7 treats every upstream layer as a **named expert** and learns how much to
trust each one. It does not learn a new model of pregnancy.

Fusion is precision-weighted pooling in logit space — a conjugate Gaussian
approximation to Bayesian model averaging:

```
l_post  = (τ_prior · logit(P_L1) + τ_emp · logit(P_predictive)) / (τ_prior + τ_emp)
σ²_post = 1 / (τ_prior + τ_emp)
P_post  = sigmoid(l_post)
```

| Expert | Trust features determining τ |
|---|---|
| Prior (L1) | CI width of `sim_p_combined`; L5 diffusion agreement |
| KAT (L3) | MC-dropout variance; Venn-Abers CI width; ECE |
| GAT (L6) | N_eff; attention entropy; neighbour outcome variance |

When both neural models are absent or OOD, `τ_emp → 0` and the posterior
collapses to the mechanistic prior — the correct Bayesian fallback. When the
prior is diffuse and evidence is sharp, evidence dominates. **The fusion pull
ratio is always reported**, so the clinician sees which source drove the number.

**Reliability Index (0–100)** — 40% expert consensus + 30% diffusion agreement
+ 20% graph stability + 10% cluster certainty. Bands: High ≥ 75, Moderate
55–74, Low < 55. Capped at 49 when OOD is flagged.

**Dual OOD detection** — two independent Mahalanobis detectors:
`OOD_clinical` over age/AMH/AFC/BMI, `OOD_embryology` over OCC/MII/2PN/blast/KPI.
`OOD_final = max(...)`. The split lets the report say *"clinically typical,
embryologically atypical"* — actionable in a way a single flag is not.

**95% credible interval** comes from the Beta-posterior of the clinic model
(L3), calibrated on the clinic's own transfer history, so the interval is
clinically meaningful (typically ±5–10 pp) rather than a logit-space artifact.

Physician-facing output:

```
═══════════════════════════════════════════════
BEFE — BAYESIAN EVIDENCE FUSION  (L7)
═══════════════════════════════════════════════
P(pregnancy) posterior:     57%
95% CI (Beta):              36% – 45%
Reliability:                89/100  (High)

Mechanistic prior (L1):     59%
Empirical evidence (L3+L6): 54%
Fusion pull:                evidence 91% / prior 9%

Consensus (empirical):      High
Patient similarity:         Strong (N_eff = 32)
Diffusion agreement (L5):   Excellent
Cluster:                    High responder phenotype
OOD:                        No
═══════════════════════════════════════════════
```

When reliability drops, BEFE names the disagreement rather than hiding it:

```
Source of uncertainty: KAT (L3) and GAT (L6) differ by 20 pp
  KAT (neural network): 50%
  GAT (patient graph):  70%
KAT receives higher weight as the better-calibrated model.
Possible cause: non-standard clinical/embryological ratio.
```
</details>

---

## Installation

### Windows — one click

1. Install **Python 3.11** from [python.org](https://www.python.org/downloads/), ticking *Add Python to PATH*.
2. Double-click **`INSTALL.bat`** — creates `.venv`, installs everything including PyTorch and PyTorch Geometric.
3. Double-click **`Start_IVF_Twin.bat`** — opens `http://localhost:8501`.

### Any platform — manual

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

### Research Mode

This repository ships **without** the licence engine, so the app starts
directly in Research Mode with a banner to that effect. Every layer for which
you supply weights is available. Clinical deployment builds add the licence
engine — see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).

### Graceful degradation

If a neural component cannot load — missing weights, PyTorch version mismatch,
insufficient memory — the system falls back to L1 + L2 + L4, which remain
fully functional, and the UI states which layers are active. It does not fail
closed and it does not silently substitute a worse model.

---

## Quickstart

### The application

```bash
streamlit run app.py
```

| Tab | Content |
|---|---|
| 📊 Overview | Summary dashboard |
| 🎯 Monte Carlo | L1 stochastic distributions |
| 🧠 Neural Network | KAT ensemble + Bayesian posterior (L3) |
| 🔬 Clusters | Phenotype classification (L4) |
| 🧬 Diffusion | CSDI outputs + conformal intervals (L5) |
| 🕸️ Graph | GAT similarity graph + neighbours (L6) |
| ⚖️ BEFE | L7 fusion report — the headline number |
| 📋 Report | PDF export |

Inputs: `female_age`, `amh` (ng/mL), `afc`, `bmi`, `attempt_number`.
Optional mid-cycle updates: `follicles_tvp`, `known_okk`, `known_mii`,
`known_pn2`, `known_blasts`, `known_good`.

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
result["CI_95"]               # (lo, hi), Wilson
result["blast_total_median"]  # median blastocyst count
result["PI_90_counts"]        # {'Число Bl': (lo, hi), ...} conformal
result["samples"]             # DataFrame, 2000 × 4
```

### Sample cohort

`data/sample/sample_patients.csv` — five synthetic reference patients:

| Age | AMH | AFC | Phenotype |
|---|---|---|---|
| 32 | 2.5 | 18 | Standard responder |
| 42 | 0.5 | 6 | Poor responder |
| 28 | 4.5 | 28 | High responder |
| 35 | 2.5 | 15 | Typical |
| 38 | 1.5 | 10 | Mid-range borderline |

### Clinic calibration

```bash
python validate_clinic_data.py  --input your_cycles.xlsx    # schema + sanity check
python calibrate_for_clinic.py  --input your_cycles.xlsx    # refit priors
```

Copy `clinic_config.template.json` to `clinic_config.json` and populate
`batches` with `[successes, transfers]` pairs from your own history. These
anchor the L3 Beta-Binomial prior and supply the CI that BEFE reports. Real
clinic data is never committed — see [SECURITY.md](SECURITY.md).

---

## Repository map

```
IVF-Digital-Twin/
├── app.py                     Streamlit clinical application
├── dt_ui.py  i18n.py          UI components, RU/EN localization
│
├── src/
│   ├── ivf_core.py            shared primitives
│   ├── ivf_digital_twin.py    core L1–L4 pipeline
│   ├── embryo_csdi_v3.py      CSDI Hybrid v3 diffusion (L5)
│   ├── embryo_tabddpm.py      TabDDPM v3, superseded (L5)
│   ├── gnn_predictor.py       GAT graph transformer (L6)
│   └── pdf_report.py          clinical PDF generator
│
├── befe.py befe_app.py        BEFE fusion engine (L7) + UI
├── befe_batch_utils.py        batch BEFE helpers
├── fit_befe_ood.py            fits the dual OOD detectors
│
├── batch_analysis.py          cohort-level batch prediction
├── dt_postprocess.py          post-processing and control charts
├── trp_engine.py              transfer-readiness scoring
├── stim_protocol.py           deterministic stimulation guidance
│
├── calibrate_for_clinic.py    clinic-specific recalibration
├── validate_clinic_data.py    intake schema validation
│
├── llm_consultant.py          narrative layer (local Ollama only)
├── guideline_rag.py           retrieval over guidelines_pack.json
├── faithfulness.py            grounding score for generated text
├── eval_retrieval.py          retrieval evaluation harness
├── patient_brief.py           patient-facing summary
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
neural network weights, any clinic's real outcome data, and any patient-level
record. The filter is enforced by `scripts/export_public_repo.py`, which
applies an allow-list, a denylist and a secret scan before copying anything.

---

## Model weights

Trained weights are **not distributed here** — they encode proprietary
training data.

| Artifact | Size | Layer |
|---|---|---|
| `models/embryo_v3_model/csdi_weights.pt` | ~25 MB | CSDI denoiser (L5) |
| `models/embryo_v3_model/normalizer.pt` | ~1 MB | QuantileNormalizer |
| `models/embryo_v3_model/lgb_state.pt` | ~5 MB | LightGBM classifier |
| `models/embryo_v3_model/platt_calibrator.pt` | ~1 KB | Platt scaling |
| `models/embryo_v3_model/conformal.pt` | ~1 KB | Conformal radii |
| `models/kat_ensemble/` | ~50 MB | KAN + FT-Transformer (L3) |
| `models/gnn_model/` | ~10 MB | GAT (L6) |

Architecture definitions, hyperparameters, configuration and training history
**are** included, so the models are reproducible from your own data. The
methods are documented in [`models/`](models/) —
`CSDI_Hybrid_v3_Technical_Description.md` and `GAT_method_description.md`.

Weights are available for **research collaboration and external validation**:
email [embryossa@gmail.com](mailto:embryossa@gmail.com). Loading them executes
pickled code — read the trust boundary in [SECURITY.md](SECURITY.md) first.

Without weights, L1 + L2 + L4 run fully; L3, L5 and L6 report "model not
loaded".

---

## Clinical interpretation

### Reading the BEFE output

| Field | Clinical meaning |
|---|---|
| **P(pregnancy) posterior** | The integrated probability — the number to use in counselling |
| **95% CI (Beta)** | Uncertainty calibrated on your clinic's own transfer history |
| **Reliability (0–100)** | How much to trust it: ≥75 high, 55–74 moderate, <55 low |
| **Fusion pull** | Which source dominated — high evidence pull = data-driven; high prior pull = mechanistic fallback |
| **Source of uncertainty** | When models disagree, the specific pair and the gap |
| **OOD status** | Whether the patient sits outside the training distribution, clinically or embryologically |

**High reliability (≥75).** Models agree, the neighbourhood is well populated,
diffusion confirms the Monte Carlo prediction. Present the posterior with
confidence.

**Moderate (55–74).** Some KAT/GAT disagreement, sparse neighbours, or weak
diffusion agreement. Use the posterior but present the CI explicitly.

**Low (<55).** Substantial disagreement or an OOD flag. BEFE still gives the
best available synthesis, but weight clinical judgment more heavily and read
the divergence source.

### Equifinality verification

When L1 (parametric, literature coefficients) and L5 (data-driven, 15,193
cycles) agree on blastocyst distributions — low KS statistic, p > 0.05 — two
genuinely independent epistemic sources have reached the same conclusion. The
BEFE prior gains precision and the diffusion component of the Reliability Index
scores near maximum.

When they diverge, prior precision drops, the CI widens, and reliability falls.
**The divergence is itself diagnostic**: it points at either an unusual patient
or a laboratory process that has drifted from published norms.

### CSDI thresholds (L5)

| P(pregnancy) | Interpretation |
|---|---|
| ≥ 0.343 | Favourable — expected laboratory outcomes support transfer |
| 0.25–0.343 | Moderate — consider additional cycles or PGT-A |
| < 0.25 | Cautious — low blastocyst yield warrants counselling |

### Conformal intervals (L5)

The 90% conformal PI for blastocyst counts carries guaranteed finite-sample
coverage without distributional assumptions. Typical output (age 35, AFC 12,
KPI 18):

```
P(pregnancy) = 46.7%   95% CI: [43.6%, 49.8%]
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
  layer talks to a local Ollama instance at `127.0.0.1:11434`. **No patient
  data is sent to any cloud LLM provider**, and there are no third-party API
  keys anywhere in this codebase.
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
  title   = {{IVF Digital Twin v7.0}: An Integrated Multi-Source Ensemble
             Platform for Stage-Stratified {IVF} Outcome Prediction},
  year    = {2026},
  version = {7.0.1},
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

---

<div align="center">

**Sergei Sergeev** · [embryossa@gmail.com](mailto:embryossa@gmail.com) · [LinkedIn](https://www.linkedin.com/in/serdj-sergeev-8b5893298/)

Research collaboration · model weight access · clinic deployment · external validation

[Licence](LICENSE) · [Commercial](COMMERCIAL-LICENSE.md) · [Security](SECURITY.md) · [Disclaimer](DISCLAIMER.md) · [Changelog](CHANGELOG.md)

</div>
