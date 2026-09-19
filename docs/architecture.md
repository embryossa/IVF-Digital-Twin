# Architecture Documentation — IVF Digital Twin 7.1

Coefficients below are the values in the code. L1 constants live in
`src/embryology.py`; the per-transfer ensemble and KAT in
`src/ivf_digital_twin.py`; L5–L7 in `src/ivf_core.py`,
`befe_batch_utils.py`, `befe_app.py` and `befe.py`.

## Order of computation

`dt_bridge.compute` (used by `app.py`) runs one case in the same order as the
7.1 clinic build:

```
ivf_core.predict_single_patient      L1–L6, transfer view, Esteves banking
  → befe_batch_utils.compute_l7_posterior   L7 on the transfer view
  → presentation.clinical_summary           headline contract
  → embryology.anchored_cycle_probability   cycle probability
  → dt_bridge.trp_anchor                    start of the TRP plan
```

The screen, PDF, BEFE tab and analytics row read that one result.

## Complete data flow

```
PatientInput(age, AMH, AFC, BMI) · attempt · sperm source · follicles at puncture
KnownValues(okk, mii, pn2, blasts, good, euploid)   (None = unknown, 0 = observed absence)
        │
        ▼  N Monte Carlo scenarios (N_SIM = 5,000 in the core; 2,000 in the app and batch)
────────────────────────────────────────────
LAYER 1 — STOCHASTIC PIPELINE
────────────────────────────────────────────

S1  OKK[i]     ~ NegBin(θ, θ/(θ+μ)), clipped to [0, 50]
    μ = exp(0.05976 + 0.26843·ln(AMH+0.1) + 0.76892·ln(AFC+1) − 0.00393·(age−35))
    θ = 5.772; no separate structural-zero component
    source: zero-truncated NB fitted on stimulated cycles (n = 687 with OCC ≥ 1,
            5-fold patient cross-validation), 2026-09-11

S2  MII[i]     ~ Binomial(OKK[i], p_MII)
    logit(p_MII) = 2.4665 + 0.005·age − 0.782 + 0.24·AMH − 0.069
    source: Herasight Table A1 col 3 (Craig et al. 2025)

S3  2PN[i]     ~ Binomial(MII[i], p_fert)
    logit(p_fert) = 1.1678 + 0.004·age − 0.303 − 0.051
    source: Herasight Table A1 col 5 (Craig et al. 2025)

S4  Blast[i]   zero-inflated beta-binomial on 2PN[i]
    m = sigmoid(0.61431 − 0.016956·max(0, age−35) + 0.019593·ln(min(max(2PN,1), 5)))
    p ~ Beta(m·κ, (1−m)·κ), κ = 7.823;  Blast = 0 with probability π₀ = 0.0095
    source: protocols_15k, culture-observed cycles, n = 7,919, 2026-09-10/11

S5  Good[i]    ~ BetaBinomial(Blast[i], m·κ, (1−m)·κ)
    m = sigmoid(0.08825 − 0.063516·max(0, age−35) + 0.35722·ln(max(Blast,1))), κ = 4.131
    source: protocols_15k

S6  Euploid[i] ~ Binomial(Good[i], p_eup[i]),  p_eup[i] ~ Beta(mean·6, (1−mean)·6)
    mean: <30→0.70, 30–34→0.65, 35–37→0.55, 38–39→0.35, 40–41→0.18, ≥42→0.10
    source: Franasiak 2014 + Armstrong 2023

Transfer scenario (after warming, survival 0.95 per blastocyst — Coello 2021)
    without PGT-A:  good-quality embryos first, then fair quality (Blast − Good)
    with PGT-A:     euploid embryos only
    PGT-A is on when an euploid count is entered (or requested explicitly)
```

### Conditioning on observations

When any count is entered, the chain is not re-run stage by stage.
`embryology.sample_conditioned_counts` computes the exact joint distribution
of all six stages on a finite grid (0–50, or 0–100 for larger observed
counts) by forward filtering, restricts it to the observed values, and samples
backward. Unknown earlier stages are drawn from their conditional
distribution given the later observation; later stages stay stochastic.
Inconsistent entries (a later stage larger than an earlier one) are rejected.

```python
from ivf_digital_twin import PatientInput, KnownValues, run_pipeline_extended

patient = PatientInput(female_age=35, amh=2.5, afc=15, bmi=23.0)
res_prior  = run_pipeline_extended(patient, KnownValues())
res_okk    = run_pipeline_extended(patient, KnownValues(okk=12))
res_blasts = run_pipeline_extended(patient, KnownValues(okk=12, blasts=8))
res_pgt    = run_pipeline_extended(patient, KnownValues(okk=12, blasts=8, euploid=3))
```

```
────────────────────────────────────────────
LAYER 2 — PER-TRANSFER ENSEMBLE
────────────────────────────────────────────

FORTUNE component (per scenario):
  logit(p_F[i]) = 0.40 − 0.55·z_age + 0.15·z_lnAMH − 0.20·z_BMI + ε[i],  ε ~ N(0, 0.07)
  z_age = (age − 36.3)/5.5,  z_lnAMH = (ln(AMH+0.1) − ln 2.1)/1.2,  z_BMI = (BMI − 24)/4.2
  source: FORTUNE (Carrasquillo et al. 2025, PMID 40889782)

KPI component (per scenario):
  KPIScore[i] = age(1/3/5) + AMH(1/3/5) + MII[i](1/3/5) + fert rate[i](1/3/5) + good[i](1/3/5)
  p_KPI[i] ~ Beta moment-matched to the 95% CI of KPIScore[i]

Ensemble:
  logit(p[i]) = (1−w)·logit(p_F[i]) + w·logit(p_KPI[i]),  w = 0.5

Transfers of one cycle:
  first embryo: p[i]; fair quality after a good one: OR = exp(−0.666) = 0.51
  shared logit-normal random effect, σ = 0.746 (transfers from one retrieval)
  → P(≥1 pregnancy) per scenario by 32-node Gauss–Hermite quadrature;
    one embryo returns p[i] exactly; σ = 0 restores independent transfers

Three levels:
  [1] per transfer          = mean p[i]  (all scenarios; p_per_transfer)
      if transfer possible  = mean p[i] over scenarios with a transfer
  [2] cum-if-viable         = mean P(≥1) over scenarios with a transfer
  [3] overall (L1–L4 cycle) = mean P(≥1) over all scenarios  (p_overall_cycle)
```

```
────────────────────────────────────────────
TRANSFER VIEW (ivf_core.transfer_view)
────────────────────────────────────────────
The headline is per transfer, so the L1 prior, KAT and the count profile fed to
GAT/CSDI/OOD are restricted to the scenarios with ≥ 1 transferable embryo.
Scenarios without a transfer enter the cycle probability and the risk panel.

────────────────────────────────────────────
LAYER 3 — KAT NEURAL NETWORK ENSEMBLE
────────────────────────────────────────────

Architecture:
  KAN   width [18, 10, 1], B-spline grid 5, order 3
  FTT   FT-Transformer (mambular 0.2.2, piecewise-linear encoding)
  Ensemble: softmax(raw weights) · [KAN, FTT]
  Calibration: isotonic regression, interpolated linearly between the centres
               of its steps (InterpolatedIsotonic)

Inputs (training definitions, protocols_15k):
  follicles at puncture = entered value, else round(OCC / 0.846)
  day-5 embryos         = round((Blast + 2PN) / 2)
  frozen embryos        = max(Good − 1, 0)
  transferred embryos   = 1
  KPIScore              = follicle-based formula of the training data

FT-Transformer encoding: src/mambular_ple_fix.py keeps the fitted PLE
thresholds, so one row's output does not depend on the rest of the batch.

NVSA correction (with the same follicle-based KPIScore):
  cf = clip(p_KPI_table(KPIScore[i]) / p_NN[i], 1/1.5, 1.5)
  p_adjusted[i] = clip(p_NN[i] · cf, 0, 1)

Bayesian posterior (Beta-Binomial, reported as bayes_mean):
  Prior: covariate-dependent Beta(α₀, β₀), κ = 20 pseudo-observations
         (global Beta(26, 74) only as a fallback)
  + clinic batches from clinic_config.json: Σ successes, Σ failures
  + NN evidence: round(p_NN·100) successes / 100 pseudo-observations

To L7: KAT mean over scenarios with a transfer (None without KAT weights —
the FORTUNE+KPI fallback is the L1 prior itself and is not used as evidence).
```

```
────────────────────────────────────────────
LAYER 4 — CLUSTER CLASSIFIER
────────────────────────────────────────────

Feature vector (18-D) per scenario:
  [age, attempt, follicles, COCs, MII, 2PN, cleaving, HQ_blasts,
   day5, cryo, transferred, fert_rate, cleav_rate, blast_rate,
   TGBDR, retrieval_eff, KPI, NN_pred]

Assignment: nearest centroid in z-standardised space (population SDs).
Centroids: Sergeev et al., 1,556 cycles, k-means k = 3 (standard / poor / high).
Output: cluster_probs, dominant_cluster and, per cluster, the median
standardised RMS distance to the centroid (used for cluster certainty in L7).
```

```
────────────────────────────────────────────
LAYER 5 — CSDI HYBRID v3
────────────────────────────────────────────
Conditioning (from the transfer view): follicles at puncture, OCC, MII, 2PN
(rounded medians), OCC rate, fertilisation rate, follicle-based KPIScore.

Applicability (befe_batch_utils.assess_csdi):
  2PN = 0                      → not run (reason no_2pn)
  no models/csdi_ood_stats.npz → run, excluded from L7 (unchecked)
  Mahalanobis ratio > 1        → run, excluded from L7 (outside_training)
  otherwise                    → enters L7

Generated blastocyst counts are capped at 2PN; P(pregnancy) interval =
2.5–97.5% quantiles of the per-scenario classifier outputs.

────────────────────────────────────────────
LAYER 6 — GAT
────────────────────────────────────────────
Features from the transfer view with the training contract (afc column =
follicles at puncture, transferred = 1, frozen = max(Good−1, 0),
good_blast_rate = Good/2PN, training KPIScore, PRAI = KAT).
Exact top-k cosine graph (float64, deterministic ties); training neighbours
cached. Output: gnn_prob and ensemble_prob = w·GNN + (1−w)·KAT, w from the
model bundle (default 0.35).

────────────────────────────────────────────
LAYER 7 — BEFE
────────────────────────────────────────────
Prior: L1 per-transfer probability on the transfer view.
Evidence: KAT (τ from its bootstrap interval), GAT (τ from N_eff, attention
entropy, neighbour variance). CSDI and the cluster enter reliability, not the
evidence pool.
Posterior: precision-weighted logit pooling.

Uncertainty range (befe_app._uncertainty_range), logit scale:
  var_scenario = Var_i[ w_prior·logit(L1_i) + w_KAT·logit(KAT_i) + w_GAT·logit(GAT) ]
  var_model    = 1 / Σ_s 1/(1/τ_s + τ_b²)   (DerSimonian–Laird over L1, KAT, GAT)
  range        = sigmoid(logit(P_post) ± 1.96·√(var_scenario + var_model))

Reliability = 100·(0.40·consensus + 0.30·diffusion + 0.20·graph + 0.10·cluster),
scaled down for OOD (to at most −35%). Bands: High ≥ 70, Moderate ≥ 45.

OOD (models/befe_ood_stats.npz, schema 3): clinical [age, AFC];
embryology [ln(1+OCC), empirical-Bayes logits of MII/OCC, 2PN/MII, Blast/2PN];
fitted thresholds; AMH, BMI, KPIScore reported as not assessed. Schema 1 files
from fit_befe_ood.py ([age, AMH, AFC, BMI] / [OCC, MII, 2PN, Blast, KPI],
χ² thresholds) are still accepted. A file with a collapsed dimension is refused.
```

## Headline and cycle probability

`presentation.clinical_summary`:

- per transfer — the L7 posterior;
- 0 for the current cycle — when an entered OCC, MII, 2PN or blastocyst count
  is 0, or the euploid count is 0 with PGT-A.

`embryology.anchored_cycle_probability` shifts the scenario probabilities by
one logit constant so their mean over scenarios with a transfer equals the
headline, then applies the transfer model above. The result is shown as the
cycle probability and anchors TRP (see `docs/TRP_clinical_description.md`).

## KPI weight sensitivity

```python
res_fortune = run_pipeline(patient, kpi_weight=0.0)   # FORTUNE only
res_equal   = run_pipeline(patient, kpi_weight=0.5)   # default
res_kpi     = run_pipeline(patient, kpi_weight=1.0)   # KPI only
```

## Transfer scenario

```python
res_all = run_pipeline(patient)            # all blastocysts transferable
res_pgt = run_pipeline(patient, pgt=True)  # euploid embryos only
res["transfer_scenario"]                   # 'all_blastocysts' | 'pgt'
```
