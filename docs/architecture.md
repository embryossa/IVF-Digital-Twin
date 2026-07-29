# Architecture Documentation

## Complete data flow

```
PatientInput(age, AMH, AFC, BMI)
        │
        ▼  N = 5,000 Monte Carlo iterations
────────────────────────────────────────────
LAYER 1 — STOCHASTIC PIPELINE
────────────────────────────────────────────

S1  OKK[i]     ~ N(μ, σ) truncated [1, 50]
    μ = 3.25 + 1.20·AMH + 0.55·AFC − 0.15·age − 0.03·BMI
    σ = 0.22·μ + 1.2
    source: ART-ONE (CONSORT/ENGAGE/ESTHER trials)

S2  MII[i]     ~ Binomial(OKK[i], p_MII)
    logit(p_MII) = 2.4665 + 0.005·age − 0.782·stim + 0.24·AMH − 0.069
    source: Herasight Table A1 col 3, n=90,479 HFEA cycles

S3  2PN[i]     ~ Binomial(MII[i], p_fert)
    logit(p_fert) = 1.1678 + 0.004·age − 0.303·stim − 0.051
    source: Herasight Table A1 col 5, n=90,088 HFEA cycles

S4  Blast[i]   ~ Binomial(2PN[i], p_blast[i])
    p_blast[i] ~ N(clip(0.70 − 0.012·max(0, age−40), 0.30, 0.75), 0.06)
    source: Romanski 2022 (n=3,362) + Sainte-Rose 2021 (n=4,952)

S5  Good[i]    ~ Binomial(Blast[i], p_good[i])
    p_good[i]  ~ Beta(μ·10, (1−μ)·10)
    μ = clip(0.78 − 0.008·max(0, age−35), 0.40, 0.85)
    source: calibrated to Herasight Serdj report + clinical data

S6  Euploid[i] ~ Binomial(Good[i], p_eup[i])
    p_eup[i]   ~ Beta(mean·6, (1−mean)·6)
    mean = age-stratified lookup: <30→0.70, 30-34→0.65, 35-37→0.55,
           38-39→0.35, 40-41→0.18, ≥42→0.10
    source: Franasiak 2014 (n=15,169) + Armstrong 2023 (n=86,208)

S6b Warmed[i]  ~ Binomial(Euploid[i], 0.95)
    source: Coello 2021

────────────────────────────────────────────
LAYER 2 — PER-TRANSFER ENSEMBLE
────────────────────────────────────────────

FORTUNE component (per iteration):
  logit(p_F[i])   = 0.40 − 0.55·z_age + 0.15·z_AMH − 0.20·z_BMI + ε[i]
  ε[i] ~ N(0, 0.07)   [biological noise]
  source: FORTUNE (IVIRMA, PMID 40889782)

KPI component (per iteration):
  KPIScore[i] ∈ {5,...,25}
  Components: age(1/3/5) + AMH(1/3/5) + MII[i](1/3/5) + fertrate[i](1/3/5) + good[i](1/3/5)
  p_KPI[i] ~ Beta moment-matched to CI(KPIScore[i])

Ensemble:
  logit(p_ens[i]) = (1−w)·logit(p_F[i]) + w·logit(p_KPI[i])
  w = KPI_WEIGHT = 0.5 (default)

Three-level decomposition:
  [1] per-transfer  = mean(p_ens)
  [2] cum-if-viable = E[1−(1−p_ens)^n_tx | n_tx ≥ 1]
  [3] overall       = P(viable) × cum-if-viable

────────────────────────────────────────────
LAYER 3 — KAT NEURAL NETWORK ENSEMBLE
────────────────────────────────────────────

Architecture:
  KAN   (18 → 10 → 1, ReLU, BCE-with-logits, Adam)
  FTT   (18 features, feature tokeniser + transformer blocks)
  Ensemble: w_KAN·sigmoid(KAN) + w_FTT·p(FTT)
  Calibration: Venn-Abers conformal wrapper

NVSA correction:
  cf = clip(p_KPI_table(KPIScore[i]) / p_NN[i], 1/1.5, 1.5)
  p_adjusted[i] = clip(p_NN[i] · cf, 0, 1)

Bayesian posterior (Beta-Binomial):
  Prior: Beta(26, 74)
  Update with real clinic batches: Beta(α + Σs_j, β + Σ(t_j − s_j))
  NN as pseudo-obs: + int(p_NN_adjusted·100) successes / 100 trials
  Posterior mean = α' / (α' + β')

Per-attempt decay:
  Run NN forward 6× varying attempt_number ∈ {1,...,6}

────────────────────────────────────────────
LAYER 4 — CLUSTER CLASSIFIER
────────────────────────────────────────────

Feature vector (18-D) per iteration:
  [age, attempt, follicles, COCs, MII, 2PN, cleaving, HQ_blasts,
   day5, cryo, transferred, fert_rate, cleav_rate, blast_rate,
   TGBDR, retrieval_eff, KPI, NN_pred]

Standardisation:
  z[i] = (x[i] − centroid_mean) / pop_SD

Assignment:
  cluster[i] = argmin_c  ||z[i] − z_centroid_c||²

Centroids from Sergeev et al. (2024), 1,556 cycles, k-means k=3:
  C0 Standard (54%): AFC ~21, KPI ~24
  C1 Poor     (33%): AFC ~12, KPI ~18
  C2 High     (63%): AFC ~35, KPI ~25

Output:
  cluster_probs = {0: p0, 1: p1, 2: p2} where Σp = 1
  dominant_cluster = argmax(cluster_probs)
```

## Bayesian conditional updating

Any observed value can be passed as a `KnownValues` field. Each non-None
field creates a point-mass distribution at that stage, and all downstream
stages are re-sampled from the existing binomial filters.

```python
# Before retrieval: full stochastic pipeline
res_prior = run_pipeline_extended(patient, KnownValues())

# After retrieval: 12 oocytes observed
res_post_okk = run_pipeline_extended(patient, KnownValues(okk=12))

# After blastocyst culture: 8 blasts confirmed
res_post_blast = run_pipeline_extended(patient, KnownValues(okk=12, blasts=8))

# After PGT-A: 3 euploid
res_final = run_pipeline_extended(patient, KnownValues(okk=12, blasts=8, euploid=3))
```

## KPI weight sensitivity

```python
# FORTUNE only
res_fortune = run_pipeline(patient, kpi_weight=0.0)

# Equal weight (default)
res_equal = run_pipeline(patient, kpi_weight=0.5)

# KPI only
res_kpi = run_pipeline(patient, kpi_weight=1.0)
```
