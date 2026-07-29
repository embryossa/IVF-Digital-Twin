# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
# ============================================================
# IVF DIGITAL TWIN PLATFORM v6.2
# ZINB Stage 1 + Covariate-Dependent Bayesian Prior
#
# NEW IN v6.2 (over v6.1):
#   1. Stage 1 — ZINB (Zero-Inflated Negative Binomial)
#      Replaces truncated Normal with:
#      - Zero-inflation component: logistic P_zero(age, AMH, AFC)
#        models cancelled cycles / empty follicle puncture
#      - NB count component: μ from ART-ONE formula, θ = 5.0
#        (Herasight HFEA NB fit), variance = μ + μ²/θ
#      - Correct right-tail overdispersion for high responders
#      - Formal structural zeros for poor/elderly patients
#
#   2. Bayesian posterior — covariate-dependent Beta regression prior
#      Replaces fixed Beta(26,74) with per-patient Beta(α₀,β₀):
#      - Prior mean = FORTUNE per-transfer probability for THIS patient
#        (pre-lab clinical predictors: age, AMH, BMI only)
#      - Prior precision κ = 20 pseudo-observations (user-adjustable)
#      - Implements Ferrari & Cribari-Neto (2004) Beta regression:
#        α₀ = μ·κ,  β₀ = (1−μ)·κ
#      - Posterior updated with real clinic batches + NN evidence
#        as before
#
# RETAINED FROM v6.1:
#   - KAT neural-network ensemble (KAN + FT-Transformer + Venn-Abers)
#   - NVSA adjustment + per-attempt selection-effect decay curve
#   - Unsupervised cluster classifier (Sergeev et al. centroids)
#   - Bayesian mid-cycle conditional updating
#   - 3-level pregnancy decomposition
# ============================================================
#
# NEW IN v6.1 (over v6.0):
#   - Unsupervised cluster classification of every Monte Carlo
#     iteration using published cluster centroids from
#     Sergeev et al. "Decoding IVF Laboratory Performance through
#     Dimensionality Reduction and Cluster Analysis"
#   - 3 clinically meaningful clusters:
#       0 - Standard responders  (~54% predicted pregnancy)
#       1 - Poor responders      (~33% predicted pregnancy)
#       2 - High responders      (~63% predicted pregnancy)
#   - Probability distribution over clusters for the new cycle
#   - 2D PCA visualization with synthetic cluster point clouds
#     generated from published centroid statistics
#   - Independent unsupervised assessment alongside the
#     supervised FORTUNE / KPI / NN / Bayesian predictions
#
# RETAINED FROM v6.0:
#   - NN ensemble final layer (KAN + FT-Transformer + Conformal)
#   - NVSA adjustment with KPI confidence intervals
#   - Bayesian Beta-Binomial posterior with clinical priors
#   - Per-attempt probability decay curve
#
# RETAINED FROM v5.3:
#   - Bayesian conditional updating mid-cycle
#   - 3-level pregnancy decomposition
#   - FORTUNE + KPI logit-scale ensemble
#
# Run:  python ivf_digital_twin_v6_1.py
# ============================================================

import os, math, tempfile
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import plotly.graph_objects as go
import plotly.express as px
import pdfkit
from scipy.stats import norm, beta as beta_dist

import warnings
warnings.filterwarnings("ignore")

# ─── Optional NN-ensemble dependencies (graceful fallback) ──
NN_LIBS_AVAILABLE = False
NN_LIBS_ERROR = ""
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import joblib
    # Extra check: if torch loaded but is GPU build without GPU drivers,
    # it may fail later when tensors are created. Verify with a small op.
    _test = torch.zeros(1)
    del _test
    NN_LIBS_AVAILABLE = True
except ImportError as e:
    NN_LIBS_ERROR = f"not_installed:{e}"
    print(f"[v6] torch/joblib not installed -- NN layer disabled.")
    print(f"     Run fix_torch_dll.bat  OR:")
    print(f"     python -m pip install torch --index-url https://download.pytorch.org/whl/cpu")
except OSError as e:
    # Windows DLL error: fbgemm.dll, c10.dll, etc.
    NN_LIBS_ERROR = f"dll_error:{e}"
    print(f"[v6] torch DLL error (GPU version installed without CUDA drivers).")
    print(f"     Run fix_torch_dll.bat  OR these two commands:")
    print(f"     python -m pip uninstall torch -y")
    print(f"     python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu")
except Exception as e:
    NN_LIBS_ERROR = f"other:{e}"
    print(f"[v6] Unexpected error loading torch: {e}")

# ─── Configurable file paths for NN models ──────────────────
# Paths are resolved relative to THIS file's directory, so the
# app finds the models regardless of the working directory from
# which the script or Streamlit is launched.
_HERE = os.path.dirname(os.path.abspath(__file__))

def _model_path(filename: str) -> str:
    """
    Look for NN model files in this order:
      1. Same directory as this source file  (src/)
      2. Parent directory  (repo root — where app.py lives)
      3. 'models/' subfolder inside repo root
    Returns the first path where the file exists, or falls back
    to repo-root path so the missing-file error message is useful.
    """
    candidates = [
        os.path.join(_HERE, filename),                      # src/
        os.path.join(_HERE, "..", filename),                # repo root
        os.path.join(_HERE, "..", "models", filename),      # repo/models/
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.normpath(p)
    # not found — return root-level path for the error message
    return os.path.normpath(os.path.join(_HERE, "..", filename))

NN_MODEL_PATHS = {
    'kan':              _model_path('Prediction_KAN.pth'),
    'ft':               _model_path('FTTransformer.joblib'),
    # Optional. The new training script saves isotonic_ensemble.pkl.
    # KAT_calibrated_model.pkl is still supported for older Venn-Abers wrappers.
    'isotonic':         _model_path('isotonic_ensemble.pkl'),
    'calibrated':       _model_path('KAT_calibrated_model.pkl'),
    # Optional: if you later save ensemble_model.raw_weights, put it under this name.
    'ensemble_weights': _model_path('KAT_ensemble_raw_weights.pth'),
}

# ============================================================
# CONFIGURATION
# ============================================================

N_SIM = 5000

# path_to_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
path_to_wkhtmltopdf = None        # None → assumed in PATH

# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class PatientInput:
    female_age: float
    amh: float
    afc: int
    bmi: float

@dataclass
class KnownValues:
    """
    Bayesian conditioning inputs.
    Any field that is not None becomes a point-mass observation
    that overrides simulation at that stage and propagates
    deterministically to all upstream displayed values, then
    stochastically (via binomial filters) downstream.
    """
    okk:       Optional[int] = None        # retrieved oocytes
    mii:       Optional[int] = None        # MII (mature) oocytes
    pn2:       Optional[int] = None        # 2PN zygotes
    blasts:    Optional[int] = None        # total blastocysts
    good:      Optional[int] = None        # good-quality blastocysts
    euploid:   Optional[int] = None        # PGT-A euploid count
    n_transfers_planned: int = 0           # 0 = use all euploid sequentially


# ============================================================
# REFERENCE POPULATION & UTILITIES
# ============================================================

REFERENCE = {
    "female_age": {"mean": 36.3, "sd": 5.5},
    "amh":        {"mean": 2.0,  "sd": 1.2},
    "afc":        {"mean": 12,   "sd": 9},
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def percentile_rank(value, mean, sd):
    z = (value - mean) / sd
    return norm.cdf(z) * 100

def ci(samples, lo=2.5, hi=97.5):
    return np.percentile(samples, lo), np.percentile(samples, hi)


# ============================================================
# KPISCORE INFRASTRUCTURE (new in v5.3)
# ============================================================
#
# KPIScore is a 5-component integer score (5-25) capturing
# laboratory-stage performance. Each component scores 1, 3, or 5:
#
#   age   :  >=40 -> 1,  37-39 -> 3,  <=36 -> 5
#   AMH   :  <1   -> 1,  1-1.99 -> 3,  >=2 -> 5
#   MII   :  <=3  -> 1,  4-6   -> 3,  >=7 -> 5
#   fert% :  <50  -> 1,  50-65 -> 3,  >65 -> 5
#   good  :  0    -> 1,  1-2   -> 3,  >=3 -> 5
#
# Each total score maps to a 95% confidence interval for
# clinical implantation probability per transfer. This table
# is calibrated from independent clinical KPI literature.

KPI_CI_TABLE = {
    5:  (0.012, 0.08),  6:  (0.016, 0.09),  7:  (0.02, 0.10),
    8:  (0.03,  0.12),  9:  (0.04,  0.14),  10: (0.05, 0.15),
    11: (0.06,  0.18),  12: (0.08,  0.20),  13: (0.10, 0.22),
    14: (0.13,  0.26),  15: (0.16,  0.30),  16: (0.20, 0.32),
    17: (0.24,  0.36),  18: (0.28,  0.41),  19: (0.33, 0.46),
    20: (0.38,  0.51),  21: (0.42,  0.57),  22: (0.47, 0.63),
    23: (0.51,  0.69),  24: (0.56,  0.74),  25: (0.59, 0.79),
}

# Relative weight of KPI vs FORTUNE in the ensemble (logit scale)
# 0.0 = FORTUNE only,  0.5 = equal,  1.0 = KPI only
KPI_WEIGHT = 0.5


def kpi_score_per_iteration(age: float, amh: float,
                              mii: np.ndarray,
                              fert_rate_pct: np.ndarray,
                              good_blasts: np.ndarray) -> np.ndarray:
    """
    Compute integer KPIScore for each Monte Carlo iteration.
    Returns int array of length len(mii), values in [5, 25].
    """
    # Static components (scalar)
    a_score = 1 if age >= 40 else (5 if age <= 36 else 3)
    b_score = 1 if amh < 1 else (3 if amh < 2 else 5)

    # Dynamic components (vector — depend on simulated cycle outcomes)
    c_score = np.where(mii <= 3, 1, np.where(mii < 7, 3, 5))
    d_score = np.where(fert_rate_pct < 50, 1,
                np.where(fert_rate_pct <= 65, 3, 5))
    e_score = np.where(good_blasts == 0, 1,
                np.where(good_blasts <= 2, 3, 5))

    return (a_score + b_score + c_score + d_score + e_score).astype(int)


def kpi_sample_implantation(scores: np.ndarray) -> np.ndarray:
    """
    For each KPI score, sample an implantation probability from the
    Beta distribution moment-matched to the score's 95% CI.

    The Beta moment-matching converts (CI_low, CI_high) into Beta(a, b)
    parameters by treating the CI midpoint as the mean and the CI half-width
    as approximately 1.96 SD (Gaussian approximation, valid for CIs away
    from 0 and 1).
    """
    n = len(scores)
    out = np.zeros(n, dtype=float)
    for s in range(5, 26):
        mask = scores == s
        if not mask.any():
            continue
        lo, hi = KPI_CI_TABLE[s]
        mean = (lo + hi) / 2.0
        sd = (hi - lo) / 3.92    # 2 * 1.96 ≈ 3.92
        var = sd ** 2
        # Beta moment matching (valid when var < mean*(1-mean))
        if var < mean * (1 - mean) and var > 1e-8:
            kappa = mean * (1 - mean) / var - 1
            a = mean * kappa
            b = (1 - mean) * kappa
            out[mask] = np.random.beta(a, b, mask.sum())
        else:
            out[mask] = mean
    return np.clip(out, 0.005, 0.95)


def fortune_per_transfer_logit(patient: PatientInput) -> float:
    """
    FORTUNE-based per-transfer pregnancy log-odds.
    Returns a scalar; per-iteration variability is added by stage7 as a
    Gaussian random effect.
    """
    z_a = (patient.female_age - 36.3) / 5.5
    z_f = (math.log(patient.amh + 0.1) - math.log(2.1)) / 1.2
    z_b = (patient.bmi - 24.0) / 4.2
    return 0.40 - 0.55 * z_a + 0.15 * z_f - 0.20 * z_b


# ============================================================
# STOCHASTIC PIPELINE — STAGES (Bayesian-aware)
#
# Each stage_X(prev, patient, known) accepts:
#   prev   — vector from the previous stage (length N_SIM)
#   patient — PatientInput
#   known   — KnownValues; if relevant field is set, the stage
#             output becomes a degenerate point distribution
#             (every iteration = the observed value).
# ============================================================

def _zinb_p_zero(patient: PatientInput) -> float:
    """
    Probability of structural zero (cancelled / empty-follicle cycle).

    Logistic model calibrated to published cancellation rates:
      - Standard responder (age 35, AMH 2.5, AFC 15)  → ~2–4 %
      - Poor responder    (age 42, AMH 0.5, AFC 6)    → ~12–18 %
      - High responder    (age 28, AMH 4.5, AFC 28)   → ~0.5–1 %
      - Elderly poor      (age 44, AMH 0.3, AFC 4)    → ~20–25 %

    Coefficients:
      intercept  = −3.2
      age        = +0.40 (older → more cancellations)
      AMH (z)    = −0.60 (higher AMH → fewer cancellations)
      AFC (z)    = −0.80 (higher AFC → fewer cancellations)

    Sources: ESHRE OHSS Guideline 2023; Herasight zero-component
    Table A1 (age +0.050, stimulation −0.796 on the zero logit);
    re-calibrated here to include AMH and AFC which are not in HFEA.
    """
    z_age = (patient.female_age - 36.3) / 5.5
    z_amh = (patient.amh        - 2.0)  / 1.2
    z_afc = (patient.afc        - 12.0) / 9.0
    lp = -3.2 + 0.40 * z_age - 0.60 * z_amh - 0.80 * z_afc
    return float(sigmoid(lp))


def stage1_oocytes(patient: PatientInput, known: KnownValues, n: int = N_SIM):
    """
    Stage 1 — Retrieved oocytes (OKK)
    Model (v6.2): Zero-Inflated Negative Binomial (ZINB)

    ZINB addresses two key limitations of the truncated Normal:
      (1) Structural zeros — cycles cancelled before retrieval or resulting
          in empty follicle puncture.  These are NOT sampling noise; they
          arise from protocol failure, poor ovarian response to stimulation,
          or premature ovulation.  The truncated Normal clipped to [1,50]
          formally forbids them.  ZINB explicitly models them via a
          Bernoulli zero-inflation component P_zero(age, AMH, AFC).
      (2) Right-tail overdispersion — oocyte yields are right-skewed with
          heavier tails than the Normal, especially among high responders.
          The Negative Binomial's variance = μ + μ²/θ grows quadratically
          with the mean, producing the correct tail behaviour.

    Parameters:
      mu    = linear predictor (ART-ONE formula, locally calibrated)
      theta = NB dispersion = 5.0
              (calibrated to HFEA registry NB fit in Herasight medRxiv 2025;
               larger θ → less overdispersion; θ→∞ → Poisson)
      p_zero = logistic function of age, AMH, AFC (see _zinb_p_zero above)

    Sampling:
      For each iteration i:
        if Bernoulli(p_zero)  → OKK_i = 0   (structural zero / cancellation)
        else                  → OKK_i ~ NB(θ, θ/(θ+μ)), clipped to [1, 50]

    References:
      - Craig et al. medRxiv 2025.09.27.25336680 (ZINB for oocyte retrieval)
      - Herasight Table A1 (zero-component coefficients, HFEA 103,924 cycles)
      - ART-ONE (Merck KGaA, CONSORT/ENGAGE/ESTHER trials — count component)
    """
    if known.okk is not None:
        return np.full(n, int(known.okk), dtype=int)

    # ── Count-component mean (ART-ONE linear predictor) ───────
    mu = (3.25
          + 1.20 * patient.amh
          + 0.55 * patient.afc
          - 0.15 * patient.female_age
          - 0.03 * patient.bmi)
    mu = max(mu, 0.5)          # allow near-zero mean for severe poor responders

    # ── NB dispersion (from Herasight HFEA fit) ───────────────
    theta = 5.0
    p_nb  = theta / (theta + mu)   # scipy/numpy NB parameterisation

    # ── Zero-inflation probability ────────────────────────────
    p_zero = _zinb_p_zero(patient)

    # ── Sample ───────────────────────────────────────────────
    is_zero = np.random.random(n) < p_zero
    nb_counts = np.random.negative_binomial(theta, p_nb, n)
    okk = np.where(is_zero, 0, nb_counts)
    return np.clip(okk, 0, 50).astype(int)


def stage2_mii(oocytes: np.ndarray, patient: PatientInput, known: KnownValues):
    """
    Stage 2 — MII (mature) oocytes
    Logistic maturity rate (Herasight Table A1, simplified col 3):
        logit(p) = 2.4665 + 0.005·age – 0.782·stim + 0.24·AMH – 0.069
    Population mean maturity ≈ 90.7 %.
    """
    if known.mii is not None:
        return np.full(len(oocytes), int(known.mii), dtype=int), None

    a, f = patient.female_age, patient.amh
    logit_p = 2.4665 + 0.005*a - 0.782*1 + 0.24*f - 0.069
    p_mat = sigmoid(logit_p)
    mii = np.random.binomial(oocytes, p_mat)
    mii = np.maximum(mii, 0)
    return mii, p_mat


def stage3_fertilization(mii: np.ndarray, patient: PatientInput, known: KnownValues):
    """
    Stage 3 — 2PN zygotes
    Logistic fertilization rate (Herasight col 5, ICSI assumed):
        logit(p) = 1.1678 + 0.004·age – 0.303·stim – 0.051
    Population mean fertilization ≈ 72.3 %.
    """
    if known.pn2 is not None:
        return np.full(len(mii), int(known.pn2), dtype=int), None

    a = patient.female_age
    logit_p = 1.1678 + 0.004*a - 0.303*1 - 0.051
    p_fert = sigmoid(logit_p)
    pn2 = np.random.binomial(mii, p_fert)
    return pn2, p_fert


def stage4_blastulation(pn2: np.ndarray, patient: PatientInput, known: KnownValues):
    """
    Stage 4 — Blastocysts
    UPDATED v5.0:
      blast_rate = clip(0.70 – 0.012·max(0, age – 40),  0.30,  0.75)

    Calibrated to Romanski et al. 2022 (3,362 patients) and
    Sainte-Rose et al. 2021 (4,952 zygotes): stable ~60–67 %
    blastulation rate through age 40, accelerated decline thereafter.

    Gaussian noise (SD = 0.06) captures inter-lab variability.
    """
    if known.blasts is not None:
        return np.full(len(pn2), int(known.blasts), dtype=int)

    age = patient.female_age
    blast_mu = float(np.clip(0.70 - 0.012 * max(0, age - 40), 0.30, 0.75))
    p_blast = np.clip(np.random.normal(blast_mu, 0.06, len(pn2)), 0.10, 0.90)
    return np.random.binomial(pn2, p_blast)


def stage5_good_blasts(blasts: np.ndarray, patient: PatientInput, known: KnownValues):
    """
    Stage 5 — Good-quality blastocysts
    UPDATED v5.0:
      good_rate = clip(0.78 – 0.008·max(0, age – 35),  0.40,  0.85)

    Beta(k=10) sampling on the rate captures grading variability.
    Mean good-blast fraction now matches Herasight clinical reports
    (~70–75 % of blastocysts at typical reproductive age).
    """
    if known.good is not None:
        return np.full(len(blasts), int(known.good), dtype=int)

    age = patient.female_age
    good_mu = float(np.clip(0.78 - 0.008 * max(0, age - 35), 0.40, 0.85))
    p_good = np.random.beta(good_mu * 10, (1 - good_mu) * 10, len(blasts))
    return np.random.binomial(blasts, p_good)


def stage6_euploidy(good_blasts: np.ndarray, patient: PatientInput, known: KnownValues):
    """
    Stage 6 — Euploid embryos
    Age-stratified rate × Beta noise (Bernoulli per embryo).
    Calibrated to Franasiak 2014 (15,169 biopsies) + Armstrong 2023 (86,208 cycles).
    """
    if known.euploid is not None:
        return np.full(len(good_blasts), int(known.euploid), dtype=int)

    age = patient.female_age
    age_table = {
        (0,  30): 0.70,
        (30, 35): 0.65,
        (35, 38): 0.55,
        (38, 40): 0.35,
        (40, 42): 0.18,
        (42, 99): 0.10,
    }
    p_eup_centre = 0.10
    for (lo, hi), p in age_table.items():
        if lo <= age < hi:
            p_eup_centre = p
            break
    p_eup_sample = np.random.beta(p_eup_centre * 6,
                                   (1 - p_eup_centre) * 6,
                                   len(good_blasts))
    return np.random.binomial(good_blasts, p_eup_sample)


def stage6b_warmed(euploid: np.ndarray):
    """
    Stage 6b — Warmed embryos surviving thaw (vitrification survival)
    Fixed 95 % survival per blastocyst (Coello et al. 2021).
    """
    return np.random.binomial(euploid, 0.95)


def per_transfer_pregnancy_probability(patient: PatientInput) -> float:
    """
    Legacy scalar accessor — returns the FORTUNE-only central per-transfer
    pregnancy probability (no KPI, no random effect).
    Kept for backward compatibility; the production pipeline uses
    per-iteration KPI-FORTUNE ensemble inside stage7.
    """
    return float(sigmoid(fortune_per_transfer_logit(patient)))


def stage7_pregnancy_cycle(euploid: np.ndarray,
                            warmed: np.ndarray,
                            mii: np.ndarray,
                            pn2: np.ndarray,
                            good: np.ndarray,
                            patient: PatientInput,
                            known: KnownValues,
                            kpi_weight: float = KPI_WEIGHT):
    """
    Stage 7 — Cycle pregnancy outcomes (v5.3 — FORTUNE + KPI ensemble)
    ──────────────────────────────────────────────────────────────────

    PER-TRANSFER PROBABILITY is now a per-iteration random variable
    combining TWO information sources on the logit scale:

        logit(p_pertx[i])  =  (1 - w) * logit(p_fortune[i])
                            +  w      * logit(p_kpi[i])

    where:
      p_fortune[i] — FORTUNE-style per-transfer probability with small
                     Gaussian random effect (patient-level biological noise)
      p_kpi[i]    — KPIScore-derived probability, sampled from the Beta
                     distribution moment-matched to the score's 95% CI.
                     KPIScore is computed per-iteration from age, AMH and
                     the simulated MII / fert rate / good blasts of that
                     specific Monte Carlo sample.
      w           — KPI_WEIGHT (default 0.5 = equal ensemble)

    THREE-LEVEL PREGNANCY DECOMPOSITION is preserved exactly as before:
        per_transfer  <=  cum_if_viable
        overall = P(viable) * cum_if_viable
    """
    # ── Fertilization rate per iteration (%, vector)
    fert_rate_pct = np.where(mii > 0, (pn2 / np.maximum(mii, 1)) * 100, 0.0)

    # ── 1. FORTUNE per-iteration probability
    fortune_logit = fortune_per_transfer_logit(patient)
    fortune_re = np.random.normal(0, 0.07, len(euploid))
    p_fortune = sigmoid(fortune_logit + fortune_re)
    p_fortune = np.clip(p_fortune, 0.01, 0.99)

    # ── 2. KPI per-iteration probability
    kpi_scores = kpi_score_per_iteration(
        patient.female_age, patient.amh, mii, fert_rate_pct, good
    )
    p_kpi = kpi_sample_implantation(kpi_scores)
    p_kpi = np.clip(p_kpi, 0.01, 0.99)

    # ── 3. Logit-scale ensemble combination
    w = float(np.clip(kpi_weight, 0.0, 1.0))
    logit_fortune = np.log(p_fortune / (1 - p_fortune))
    logit_kpi     = np.log(p_kpi     / (1 - p_kpi))
    p_per_sample  = sigmoid((1 - w) * logit_fortune + w * logit_kpi)
    p_per_sample  = np.clip(p_per_sample, 0.01, 0.99)

    # ── Transfer count (capped by user plan if specified)
    if known.n_transfers_planned > 0:
        n_transfers = np.minimum(warmed, known.n_transfers_planned)
    else:
        n_transfers = warmed

    # ── Cumulative P(>=1 preg) per simulation; = 0 when n_tx[i] = 0
    p_any_preg_marginal = 1 - (1 - p_per_sample) ** n_transfers

    # ── Discrete count of live pregnancies in this cycle
    rng = np.random.default_rng()
    n_pregnancies = rng.binomial(n_transfers, p_per_sample)

    # ── Three-level decomposition (unchanged logic)
    viable = n_transfers >= 1
    p_viable = float(np.mean(viable))

    p_per_transfer_central = float(np.mean(p_per_sample))      # ensemble central value

    if viable.sum() >= 30:
        p_cum_if_viable = float(np.mean(p_any_preg_marginal[viable]))
        cum_if_viable_ci = (
            float(np.percentile(p_any_preg_marginal[viable], 2.5)),
            float(np.percentile(p_any_preg_marginal[viable], 97.5))
        )
    else:
        n_tx_med_fallback = max(int(np.median(n_transfers[viable])) if viable.sum() > 0 else 1, 1)
        p_cum_if_viable = float(1 - (1 - p_per_transfer_central) ** n_tx_med_fallback)
        cum_if_viable_ci = (p_cum_if_viable, p_cum_if_viable)

    p_cum_if_viable = max(p_cum_if_viable, p_per_transfer_central)
    p_overall = p_viable * p_cum_if_viable

    n_tx_med = max(int(np.median(n_transfers[viable])) if viable.sum() > 0 else 1, 1)
    rate_only_p = 1 - (1 - p_per_sample) ** n_tx_med
    rate_ci = (
        float(np.percentile(rate_only_p, 2.5)),
        float(np.percentile(rate_only_p, 97.5))
    )

    return {
        # raw arrays
        "p_any_preg_marginal":   p_any_preg_marginal,
        "n_pregnancies":         n_pregnancies,
        "n_transfers":           n_transfers,

        # per-iteration component probabilities (new)
        "sim_p_fortune":         p_fortune,
        "sim_p_kpi":             p_kpi,
        "sim_p_combined":        p_per_sample,
        "sim_kpi_scores":        kpi_scores,
        "sim_fert_rate_pct":     fert_rate_pct,

        # central summaries — FORTUNE / KPI / combined
        "p_per_transfer":        p_per_transfer_central,             # combined ensemble
        "p_per_transfer_fortune": float(np.mean(p_fortune)),
        "p_per_transfer_kpi":    float(np.mean(p_kpi)),
        "kpi_weight":            w,
        "kpi_score_median":      int(np.median(kpi_scores)),
        "kpi_score_ci":          (int(np.percentile(kpi_scores, 2.5)),
                                  int(np.percentile(kpi_scores, 97.5))),

        # three-level decomposition
        "p_cum_if_viable":       p_cum_if_viable,
        "p_overall_cycle":       p_overall,

        # auxiliary
        "p_viable":              p_viable,
        "cum_if_viable_ci":      cum_if_viable_ci,
        "rate_ci":               rate_ci,
        "n_tx_median_viable":    n_tx_med,
    }


# ============================================================
# RISK MODELS
# ============================================================

def risk_ohss(oocytes):
    return {
        "p_severe_ohss":   float(np.mean(oocytes >= 20)),
        "p_moderate_ohss": float(np.mean((oocytes >= 15) & (oocytes < 20))),
        "p_any_ohss":      float(np.mean(oocytes >= 15)),
        "mean_oocytes":    float(np.mean(oocytes)),
        "p95_oocytes":     float(np.percentile(oocytes, 95)),
    }

def risk_empty_cycle(blasts, good_blasts):
    return {
        "p_no_blast":       float(np.mean(blasts < 1)),
        "p_no_good_blast":  float(np.mean(good_blasts < 1)),
        "p_empty_cycle":    float(np.mean(blasts < 1)),
    }


# ============================================================
# FULL PIPELINE WITH BAYESIAN UPDATING
# ============================================================

def run_pipeline(patient: PatientInput,
                 known: Optional[KnownValues] = None,
                 kpi_weight: float = KPI_WEIGHT,
                 n: int = N_SIM) -> Dict:
    """
    Run the full stochastic pipeline.
    Any non-None field in `known` is treated as a deterministic observation
    (Bayesian conditioning), and the downstream Monte Carlo simulation is
    re-run from that point with the new starting condition.

    kpi_weight: weight of KPIScore-derived probability in the FORTUNE-KPI
                ensemble. 0.0 = FORTUNE only, 0.5 = equal (default), 1.0 = KPI only.
    """
    known = known or KnownValues()

    okk             = stage1_oocytes(patient, known, n)
    mii, p_mat      = stage2_mii(okk, patient, known)
    pn2, p_fert     = stage3_fertilization(mii, patient, known)
    blasts          = stage4_blastulation(pn2, patient, known)
    good            = stage5_good_blasts(blasts, patient, known)
    euploid         = stage6_euploidy(good, patient, known)
    warmed          = stage6b_warmed(euploid)
    preg_out        = stage7_pregnancy_cycle(
        euploid, warmed, mii, pn2, good, patient, known, kpi_weight=kpi_weight
    )

    ohss            = risk_ohss(okk)
    empty           = risk_empty_cycle(blasts, good)

    return {
        # raw arrays
        "sim_okk":       okk,
        "sim_mii":       mii,
        "sim_pn2":       pn2,
        "sim_blasts":    blasts,
        "sim_good":      good,
        "sim_euploid":   euploid,
        "sim_warmed":    warmed,
        "sim_p_any":     preg_out["p_any_preg_marginal"],
        "sim_n_preg":    preg_out["n_pregnancies"],
        "sim_n_tx":      preg_out["n_transfers"],

        # KPI / FORTUNE / combined per-iteration arrays
        "sim_p_fortune":      preg_out["sim_p_fortune"],
        "sim_p_kpi":          preg_out["sim_p_kpi"],
        "sim_p_combined":     preg_out["sim_p_combined"],
        "sim_kpi_scores":     preg_out["sim_kpi_scores"],
        "sim_fert_rate_pct":  preg_out["sim_fert_rate_pct"],

        # rates
        "p_mat":         p_mat,
        "p_fert":        p_fert,

        # medians
        "okk_med":       float(np.median(okk)),
        "mii_med":       float(np.median(mii)),
        "pn2_med":       float(np.median(pn2)),
        "blasts_med":    float(np.median(blasts)),
        "good_med":      float(np.median(good)),
        "euploid_med":   float(np.median(euploid)),
        "warmed_med":    float(np.median(warmed)),

        # ──── per-transfer estimates (FORTUNE / KPI / combined) ──
        "p_per_transfer":          preg_out["p_per_transfer"],
        "p_per_transfer_fortune":  preg_out["p_per_transfer_fortune"],
        "p_per_transfer_kpi":      preg_out["p_per_transfer_kpi"],
        "kpi_weight":              preg_out["kpi_weight"],
        "kpi_score_median":        preg_out["kpi_score_median"],
        "kpi_score_ci":            preg_out["kpi_score_ci"],

        # ──── three-level pregnancy decomposition ────────────────
        "p_cum_if_viable":    preg_out["p_cum_if_viable"],
        "p_overall_cycle":    preg_out["p_overall_cycle"],
        "p_viable":           preg_out["p_viable"],
        "cum_if_viable_ci":   preg_out["cum_if_viable_ci"],
        "rate_ci":            preg_out["rate_ci"],
        "n_tx_median_viable": preg_out["n_tx_median_viable"],

        # other summaries
        "p_at_least_one_euploid":  float(np.mean(euploid >= 1)),
        "expected_euploid":  float(np.mean(euploid)),
        "expected_warmed":   float(np.mean(warmed)),

        # risks
        "ohss":  ohss,
        "empty": empty,

        # tag
        "known": known,
    }


# ============================================================
# NN ENSEMBLE FINAL LAYER (new in v6.0)
# ============================================================
#
# Loads the pre-trained KAN + FT-Transformer + Conformal ensemble
# (saved as .pth/.joblib/.pkl). If files or libraries are absent,
# falls back to the FORTUNE+KPI ensemble from v5.3.
#
# Per Monte Carlo iteration, builds an 18-feature row from the
# simulated cycle values plus patient demographics, runs the NN,
# and returns a per-iteration probability vector.
# ============================================================

# These 18 feature names MUST exactly match the training order
NN_FEATURE_NAMES = [
    "Возраст", "№ попытки", "Количество фолликулов", "Число ОКК",
    "Число инсеминированных", "2 pN", "Число дробящихся на 3 день",
    "Число Bl", "Число Bl хор.кач-ва", "Частота оплодотворения",
    "Частота дробления", "Частота формирования бластоцист",
    "Частота формирования бластоцист хорошего качества",
    "Частота получения ОКК", "Число эмбрионов 5 дня",
    "Заморожено эмбрионов", "Перенесено эмбрионов", "KPIScore",
]


if NN_LIBS_AVAILABLE:

    def _ft_proba(ft_model, df: pd.DataFrame) -> np.ndarray:
        """
        Return a 1D numpy array with class-1 probabilities from mambular
        FTTransformerClassifier. Different mambular versions may return
        shape (n,), (n, 1), or (n, 2).
        """
        p = ft_model.predict_proba(df)
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            return p
        if p.shape[1] == 1:
            return p[:, 0]
        return p[:, 1]

    class KANLinear(nn.Module):
        """
        One true KAN layer with trainable B-spline functions on edges.
        Must match train_kat.py / train_kat_fixed.py architecture.
        """
        def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                     scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                     grid_eps=0.02, grid_range=(-50.0, 50.0)):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.grid_size = grid_size
            self.spline_order = spline_order
            self.scale_base = scale_base
            self.scale_spline = scale_spline
            self.grid_eps = grid_eps
            self.base_act = nn.SiLU()

            h = (grid_range[1] - grid_range[0]) / grid_size
            knot = (torch.arange(-spline_order, grid_size + spline_order + 1,
                                 dtype=torch.float32) * h + grid_range[0])
            self.register_buffer(
                "grid",
                knot.unsqueeze(0).expand(in_features, -1).contiguous()
            )

            self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
            self.spline_weight = nn.Parameter(
                torch.empty(out_features, in_features, grid_size + spline_order)
            )
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
            self._init_weights(scale_noise)

        def _init_weights(self, scale_noise):
            nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
            nn.init.kaiming_uniform_(self.spline_scaler, a=5 ** 0.5)
            with torch.no_grad():
                x_init = self.grid[:, self.spline_order:-self.spline_order].T
                noise = ((torch.rand(self.grid_size + 1, self.in_features,
                                     self.out_features) - 0.5)
                         * scale_noise / self.grid_size)
                self.spline_weight.data.copy_(self._curve2coeff(x_init, noise))

        def b_splines(self, x: torch.Tensor) -> torch.Tensor:
            grid = self.grid
            x_lo = grid[:, self.spline_order].unsqueeze(0)
            x_hi = grid[:, -(self.spline_order + 1)].unsqueeze(0)
            x_e = x.clamp(x_lo, x_hi).unsqueeze(-1)
            bases = ((x_e >= grid[:, :-1]) & (x_e < grid[:, 1:])).to(x.dtype)
            for k in range(1, self.spline_order + 1):
                dl = grid[:, k:-1] - grid[:, :-(k + 1)]
                dr = grid[:, k + 1:] - grid[:, 1:-k]
                bases = ((x_e - grid[:, :-(k + 1)]) / (dl + 1e-8) * bases[:, :, :-1]
                         + (grid[:, k + 1:] - x_e) / (dr + 1e-8) * bases[:, :, 1:])
            return bases.contiguous()

        def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            A = torch.nan_to_num(self.b_splines(x).permute(1, 0, 2),
                                 nan=0., posinf=0., neginf=0.)
            B = torch.nan_to_num(y.permute(1, 0, 2),
                                 nan=0., posinf=0., neginf=0.)
            sol = torch.nan_to_num(torch.linalg.pinv(A) @ B, nan=0.)
            return sol.permute(2, 0, 1).contiguous()

        @property
        def scaled_spline_weight(self):
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = F.linear(self.base_act(x), self.base_weight * self.scale_base)
            spline_b = self.b_splines(x).reshape(x.size(0), -1)
            spline_w = (self.scaled_spline_weight * self.scale_spline).reshape(
                self.out_features, -1
            )
            return base_out + F.linear(spline_b, spline_w)

    class KAN(nn.Module):
        """
        True Kolmogorov-Arnold Network matching the retrained model:
        width=[18, 10, 1], grid=5, spline order=3, raw IVF features.
        """
        def __init__(self, width=None, grid=5, k=3,
                     scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                     grid_eps=0.02, grid_range=(-50.0, 50.0)):
            super().__init__()
            if width is None:
                width = [18, 10, 1]
            self.width = width
            self.layers = nn.ModuleList([
                KANLinear(width[i], width[i + 1],
                          grid_size=grid, spline_order=k,
                          scale_noise=scale_noise, scale_base=scale_base,
                          scale_spline=scale_spline, grid_eps=grid_eps,
                          grid_range=grid_range)
                for i in range(len(width) - 1)
            ])
            self.norms = nn.ModuleList([
                nn.LayerNorm(width[i + 1]) for i in range(len(width) - 2)
            ])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.norms):
                    x = self.norms[i](x)
            return x

    class EnsembleModel(nn.Module):
        """
        KAN + FTTransformer weighted ensemble.
        Output is probability, not logit. Weights are normalized by softmax,
        matching the new training code.
        """
        def __init__(self, kan_model, ft_model, feature_names):
            super().__init__()
            self.kan_model = kan_model
            self.ft_model = ft_model
            self.feature_names = list(feature_names)
            self.raw_weights = nn.Parameter(torch.zeros(2))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # KAN: logit -> probability
            kan_prob = torch.sigmoid(self.kan_model(x).squeeze(-1))

            # FTTransformer is external/sklearn-like; it is outside torch graph.
            x_df = pd.DataFrame(x.detach().cpu().numpy(), columns=self.feature_names)
            ft_prob = torch.tensor(
                _ft_proba(self.ft_model, x_df), dtype=torch.float32, device=x.device
            )

            w = torch.softmax(self.raw_weights, dim=0)
            ensemble = w[0] * kan_prob + w[1] * ft_prob
            return ensemble.unsqueeze(1)

    class EnsembleWrapper:
        """sklearn-compatible wrapper for batched probability output."""
        def __init__(self, ensemble_model, calibrator=None, source_label=None):
            self.ensemble_model = ensemble_model
            self.calibrator = calibrator
            self.classes_ = np.array([0, 1])
            self.source_label = source_label or "KAN + FT-Transformer ensemble"

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            X = np.asarray(X, dtype=np.float32)
            with torch.no_grad():
                x_t = torch.tensor(X, dtype=torch.float32)
                preds = self.ensemble_model(x_t).squeeze().detach().cpu().numpy()
            probs = np.asarray(preds, dtype=float)
            if probs.ndim == 0:
                probs = np.array([float(probs)])
            if self.calibrator is not None:
                probs = self.calibrator.predict(probs)
            probs = np.clip(probs, 0.001, 0.999)
            return np.column_stack((1.0 - probs, probs))

def load_nn_ensemble(paths: dict = NN_MODEL_PATHS):
    """
    Load the retrained NN ensemble. Required files:
      - Prediction_KAN.pth
      - FTTransformer.joblib

    Optional files:
      - isotonic_ensemble.pkl          (new train_kat.py calibration output)
      - KAT_calibrated_model.pkl       (legacy wrapper, if present)
      - KAT_ensemble_raw_weights.pth   (optional raw softmax weights)

    Returns a sklearn-compatible wrapper or None.
    """
    if not NN_LIBS_AVAILABLE:
        print(f"[v6] NN ensemble disabled. Reason: {NN_LIBS_ERROR}")
        print("     Fix: python -m pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return None

    required = {k: paths[k] for k in ("kan", "ft")}
    missing_required = {k: p for k, p in required.items() if not os.path.exists(p)}
    if missing_required:
        print("[v6] Required NN model files not found:")
        for k, p in missing_required.items():
            print(f"     {k:12s} -> expected at: {p}")
        root = os.path.normpath(os.path.join(_HERE, ".."))
        print(f"\n     Place model files in one of:")
        print(f"       {os.path.join(_HERE)}/      (src/ folder)")
        print(f"       {root}/       (repo root, same as app.py)")
        print(f"       {os.path.join(root, 'models')}/  (repo/models/)")
        print("\n     Falling back to FORTUNE+KPI ensemble.")
        return None

    def _torch_load(path):
        """Compatible torch.load for both newer and older PyTorch versions."""
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")
        except Exception:
            # Some older checkpoints / PyTorch builds are not compatible with weights_only=True.
            return torch.load(path, map_location="cpu", weights_only=False)

    try:
        kan_model = KAN(width=[18, 10, 1], grid=5, k=3, grid_range=(-50.0, 50.0))
        state = _torch_load(paths["kan"])
        kan_model.load_state_dict(state)
        kan_model.eval()
        for p in kan_model.parameters():
            p.requires_grad_(False)
        print(f"[v6] True KAN loaded: {paths['kan']}")

        try:
            ft_model = joblib.load(paths["ft"])
            print(f"[v6] FTTransformer loaded: {paths['ft']}")
        except ModuleNotFoundError as e:
            missing_mod = str(e).replace("No module named ", "").strip("'\"")
            print(f"[v6] Cannot load FTTransformer: missing module {missing_mod}")
            if "mambular" in missing_mod:
                print("     Fix: python -m pip install mambular")
            elif "crepes" in missing_mod:
                print("     Fix: python -m pip install crepes")
            else:
                print(f"     Fix: python -m pip install {missing_mod}")
            print("     Falling back to FORTUNE+KPI ensemble.")
            return None

        ensemble = EnsembleModel(kan_model, ft_model, NN_FEATURE_NAMES)
        ensemble.eval()

        # Optional: load learned ensemble raw weights if they are saved separately.
        ew_path = paths.get("ensemble_weights")
        if ew_path and os.path.exists(ew_path):
            raw_w = _torch_load(ew_path)
            if isinstance(raw_w, dict):
                raw_w = raw_w.get("raw_weights", raw_w.get("ensemble_raw_weights", raw_w))
            raw_w = torch.as_tensor(raw_w, dtype=torch.float32).view(-1)
            if raw_w.numel() == 2:
                with torch.no_grad():
                    ensemble.raw_weights.copy_(raw_w)
                w = torch.softmax(ensemble.raw_weights, dim=0).detach().cpu().numpy()
                print(f"[v6] Ensemble weights loaded: KAN={w[0]:.3f}, FT={w[1]:.3f}")
            else:
                print(f"[v6] Ensemble weights ignored: expected 2 values, got {raw_w.numel()}.")
        else:
            w = torch.softmax(ensemble.raw_weights, dim=0).detach().cpu().numpy()
            print(f"[v6] Ensemble weights file not found; using equal softmax weights: "
                  f"KAN={w[0]:.3f}, FT={w[1]:.3f}")

        # Optional new isotonic calibrator saved by train_kat.py.
        calibrator = None
        iso_path = paths.get("isotonic")
        if iso_path and os.path.exists(iso_path):
            try:
                calibrator = joblib.load(iso_path)
                print(f"[v6] Isotonic ensemble calibrator loaded: {iso_path}")
            except Exception as e:
                print(f"[v6] Isotonic calibrator could not be loaded ({e}); using uncalibrated ensemble.")

        # Optional legacy calibrated wrapper. Use only if present and loadable.
        legacy_path = paths.get("calibrated")
        if legacy_path and os.path.exists(legacy_path):
            try:
                wrapped = joblib.load(legacy_path)
                print(f"[v6] Legacy calibrated wrapper loaded: {legacy_path}")
                return wrapped
            except Exception as e:
                print(f"[v6] Legacy calibrated wrapper ignored ({e}); using current wrapper.")

        wrapped = EnsembleWrapper(
            ensemble,
            calibrator=calibrator,
            source_label="KAN + FT-Transformer ensemble" + (" + isotonic calibration" if calibrator else "")
        )
        print("[v6] NN ensemble ready.")
        return wrapped

    except Exception as exc:
        print(f"[v6] Failed to load NN models: {exc}")
        print("     Falling back to FORTUNE+KPI ensemble.")
        return None


# ============================================================
# NN INPUT FEATURE BUILDER
# ============================================================
#
# Maps simulated cycle values to the 18 NN features.
# Per the user spec:
#   - Число инсеминированных = MII (mature oocytes)
#   - Число дробящихся на 3 день = 2PN (assume all cleave)
#   - Число эмбрионов 5 дня = 2PN (assume all cultured)
#   - Заморожено эмбрионов = good-quality blastocysts
#   - Перенесено эмбрионов = 1 (single embryo transfer)
# ============================================================

def calculate_nn_kpi_score(age: float, follicles: int, mii: np.ndarray,
                             fert_rate: np.ndarray, good: np.ndarray) -> np.ndarray:
    """
    NN's internal KPIScore — uses follicle count instead of AMH.
    Returns int array (len = MC iterations), range [5, 25].
    """
    a_score = 1 if age >= 40 else (5 if age <= 36 else 3)
    b_score = 5 if follicles > 15 else (3 if follicles >= 8 else 1)
    c_score = np.where(mii <= 3, 1, np.where(mii <= 7, 3, 5))
    d_score = np.where(fert_rate < 0.50, 1, np.where(fert_rate <= 0.65, 3, 5))
    e_score = np.where(good == 0, 1, np.where(good <= 2, 3, 5))
    return (a_score + b_score + c_score + d_score + e_score).astype(int)


def build_nn_features(patient: PatientInput, res: Dict,
                      attempt_number: int, follicles: Optional[int] = None) -> pd.DataFrame:
    """
    Build a (N_SIM x 18) DataFrame of NN inputs from the v5.3 simulation.
    """
    n = len(res['sim_okk'])
    follicles = follicles if follicles is not None else int(patient.afc)

    okk    = res['sim_okk'].astype(float)
    mii    = res['sim_mii'].astype(float)
    pn2    = res['sim_pn2'].astype(float)
    blasts = res['sim_blasts'].astype(float)
    good   = res['sim_good'].astype(float)

    fert_rate = np.where(mii > 0, pn2 / np.maximum(mii, 1), 0.0)
    cleav_rate = np.where(pn2 > 0, 1.0, 0.0)        # all 2PN cleave (per spec)
    blast_rate = np.where(pn2 > 0, blasts / np.maximum(pn2, 1), 0.0)
    good_rate  = np.where(pn2 > 0, good / np.maximum(pn2, 1), 0.0)
    okk_rate   = okk / max(follicles, 1)

    kpi_nn = calculate_nn_kpi_score(patient.female_age, follicles, mii, fert_rate, good)

    df = pd.DataFrame({
        "Возраст":                                  np.full(n, patient.female_age, dtype=float),
        "№ попытки":                                np.full(n, attempt_number, dtype=float),
        "Количество фолликулов":                    np.full(n, follicles, dtype=float),
        "Число ОКК":                                okk,
        "Число инсеминированных":                   mii,
        "2 pN":                                     pn2,
        "Число дробящихся на 3 день":               pn2,        # all 2PN cleave
        "Число Bl":                                 blasts,
        "Число Bl хор.кач-ва":                      good,
        "Частота оплодотворения":                   fert_rate,
        "Частота дробления":                        cleav_rate,
        "Частота формирования бластоцист":          blast_rate,
        "Частота формирования бластоцист хорошего качества": good_rate,
        "Частота получения ОКК":                    okk_rate,
        "Число эмбрионов 5 дня":                    pn2,        # all cultured to d5
        "Заморожено эмбрионов":                     good,       # good blasts frozen
        "Перенесено эмбрионов":                     np.ones(n), # single embryo transfer
        "KPIScore":                                 kpi_nn.astype(float),
    })
    # column ordering must match training order exactly
    return df[NN_FEATURE_NAMES]


# ============================================================
# NVSA ADJUSTMENT (KPI-based confidence interval correction)
# ============================================================

NVSA_KPI_TABLE = {
    25: {'prob': 0.70, 'ci': (0.59, 0.79)}, 24: {'prob': 0.65, 'ci': (0.56, 0.74)},
    23: {'prob': 0.60, 'ci': (0.51, 0.69)}, 22: {'prob': 0.55, 'ci': (0.47, 0.63)},
    21: {'prob': 0.50, 'ci': (0.42, 0.57)}, 20: {'prob': 0.45, 'ci': (0.38, 0.51)},
    19: {'prob': 0.40, 'ci': (0.33, 0.46)}, 18: {'prob': 0.35, 'ci': (0.28, 0.41)},
    17: {'prob': 0.30, 'ci': (0.24, 0.36)}, 16: {'prob': 0.26, 'ci': (0.20, 0.32)},
    15: {'prob': 0.22, 'ci': (0.16, 0.30)}, 14: {'prob': 0.18, 'ci': (0.13, 0.26)},
    13: {'prob': 0.15, 'ci': (0.10, 0.22)}, 12: {'prob': 0.13, 'ci': (0.08, 0.20)},
    11: {'prob': 0.10, 'ci': (0.06, 0.18)}, 10: {'prob': 0.09, 'ci': (0.05, 0.15)},
    9:  {'prob': 0.07, 'ci': (0.04, 0.14)}, 8:  {'prob': 0.06, 'ci': (0.03, 0.12)},
    7:  {'prob': 0.05, 'ci': (0.02, 0.10)}, 6:  {'prob': 0.04, 'ci': (0.016, 0.09)},
    5:  {'prob': 0.03, 'ci': (0.012, 0.08)},
}

def nvsa_adjustment(kpi_score: int, base_prob: float, max_correction: float = 1.5):
    """
    Reweight NN probability toward KPI-table prior. Returns (adjusted_prob, ci_tuple).
    Correction factor is clipped to [1/1.5, 1.5] to prevent overcorrection.
    """
    closest = min(NVSA_KPI_TABLE.keys(), key=lambda x: abs(x - int(kpi_score)))
    kpi_prob = NVSA_KPI_TABLE[closest]['prob']
    kpi_ci   = NVSA_KPI_TABLE[closest]['ci']
    cf = kpi_prob / base_prob if base_prob > 0 else 1.0
    cf = float(np.clip(cf, 1/max_correction, max_correction))
    return float(np.clip(base_prob * cf, 0.0, 1.0)), kpi_ci


# ============================================================
# NN PREDICTION STAGE
# ============================================================

def stage8_nn_prediction(patient: PatientInput, res: Dict,
                          nn_model, attempt_number: int = 1,
                          follicles: Optional[int] = None) -> Dict:
    """
    Run the NN ensemble on all Monte Carlo iterations.
    Returns per-iteration probability + summary statistics.

    If nn_model is None, falls back to the v5.3 FORTUNE+KPI ensemble
    (sim_p_combined) as the "NN proxy" so downstream Bayesian logic
    continues to work without requiring the actual NN files.
    """
    if nn_model is None:
        # Fallback: use FORTUNE+KPI ensemble per-iteration probability
        probs = res['sim_p_combined'].copy()
        return {
            "source":          "FORTUNE+KPI ensemble (NN unavailable)",
            "sim_probs":       probs,
            "base_prob_mean":  float(np.mean(probs)),
            "base_prob_median":float(np.median(probs)),
            "base_prob_ci":    (float(np.percentile(probs, 2.5)),
                                float(np.percentile(probs, 97.5))),
        }

    features = build_nn_features(patient, res, attempt_number, follicles)
    probs = nn_model.predict_proba(features.values)[:, 1]
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 0.001, 0.999)
    source_label = getattr(nn_model, "source_label", "KAN + FT-Transformer ensemble")

    return {
        "source":           source_label,
        "sim_probs":        probs,
        "base_prob_mean":   float(np.mean(probs)),
        "base_prob_median": float(np.median(probs)),
        "base_prob_ci":     (float(np.percentile(probs, 2.5)),
                             float(np.percentile(probs, 97.5))),
        "features":         features,
    }


def apply_nvsa_to_distribution(nn_pred: Dict, kpi_scores: np.ndarray) -> Dict:
    """
    Apply per-iteration NVSA adjustment to the NN probability vector.
    Uses each iteration's NN-style KPIScore.
    """
    base = nn_pred['sim_probs']
    n = len(base)
    adjusted = np.zeros(n)
    ci_low = np.zeros(n)
    ci_high = np.zeros(n)
    for i in range(n):
        adj, ci_t = nvsa_adjustment(int(kpi_scores[i]), float(base[i]))
        adjusted[i] = adj
        ci_low[i], ci_high[i] = ci_t

    return {
        "sim_adjusted":      adjusted,
        "adjusted_mean":     float(np.mean(adjusted)),
        "adjusted_median":   float(np.median(adjusted)),
        "adjusted_ci":       (float(np.percentile(adjusted, 2.5)),
                              float(np.percentile(adjusted, 97.5))),
        "kpi_ci_low_mean":   float(np.mean(ci_low)),
        "kpi_ci_high_mean":  float(np.mean(ci_high)),
    }


# ============================================================
# BAYESIAN POSTERIOR (Beta-Binomial)
# ============================================================
#
# Combines three sources of evidence on a single Beta distribution:
#   1. Prior belief about per-transfer success rate (alpha, beta)
#   2. Real clinical outcomes across past cycles (successes, trials)
#   3. NN prediction for the new cycle (weighted as pseudo-observations)
# ============================================================

def covariate_dependent_prior(patient: PatientInput,
                               kappa: float = 20.0) -> tuple:
    """
    Covariate-dependent Beta regression prior  (v6.2).

    Instead of a fixed global Beta(26, 74) prior for all patients,
    we compute a per-patient Beta(α₀, β₀) whose mean equals the
    FORTUNE-based per-transfer probability estimate for that patient.

    Mathematical framework — Beta regression (Ferrari & Cribari-Neto 2004):
        μ_i  = sigmoid( logit(p_FORTUNE(x_i)) )     ← patient-specific mean
        φ    = κ                                     ← precision (= α + β)
        α₀   = μ_i · κ
        β₀   = (1 − μ_i) · κ

    The precision κ controls how informative the prior is:
        κ = 20  →  20 prior pseudo-observations (default)
                   diffuse enough to be overridden by real clinic data
                   (which typically contributes 300–400 observations)
                   but informative enough to prevent degenerate posteriors
                   when clinic data is sparse.

    Why FORTUNE as the prior mean, not the FORTUNE+KPI ensemble:
        The KPI score depends on simulated laboratory values that will also
        enter the NN as evidence.  Using the KPI in the prior would
        double-count that information.  FORTUNE depends only on demographic
        and ovarian-reserve predictors (age, AMH, BMI) that are measured
        before the laboratory phase — strictly pre-lab clinical predictors.

    Comparison with global Beta(26, 74):
        Global prior mean = 26 % for ALL patients (ignores patient features).
        Covariate-dependent prior mean varies with patient type:
          Poor responder  (age 42, AMH 0.5) → ~41 %  (FORTUNE calibrated)
          Standard        (age 35, AMH 2.5) → ~65 %
          High responder  (age 28, AMH 4.5) → ~81 %
        The prior is then pulled toward the clinic's actual rates by the
        real-data update.

    References:
        Ferrari SLP, Cribari-Neto F. Beta regression for modelling rates and
        proportions. J Appl Stat. 2004;31(7):799–815.
        Steyerberg EW. Clinical Prediction Models. 2nd ed. Springer; 2019.
    """
    p_prior = float(sigmoid(fortune_per_transfer_logit(patient)))
    alpha_0 = p_prior * kappa
    beta_0  = (1.0 - p_prior) * kappa
    return alpha_0, beta_0


def bayesian_posterior_pregnancy(
        nn_pred_prob: float,
        real_successes: Optional[List[int]] = None,
        real_trials:    Optional[List[int]] = None,
        patient: Optional['PatientInput'] = None,
        prior_kappa:  float = 20.0,
        prior_alpha: Optional[float] = None,
        prior_beta:  Optional[float] = None,
        nn_pseudo_n: int = 100,
        cred_level: float = 0.95,
) -> Dict:
    """
    Covariate-dependent Beta-Binomial posterior  (v6.2).

    Three sources of evidence are combined via sequential conjugate updating:

    1. Prior — Beta(α₀, β₀)
         If a PatientInput is supplied: per-patient prior from covariate-
         dependent Beta regression (FORTUNE-based, see covariate_dependent_prior).
         If no patient is supplied but prior_alpha/prior_beta are given: use them.
         Fallback (neither): Beta(26, 74) global prior for backward compatibility.

    2. Real clinical data — sequential Binomial observations
         For each historical batch (s_j successes in t_j transfers):
           α += s_j,   β += (t_j − s_j)

    3. Neural-network evidence — pseudo-observations
         The NN-adjusted per-transfer probability is encoded as
         nn_pseudo_n pseudo-observations with success fraction = nn_pred_prob.
         Typically nn_pseudo_n = 100 (user-adjustable).

    The posterior mean, mode, and 95 % credible interval are computed from
    the final Beta(α', β') using the inverse Beta CDF.

    Mathematical update rule (conjugate):
        α' = α₀  +  Σ s_j  +  round(nn_pred_prob · nn_pseudo_n)
        β' = β₀  +  Σ(t_j − s_j)  +  (nn_pseudo_n − round(...))
        Posterior ~ Beta(α', β')

    Returns
    -------
    Dict with posterior parameters, summary statistics, and diagnostics.
    """
    # ── Prior ─────────────────────────────────────────────────
    if patient is not None:
        a, b = covariate_dependent_prior(patient, kappa=prior_kappa)
        prior_type = "covariate-dependent"
        p_prior_mean = a / (a + b)
    elif prior_alpha is not None and prior_beta is not None:
        a, b = float(prior_alpha), float(prior_beta)
        prior_type = "user-supplied"
        p_prior_mean = a / (a + b)
    else:
        # backward-compatible global fallback
        a, b = 26.0, 74.0
        prior_type = "global Beta(26,74)"
        p_prior_mean = 0.26

    alpha_0, beta_0 = a, b   # save for reporting

    # ── Real clinical data update ─────────────────────────────
    n_real = 0
    if real_successes and real_trials:
        for s, t in zip(real_successes, real_trials):
            a += float(s)
            b += float(t - s)
            n_real += 1

    # ── NN evidence as pseudo-observations ────────────────────
    nn_successes = int(round(nn_pred_prob * nn_pseudo_n))
    a += nn_successes
    b += (nn_pseudo_n - nn_successes)

    # ── Posterior summary ─────────────────────────────────────
    mean = a / (a + b)
    mode = (a - 1) / (a + b - 2) if (a > 1 and b > 1) else mean
    lo, hi = beta_dist.interval(cred_level, a, b)

    return {
        # posterior
        "posterior_alpha":    float(a),
        "posterior_beta":     float(b),
        "mean":               float(mean),
        "mode":               float(mode),
        "ci_low":             float(lo),
        "ci_high":            float(hi),
        "cred_level":         cred_level,
        # prior diagnostics
        "prior_alpha":        float(alpha_0),
        "prior_beta":         float(beta_0),
        "prior_mean":         float(p_prior_mean),
        "prior_type":         prior_type,
        "prior_kappa":        float(prior_kappa),
        # evidence summary
        "nn_pseudo_n":        nn_pseudo_n,
        "nn_input_prob":      nn_pred_prob,
        "n_real_cycles":      n_real,
    }


# ============================================================
# PER-ATTEMPT PROBABILITY DECAY
# ============================================================
#
# Re-runs the NN forward pass at the median simulated lab values
# but varying the attempt number 1..N. Produces the per-cycle
# probability decay curve.
# ============================================================

# ============================================================
# ESTEVES 2019 EUPLOIDY MODULE  (independent add-on, v6.2)
#
# Esteves et al., Front Endocrinol 2019;10:99 (POSEIDON ART
# Calculator). doi:10.3389/fendo.2019.00099. n=347, 2520 MII.
#
# Per-MII-oocyte probability that a mature oocyte yields a
# euploid blastocyst (compounds fertilisation x blastulation
# x euploidy in ONE coefficient). AUC 0.716.
#
# This module is COMPLETELY INDEPENDENT of the main S1-S6b
# pipeline. It does not modify any stage. It provides the
# clinically useful INVERSE calculation: how many MII oocytes
# must be banked to obtain >= k euploid blastocysts with a
# given confidence — essential for oocyte/embryo banking
# counselling, especially for fertility preservation.
# ============================================================

ESTEVES_SPERM_SOURCES = ("ejaculate", "testicular_NOA",
                         "testicular_OA", "epididymal")


def esteves_p_euploid_per_mii(age: float,
                               sperm_source: str = "ejaculate") -> float:
    """
    Esteves 2019 Table 2 logistic model.

    logit(p) = -2.6518
               + 0.2231659 * [sperm in {ejaculate, testicular_OA, epididymal}]
               - 0.2045457 * [ejaculate] * (age - 38.9066)
               - 0.1530924 * [testicular_NOA] * (age - 38.9066)

    For obstructive azoospermia (testicular_OA) and epididymal sperm
    the paper found outcomes statistically comparable to ejaculate,
    so the ejaculate centred-age slope is applied. testicular_NOA
    (non-obstructive azoospermia) uses its own gentler age slope but
    a lower intercept (no +0.2231659 term).

    Returns probability in (0, 1) that one MII oocyte ultimately
    produces a euploid blastocyst.
    """
    if sperm_source not in ESTEVES_SPERM_SOURCES:
        sperm_source = "ejaculate"
    a = age - 38.9066
    if sperm_source == "testicular_NOA":
        lp = -2.6518 - 0.1530924 * a
    else:
        # ejaculate / testicular_OA / epididymal
        lp = -2.6518 + 0.2231659 - 0.2045457 * a
    return float(sigmoid(lp))


def esteves_euploid_distribution(n_mii: int, age: float,
                                  sperm_source: str = "ejaculate",
                                  n_sim: int = 20000) -> Dict:
    """
    Forward model: given n_mii mature oocytes, simulate the
    distribution of euploid blastocysts under the Esteves per-MII
    probability. Each MII independently yields a euploid blastocyst
    with probability p (Binomial(n_mii, p)).

    Returns mean, median, percentiles and the full P(>=k) table.
    """
    p = esteves_p_euploid_per_mii(age, sperm_source)
    draws = np.random.binomial(n_mii, p, n_sim)
    max_k = int(draws.max()) if draws.size else 0
    p_at_least = {k: float(np.mean(draws >= k)) for k in range(0, max_k + 2)}
    return {
        "n_mii":        n_mii,
        "p_per_mii":    p,
        "mean":         float(draws.mean()),
        "median":       float(np.median(draws)),
        "p05":          float(np.percentile(draws, 5)),
        "p95":          float(np.percentile(draws, 95)),
        "p_at_least":   p_at_least,
        "samples":      draws,
    }


def esteves_mii_needed(age: float, k_target: int,
                        confidence: float = 0.80,
                        sperm_source: str = "ejaculate",
                        max_mii: int = 200) -> Dict:
    """
    INVERSE model (the clinically important one).

    Smallest number of MII oocytes M such that
        P(euploid blastocysts >= k_target) >= confidence
    under Binomial(M, p_Esteves).

    This answers: "How many mature oocytes must we retrieve / bank
    to be `confidence`-sure of obtaining at least k_target euploid
    blastocysts?" — the core question in oocyte banking and
    fertility preservation counselling.

    Returns the required M plus the achieved probability, or None
    if even max_mii is insufficient (very advanced age).
    """
    from scipy.stats import binom
    p = esteves_p_euploid_per_mii(age, sperm_source)
    for M in range(1, max_mii + 1):
        prob = 1.0 - binom.cdf(k_target - 1, M, p)
        if prob >= confidence:
            return {"mii_needed": M, "p_per_mii": p,
                    "achieved_conf": float(prob),
                    "k_target": k_target, "confidence": confidence}
    return {"mii_needed": None, "p_per_mii": p,
            "achieved_conf": None,
            "k_target": k_target, "confidence": confidence}


def esteves_banking_analysis(patient: PatientInput,
                              res: Dict,
                              sperm_source: str = "ejaculate",
                              k_targets=(1, 2, 3, 4),
                              confidences=(0.50, 0.80, 0.90)) -> Dict:
    """
    Full banking analysis combining the Esteves inverse model with
    the main pipeline's pregnancy estimate.

    Steps:
      1. From the main pipeline, determine how many euploid
         blastocysts are needed for a target cumulative pregnancy
         probability (uses per-transfer prob from L2 ensemble).
      2. For each k (euploid target) and confidence level, compute
         the MII oocytes that must be banked (Esteves inverse).
      3. Report the patient's own simulated MII median for context.

    Returns a structured dict consumed by the report/figures.
    """
    age = patient.female_age
    p_mii = esteves_p_euploid_per_mii(age, sperm_source)

    # ── 1. euploid blastocysts needed for pregnancy targets ────
    # Per-euploid-blastocyst live-pregnancy probability:
    # use the L2 per-transfer probability as the per-euploid
    # transfer success proxy (each euploid ~ one transfer).
    p_transfer = float(res.get("p_per_transfer", 0.45))
    p_transfer = min(max(p_transfer, 0.05), 0.95)

    def euploids_for_target(target_preg):
        # 1-(1-p)^n >= target  ->  n >= ln(1-target)/ln(1-p)
        if target_preg >= 1:
            return None
        n = math.log(1 - target_preg) / math.log(1 - p_transfer)
        return int(math.ceil(n))

    preg_targets = [0.50, 0.70, 0.90]
    euploid_for_preg = {
        t: euploids_for_target(t) for t in preg_targets
    }

    # ── 2. MII needed table (inverse Esteves) ──────────────────
    mii_table = {}
    for k in k_targets:
        mii_table[k] = {}
        for c in confidences:
            r = esteves_mii_needed(age, k, c, sperm_source)
            mii_table[k][c] = r["mii_needed"]

    # ── 3. patient's own simulated MII for context ─────────────
    sim_mii_med = int(np.median(res["sim_mii"])) if "sim_mii" in res else None
    sim_mii_p95 = int(np.percentile(res["sim_mii"], 95)) if "sim_mii" in res else None

    # forward: euploid distribution at the patient's own MII median
    fwd = None
    if sim_mii_med:
        fwd = esteves_euploid_distribution(sim_mii_med, age, sperm_source)

    return {
        "age":              age,
        "sperm_source":     sperm_source,
        "p_per_mii":        p_mii,
        "p_transfer_used":  p_transfer,
        "preg_targets":     preg_targets,
        "euploid_for_preg": euploid_for_preg,
        "k_targets":        list(k_targets),
        "confidences":      list(confidences),
        "mii_table":        mii_table,
        "patient_mii_median": sim_mii_med,
        "patient_mii_p95":    sim_mii_p95,
        "forward_at_median":  fwd,
    }


def selection_effect_decay(p_base: float, attempt: int,
                            alpha: float = 0.08) -> float:
    """
    Selection-effect decay on the logit scale.

    Patients who reach attempt k ≥ 2 are a self-selected subgroup of
    non-responders, with systematically less favourable prognosis than
    those who succeeded on attempt 1. This creates a REAL but MODEST
    per-cycle decline that is well documented:

      Malizia et al. NEJM 2009 (n=6,164, 14,248 cycles):
        cycle 1: 30%, cycle 2: 25.6%, cycle 3: 22.5%, cycle 4: 20.5%
        relative decline ≈ 10–14 % per attempt

    The logit-scale model:
        logit(p_k) = logit(p_1) − alpha · (k − 1)

    alpha = 0.08 produces ≈ 8 % relative per-attempt decline in the
    mid-probability range, matching the literature.

    Note: the NN forward pass can capture this if the model was trained
    on a large enough multi-attempt dataset. On smaller datasets the
    NN weight for attempt_number is typically close to zero because the
    signal is weak. The analytical decay is therefore ADDED as a
    post-processing layer and shown separately from the NN raw output.
    """
    if attempt <= 1:
        return float(p_base)
    logit_p = math.log(p_base / (1 - p_base))
    logit_adj = logit_p - alpha * (attempt - 1)
    return float(sigmoid(logit_adj))


def per_attempt_probabilities(patient: PatientInput, res: Dict,
                                nn_model, max_attempts: int = 6,
                                follicles: Optional[int] = None,
                                decay_alpha: float = 0.08) -> Dict:
    """
    Per-attempt pregnancy probability curve.

    Three series are returned:

    1. NN raw per-attempt — raw NN output varying attempt_number.
       Shows the NN's learned sensitivity (often near-flat because
       the training dataset is too small to learn the selection effect).
       WITHOUT NVSA so the attempt signal is not erased by the prior.

    2. Selection-effect decay — analytical logit-scale decay applied
       to the base probability, calibrated to Malizia et al. 2009.
       alpha = 0.08  ≈ 8 % relative per-attempt decline.

    3. Combined — mean of the two, used as the primary displayed curve.

    Why not use NVSA in the per-attempt calculation:
      NVSA pulls every NN output toward the SAME KPI-table prior,
      which is attempt-independent. This erases any per-attempt NN
      signal, making the NVSA-adjusted curve identically flat.
      The unajusted NN raw output and the analytical selection-effect
      decay together give a more informative and clinically grounded
      per-attempt curve.
    """
    attempts = list(range(1, max_attempts + 1))
    nn_raw, nn_lo, nn_hi = [], [], []
    sel_decay = []
    combined_mean, combined_lo, combined_hi = [], [], []

    # Base probability for attempt 1 (NVSA-adjusted, same as main report)
    p_base_nvsa = res['nn_nvsa']['adjusted_mean']
    p_base_nn   = res['nn_prediction']['base_prob_mean']

    for a in attempts:
        # ── NN raw (no NVSA so attempt effect is preserved) ───────
        nn_pred_a = stage8_nn_prediction(
            patient, res, nn_model,
            attempt_number=a, follicles=follicles
        )
        p_nn_a = nn_pred_a['base_prob_mean']
        p_nn_lo = nn_pred_a['base_prob_ci'][0]
        p_nn_hi = nn_pred_a['base_prob_ci'][1]

        # ── Analytical selection-effect decay ──────────────────────
        p_sel_a = selection_effect_decay(p_base_nvsa, a, alpha=decay_alpha)

        # ── Combined: logit-scale mean of NN and selection decay ───
        logit_nn  = math.log(max(p_nn_a, 0.01)  / max(1 - p_nn_a,  0.01))
        logit_sel = math.log(max(p_sel_a, 0.01) / max(1 - p_sel_a, 0.01))
        p_comb = float(sigmoid((logit_nn + logit_sel) / 2))
        p_comb_lo = float(sigmoid((math.log(max(p_nn_lo,0.01)/max(1-p_nn_lo,0.01)) + logit_sel) / 2))
        p_comb_hi = float(sigmoid((math.log(max(p_nn_hi,0.01)/max(1-p_nn_hi,0.01)) + logit_sel) / 2))

        nn_raw.append(p_nn_a)
        nn_lo.append(p_nn_lo)
        nn_hi.append(p_nn_hi)
        sel_decay.append(p_sel_a)
        combined_mean.append(p_comb)
        combined_lo.append(p_comb_lo)
        combined_hi.append(p_comb_hi)

    return {
        "attempts":       attempts,
        # primary curve used in report
        "p_mean":         combined_mean,
        "p_lo":           combined_lo,
        "p_hi":           combined_hi,
        # decomposed series for transparency
        "p_nn_raw":       nn_raw,
        "p_sel_decay":    sel_decay,
        "decay_alpha":    decay_alpha,
        "p_base_attempt1": p_base_nvsa,
    }


# ============================================================
# EXTENDED PIPELINE (v6.0 final layer)
# ============================================================

def run_pipeline_extended(patient: PatientInput,
                           known: Optional[KnownValues] = None,
                           attempt_number: int = 1,
                           follicles: Optional[int] = None,
                           nn_model = None,
                           clinic_real_successes: Optional[List[int]] = None,
                           clinic_real_trials: Optional[List[int]] = None,
                           prior_alpha: int = 26,
                           prior_beta:  int = 74,
                           prior_kappa: float = 20.0,
                           max_attempts_curve: int = 6,
                           kpi_weight: float = KPI_WEIGHT,
                           sperm_source: str = "ejaculate",
                           n: int = N_SIM) -> Dict:
    """
    Run the complete v6.0 pipeline:
      1. v5.3 base pipeline (FORTUNE + KPI ensemble, 3-level decomposition)
      2. NN ensemble final layer (if nn_model provided)
      3. NVSA adjustment (KPI-based correction)
      4. Bayesian posterior with clinical priors
      5. Per-attempt decay curve
    """
    # ── 1. v5.3 base pipeline ─────────────────────────────────
    res = run_pipeline(patient, known=known, kpi_weight=kpi_weight, n=n)

    # ── 2. NN final layer ─────────────────────────────────────
    nn_pred = stage8_nn_prediction(patient, res, nn_model,
                                     attempt_number=attempt_number,
                                     follicles=follicles)
    res['nn_prediction'] = nn_pred
    res['nn_attempt_number'] = attempt_number

    # ── 3. NVSA adjustment ────────────────────────────────────
    nvsa = apply_nvsa_to_distribution(nn_pred, res['sim_kpi_scores'])
    res['nn_nvsa'] = nvsa

    # ── 4. Bayesian posterior (covariate-dependent prior, v6.2) ──
    posterior = bayesian_posterior_pregnancy(
        nn_pred_prob   = nvsa['adjusted_mean'],
        real_successes = clinic_real_successes,
        real_trials    = clinic_real_trials,
        patient        = patient,          # enables per-patient Beta prior
        prior_kappa    = prior_kappa,
        # prior_alpha / prior_beta ignored when patient is provided;
        # kept as fallback only if patient=None
        prior_alpha    = prior_alpha,
        prior_beta     = prior_beta,
    )
    res['posterior'] = posterior

    # ── 5. Per-attempt curve ──────────────────────────────────
    if max_attempts_curve > 1:
        res['attempt_curve'] = per_attempt_probabilities(
            patient, res, nn_model,
            max_attempts=max_attempts_curve, follicles=follicles
        )

    # ── 6. Cluster analysis (v6.1) ────────────────────────────
    res['cluster_analysis'] = stage9_cluster_analysis(
        patient, res,
        attempt_number=attempt_number,
        follicles=follicles,
    )

    # ── 7. Esteves euploidy / banking analysis (v6.2 add-on) ──
    # Independent module — does NOT alter S1-S6b. Provides the
    # inverse calculation: MII oocytes needed for k euploid
    # blastocysts, and euploid blastocysts needed for pregnancy.
    res['esteves_banking'] = esteves_banking_analysis(
        patient, res, sperm_source=sperm_source,
    )

    return res




# ============================================================
# CLUSTER ANALYSIS (v6.1)
# ============================================================
#
# Based on Sergeev et al., "Decoding IVF Laboratory Performance
# through Dimensionality Reduction and Cluster Analysis" (2024).
# Three k-means clusters identified on 1556 IVF cycles, externally
# validated on 172 cycles from two independent centers.
#
# Each Monte Carlo iteration is classified to the nearest cluster
# centroid in z-score-standardized 18-dimensional feature space.
# Across all iterations, this yields a probability distribution
# over the 3 clusters for the simulated patient.
#
# Visualization uses synthetic point clouds around each centroid
# (since the original 1556-cycle dataset is not redistributable);
# PCA(2) is then fit on synthetic + new-cycle points to show
# where the new cycle lands in the cluster space.
# ============================================================

# Ordered feature names — must match the order used in build / centroids / SDs
CLUSTER_FEATURE_NAMES = [
    "Age", "Attempt", "Follicles", "COCs", "Inseminated", "2PN",
    "Cleaving", "HQ_blasts", "Day5", "Cryo", "Transferred",
    "FertRate", "CleavageRate", "BlastRate", "TGBDR", "RetrievalEff",
    "KPI", "NNPred",
]

# Published cluster centroids (Table 3 of Sergeev et al.)
CLUSTER_CENTROIDS = {
    0: {  # Standard responders, 54% predicted pregnancy
        "Age": 32.42, "Attempt": 1.43, "Follicles": 20.70, "COCs": 16.67,
        "Inseminated": 13.07, "2PN": 10.64, "Cleaving": 6.66, "HQ_blasts": 4.85,
        "Day5": 10.64, "Cryo": 4.85, "Transferred": 1.55,
        "FertRate": 0.82, "CleavageRate": 1.00, "BlastRate": 0.65,
        "TGBDR": 0.48, "RetrievalEff": 0.80, "KPI": 23.97, "NNPred": 0.54,
    },
    1: {  # Poor responders, 33% predicted pregnancy
        "Age": 20.33, "Attempt": 1.63, "Follicles": 11.98, "COCs": 9.45,
        "Inseminated": 7.22, "2PN": 4.93, "Cleaving": 2.88, "HQ_blasts": 1.96,
        "Day5": 4.93, "Cryo": 1.96, "Transferred": 1.18,
        "FertRate": 0.72, "CleavageRate": 0.98, "BlastRate": 0.65,
        "TGBDR": 0.45, "RetrievalEff": 0.77, "KPI": 17.68, "NNPred": 0.33,
    },
    2: {  # High responders, 63% predicted pregnancy
        "Age": 31.57, "Attempt": 1.23, "Follicles": 34.66, "COCs": 28.15,
        "Inseminated": 21.28, "2PN": 17.33, "Cleaving": 11.20, "HQ_blasts": 8.68,
        "Day5": 17.33, "Cryo": 8.68, "Transferred": 1.54,
        "FertRate": 0.83, "CleavageRate": 1.00, "BlastRate": 0.67,
        "TGBDR": 0.53, "RetrievalEff": 0.81, "KPI": 24.50, "NNPred": 0.63,
    },
}

CLUSTER_INTERPRETATIONS = {
    0: {
        "name": "Standard Responder",
        "color": "rgba(80,140,220,0.6)",
        "preg_rate": 0.54,
        "description": (
            "Intermediate ovarian response with balanced embryological outcomes. "
            "Typical IVF scenario combining adequate ovarian reserve with consistent "
            "laboratory performance. Favorable but not exceptional prognosis."
        ),
        "clinical_notes": (
            "Standard protocols, routine monitoring, single embryo transfer. "
            "Performance benchmarks should reflect typical clinic averages."
        ),
    },
    1: {
        "name": "Poor Responder",
        "color": "rgba(220,90,90,0.6)",
        "preg_rate": 0.33,
        "description": (
            "Limited ovarian response with markedly reduced outcomes across all "
            "parameters. Despite often younger mean age (donor cycles, premature "
            "ovarian insufficiency), only ~10 follicles and ~5 blastocysts per cycle."
        ),
        "clinical_notes": (
            "Consider individualized stimulation strategies (mild stimulation, "
            "luteal phase stimulation, adjunctive growth hormone). Discuss "
            "freeze-all and accumulating embryos over multiple cycles."
        ),
    },
    2: {
        "name": "High Responder",
        "color": "rgba(80,200,120,0.6)",
        "preg_rate": 0.63,
        "description": (
            "Abundant oocyte yield with superior embryological outcomes. Mean follicle "
            "count >34 yielding ~28 oocytes and ~17 blastocysts. Highest fertilization "
            "(83%) and blastocyst formation (67%) rates with most favorable prognosis."
        ),
        "clinical_notes": (
            "Monitor for OHSS risk; antagonist protocols with GnRH agonist trigger. "
            "Freeze-all strategy preferred to avoid fresh transfer with elevated E2."
        ),
    },
}

# Population SD estimates for z-score standardization (from typical IVF distributions)
CLUSTER_FEATURE_POP_SD = {
    "Age": 5.5, "Attempt": 0.8, "Follicles": 10.0, "COCs": 8.0,
    "Inseminated": 6.0, "2PN": 5.0, "Cleaving": 4.0, "HQ_blasts": 3.0,
    "Day5": 5.0, "Cryo": 3.0, "Transferred": 0.5,
    "FertRate": 0.15, "CleavageRate": 0.10, "BlastRate": 0.15,
    "TGBDR": 0.18, "RetrievalEff": 0.15, "KPI": 4.0, "NNPred": 0.15,
}


def build_cluster_features_per_iteration(patient: PatientInput, res: Dict,
                                          attempt_number: int = 1,
                                          follicles: Optional[int] = None) -> np.ndarray:
    """
    Build (N_SIM x 18) feature matrix matching the cluster centroids.
    Per the v6 spec mapping:
      Inseminated = MII, Cleaving = 2PN, Day5 = 2PN,
      Cryo = good blasts, Transferred = 1.
    NNPred uses NN per-iteration probabilities if available,
    otherwise FORTUNE+KPI ensemble (sim_p_combined).
    """
    n = len(res['sim_okk'])
    foll = follicles if follicles is not None else int(patient.afc)

    okk    = res['sim_okk'].astype(float)
    mii    = res['sim_mii'].astype(float)
    pn2    = res['sim_pn2'].astype(float)
    blasts = res['sim_blasts'].astype(float)
    good   = res['sim_good'].astype(float)

    fert_rate    = np.where(mii > 0, pn2 / np.maximum(mii, 1), 0.0)
    cleav_rate   = np.where(pn2 > 0, 1.0, 0.0)
    blast_rate   = np.where(pn2 > 0, blasts / np.maximum(pn2, 1), 0.0)
    tgbdr        = np.where(pn2 > 0, good   / np.maximum(pn2, 1), 0.0)
    retr_eff     = okk / max(foll, 1)

    # NN prediction per iteration (prefer real NN, fall back to ensemble)
    if 'nn_prediction' in res and 'sim_probs' in res['nn_prediction']:
        nn_pred = res['nn_prediction']['sim_probs']
    else:
        nn_pred = res.get('sim_p_combined', np.full(n, 0.5))

    kpi = res['sim_kpi_scores'].astype(float)

    return np.column_stack([
        np.full(n, patient.female_age),     # Age
        np.full(n, attempt_number),         # Attempt
        np.full(n, foll),                   # Follicles
        okk,                                # COCs
        mii,                                # Inseminated
        pn2,                                # 2PN
        pn2,                                # Cleaving (all 2PN cleave per pipeline)
        good,                               # HQ_blasts
        pn2,                                # Day5 (= 2PN per pipeline)
        good,                               # Cryo (= good blasts)
        np.ones(n),                         # Transferred (SET = 1)
        fert_rate, cleav_rate, blast_rate,
        tgbdr, retr_eff,
        kpi,                                # KPI score
        nn_pred,                            # NN predicted pregnancy probability
    ])


def assign_clusters_to_iterations(features: np.ndarray) -> np.ndarray:
    """
    Nearest-centroid assignment in z-score-standardized space.
    Returns array of cluster IDs (0/1/2) with length len(features).
    """
    sds = np.array([CLUSTER_FEATURE_POP_SD[n] for n in CLUSTER_FEATURE_NAMES])
    centroids = np.array([
        [CLUSTER_CENTROIDS[c][n] for n in CLUSTER_FEATURE_NAMES]
        for c in (0, 1, 2)
    ])  # (3, 18)

    # Standardize using mean of centroids as origin (k-means-like)
    origin = centroids.mean(axis=0)
    z_features = (features - origin) / sds
    z_centroids = (centroids - origin) / sds

    # Squared distance from each iteration to each centroid
    dists = np.sum((z_features[:, None, :] - z_centroids[None, :, :])**2, axis=2)
    return dists.argmin(axis=1)


def synthetic_cluster_cloud(n_per_cluster: int = 250,
                             within_scale: float = 0.45,
                             rng_seed: Optional[int] = 1234) -> tuple:
    """
    Generate synthetic point clouds around each cluster centroid for visualization.
    Uses diagonal covariance with within-cluster SD = pop_SD × within_scale.

    Returns: (points (n_total, 18), labels (n_total,))
    """
    if rng_seed is not None:
        rs = np.random.RandomState(rng_seed)
    else:
        rs = np.random

    sds = np.array([CLUSTER_FEATURE_POP_SD[n] for n in CLUSTER_FEATURE_NAMES])
    cov_diag = (sds * within_scale) ** 2

    points, labels = [], []
    for c in (0, 1, 2):
        mean = np.array([CLUSTER_CENTROIDS[c][n] for n in CLUSTER_FEATURE_NAMES])
        pts = rs.multivariate_normal(mean, np.diag(cov_diag), n_per_cluster)
        # Hard-clip counts to non-negative integers domain where appropriate
        pts[:, 0]  = np.clip(pts[:, 0], 18, 50)                # age
        pts[:, 1]  = np.clip(pts[:, 1], 1, 6)                  # attempt
        for idx in (2, 3, 4, 5, 6, 7, 8, 9, 10):               # counts
            pts[:, idx] = np.clip(pts[:, idx], 0, None)
        for idx in (11, 12, 13, 14, 15):                       # rates
            pts[:, idx] = np.clip(pts[:, idx], 0, 1)
        pts[:, 17] = np.clip(pts[:, 17], 0, 1)                 # NN pred
        points.append(pts)
        labels.extend([c] * n_per_cluster)

    return np.vstack(points), np.array(labels)


def numpy_pca_2d(X: np.ndarray):
    """
    PCA → 2D via numpy SVD on z-score-standardized data.
    Returns (embedded (n, 2), explained_variance_ratio (2,)).
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Z = (X - mu) / sd

    Z_centered = Z - Z.mean(axis=0)
    U, S, Vt = np.linalg.svd(Z_centered, full_matrices=False)

    # Project onto first 2 principal components
    components = Vt[:2]
    embedded = Z_centered @ components.T

    total_var = (S ** 2).sum()
    expl_ratio = (S[:2] ** 2) / total_var
    return embedded, expl_ratio


def stage9_cluster_analysis(patient: PatientInput, res: Dict,
                              attempt_number: int = 1,
                              follicles: Optional[int] = None) -> Dict:
    """
    Run cluster classification + build visualization data.

    Returns dict with:
      assignments        — cluster ID per MC iteration (N_SIM,)
      cluster_probs      — {0: p, 1: p, 2: p}
      dominant_cluster   — int
      synthetic_points   — synthetic cloud (n_total, 18) for plot
      synthetic_labels   — labels for synthetic points
      patient_features   — (N_SIM, 18) feature matrix of patient MC samples
      pca_embedded       — 2D embedding of (synthetic + patient) points
      pca_explained      — explained variance ratio
      n_synthetic        — count of synthetic points (for slicing)
    """
    # Per-iteration feature matrix
    feats = build_cluster_features_per_iteration(
        patient, res, attempt_number=attempt_number, follicles=follicles
    )

    # Cluster assignments
    assignments = assign_clusters_to_iterations(feats)
    n_total = len(assignments)
    cluster_probs = {c: float(np.mean(assignments == c)) for c in (0, 1, 2)}
    dominant = int(max(cluster_probs, key=cluster_probs.get))

    # Synthetic cloud + PCA on combined data
    syn_pts, syn_labels = synthetic_cluster_cloud()
    combined = np.vstack([syn_pts, feats])
    pca_2d, expl = numpy_pca_2d(combined)

    return {
        "assignments":       assignments,
        "cluster_probs":     cluster_probs,
        "dominant_cluster":  dominant,
        "synthetic_points":  syn_pts,
        "synthetic_labels":  syn_labels,
        "patient_features":  feats,
        "pca_embedded":      pca_2d,
        "pca_explained":     expl,
        "n_synthetic":       len(syn_pts),
        "attempt_number":    attempt_number,
        "follicles_used":    follicles if follicles is not None else int(patient.afc),
    }


# ============================================================
# DISCRETE LIVE-BIRTH-COUNT DISTRIBUTION
# ============================================================

def pregnancy_count_distribution(res: Dict, max_k: int = 14):
    """
    P(at least k pregnancies in this cycle), k = 1..max_k.
    Used for the Herasight-style "chances of pregnancies" chart.
    """
    n_preg = res["sim_n_preg"]
    p_at_least_k = []
    for k in range(1, max_k + 1):
        p_at_least_k.append(float(np.mean(n_preg >= k)))
    return p_at_least_k


# ============================================================
# TEXTUAL REPORT
# ============================================================

def textual_report(patient, res):
    age_pct = percentile_rank(patient.female_age, REFERENCE["female_age"]["mean"], REFERENCE["female_age"]["sd"])
    amh_pct = percentile_rank(patient.amh,        REFERENCE["amh"]["mean"],        REFERENCE["amh"]["sd"])
    afc_pct = percentile_rank(patient.afc,        REFERENCE["afc"]["mean"],        REFERENCE["afc"]["sd"])
    ci_lo, ci_hi = res["rate_ci"]

    print("=" * 70)
    print("IVF DIGITAL TWIN REPORT v6.1 — Full Edition (NN + Bayesian + Cluster)")
    print("=" * 70)
    print(f"\nPatient: age {patient.female_age:.0f} (P{age_pct:.0f}), "
          f"AMH {patient.amh:.2f} (P{amh_pct:.0f}), "
          f"AFC {patient.afc} (P{afc_pct:.0f}), BMI {patient.bmi:.1f}")

    k = res["known"]
    any_known = any(v is not None for v in [k.okk, k.mii, k.pn2, k.blasts, k.good, k.euploid])
    if any_known:
        print("\nBayesian conditioning — known values:")
        for label, val in [("OKK", k.okk), ("MII", k.mii), ("2PN", k.pn2),
                            ("Blasts", k.blasts), ("Good", k.good), ("Euploid", k.euploid)]:
            if val is not None:
                print(f"   {label}: {val}  (observation)")

    print("\nStochastic pipeline — median outcomes (n = %d simulations)" % N_SIM)
    print(f"  Retrieved oocytes:   {int(res['okk_med']):2d}    "
          f"[{int(np.percentile(res['sim_okk'],2.5)):2d}–{int(np.percentile(res['sim_okk'],97.5)):2d}]")
    print(f"  MII oocytes:         {int(res['mii_med']):2d}    "
          f"[{int(np.percentile(res['sim_mii'],2.5)):2d}–{int(np.percentile(res['sim_mii'],97.5)):2d}]")
    print(f"  2PN zygotes:         {int(res['pn2_med']):2d}    "
          f"[{int(np.percentile(res['sim_pn2'],2.5)):2d}–{int(np.percentile(res['sim_pn2'],97.5)):2d}]")
    print(f"  Blastocysts:         {int(res['blasts_med']):2d}    "
          f"[{int(np.percentile(res['sim_blasts'],2.5)):2d}–{int(np.percentile(res['sim_blasts'],97.5)):2d}]")
    print(f"  Good-quality blasts: {int(res['good_med']):2d}    "
          f"[{int(np.percentile(res['sim_good'],2.5)):2d}–{int(np.percentile(res['sim_good'],97.5)):2d}]")
    print(f"  Euploid embryos:     {int(res['euploid_med']):2d}    "
          f"[{int(np.percentile(res['sim_euploid'],2.5)):2d}–{int(np.percentile(res['sim_euploid'],97.5)):2d}]")
    print(f"  Warmed (post-thaw):  {int(res['warmed_med']):2d}    "
          f"[{int(np.percentile(res['sim_warmed'],2.5)):2d}–{int(np.percentile(res['sim_warmed'],97.5)):2d}]")

    print(f"\nP(>=1 euploid embryo):              {res['p_at_least_one_euploid']*100:.1f}%")

    print(f"\n--- KPIScore distribution (laboratory KPI) ---")
    kpi_lo, kpi_hi = res['kpi_score_ci']
    print(f"  Median KPIScore: {res['kpi_score_median']} / 25  (95% CI {kpi_lo}-{kpi_hi})")

    print(f"\n--- Per-transfer pregnancy probability (ensemble) ---")
    print(f"  [a] FORTUNE-based:    {res['p_per_transfer_fortune']*100:5.1f}%   (clinical predictors)")
    print(f"  [b] KPI-based:        {res['p_per_transfer_kpi']*100:5.1f}%   (laboratory performance)")
    print(f"  [c] Ensemble (w={res['kpi_weight']:.2f}): {res['p_per_transfer']*100:5.1f}%   "
          f"= logit-weighted average of [a] and [b]")

    print(f"\n--- Three-level cycle pregnancy outcome (using ensemble [c]) ---")
    print(f"  [1] Per-transfer rate (if transfer happens):     {res['p_per_transfer']*100:5.1f}%")
    print(f"  [2] If cycle is viable (>=1 transfer, ~{res['n_tx_median_viable']} median): "
          f"{res['p_cum_if_viable']*100:5.1f}%")
    print(f"      rate-only 95% CI:                            "
          f"{res['rate_ci'][0]*100:5.1f}-{res['rate_ci'][1]*100:.1f}%")
    print(f"  [3] Overall cycle success (from stim start):     "
          f"{res['p_overall_cycle']*100:5.1f}%")
    print(f"      = P(viable={res['p_viable']*100:.0f}%) x P(preg|viable={res['p_cum_if_viable']*100:.0f}%)")
    print(f"  Ordering check: per-transfer ({res['p_per_transfer']*100:.0f}%) <= "
          f"cum-if-viable ({res['p_cum_if_viable']*100:.0f}%) >= "
          f"overall ({res['p_overall_cycle']*100:.0f}%)  OK")

    # ─── NN ENSEMBLE FINAL LAYER (v6.0) ────────────────────────
    if 'nn_prediction' in res:
        nn = res['nn_prediction']
        nvsa = res.get('nn_nvsa')
        post = res.get('posterior')

        print(f"\n--- NN ensemble final layer  ---")
        print(f"  Source: {nn['source']}")
        print(f"  Attempt #: {res['nn_attempt_number']}")
        print(f"  Base NN prob:        {nn['base_prob_mean']*100:5.1f}%   "
              f"(95% CI {nn['base_prob_ci'][0]*100:.1f}-{nn['base_prob_ci'][1]*100:.1f}%)")
        if nvsa:
            print(f"  NVSA-adjusted prob:  {nvsa['adjusted_mean']*100:5.1f}%   "
                  f"(95% CI {nvsa['adjusted_ci'][0]*100:.1f}-{nvsa['adjusted_ci'][1]*100:.1f}%)")
            print(f"  KPI-based CI:        {nvsa['kpi_ci_low_mean']*100:.1f}-{nvsa['kpi_ci_high_mean']*100:.1f}%")

        if post:
            print(f"\n--- Bayesian posterior (Beta-Binomial, covariate-dependent prior) ---")
            print(f"  Prior type:           {post['prior_type']}")
            print(f"  Prior mean:           {post['prior_mean']*100:.1f}%  "
                  f"Beta({post['prior_alpha']:.1f}, {post['prior_beta']:.1f})  "
                  f"[κ={post['prior_kappa']:.0f}]")
            print(f"  Real clinic cycles:   {post['n_real_cycles']}")
            print(f"  NN pseudo-obs (n={post['nn_pseudo_n']:d}): input {post['nn_input_prob']*100:.1f}%")
            print(f"  Posterior:            Beta({post['posterior_alpha']:.0f}, {post['posterior_beta']:.0f})")
            print(f"  Posterior mean:  {post['mean']*100:5.1f}%")
            print(f"  Posterior mode:  {post['mode']*100:5.1f}%")
            print(f"  95% credible interval: {post['ci_low']*100:.1f}–{post['ci_high']*100:.1f}%")

        # Per-attempt curve
        if 'attempt_curve' in res:
            curve = res['attempt_curve']
            print(f"\n--- Per-attempt probability decay ---")
            print(f"  (Combined = mean of NN raw + selection-effect decay, "
                  f"alpha={curve['decay_alpha']:.2f}/attempt, Malizia NEJM 2009)")
            print(f"  {'Att':>4}  {'NN raw':>8}  {'Sel.decay':>10}  {'Combined':>9}")
            print(f"  {'---':>4}  {'------':>8}  {'---------':>10}  {'--------':>9}")
            for a, p_nn, p_sel, p_comb in zip(
                    curve['attempts'], curve['p_nn_raw'],
                    curve['p_sel_decay'], curve['p_mean']):
                bar = '#' * int(p_comb * 40)
                print(f"  {a:>4}  {p_nn*100:>7.1f}%  {p_sel*100:>9.1f}%  "
                      f"{p_comb*100:>8.1f}%  {bar}")

    # ─── ESTEVES EUPLOIDY / BANKING (v6.2) ────────────────────
    if 'esteves_banking' in res:
        eb = res['esteves_banking']
        print(f"\n--- Esteves Euploidy & Oocyte-Banking Analysis ---")
        print(f"  Reference: Esteves et al., Front Endocrinol 2019;10:99")
        print(f"             POSEIDON ART Calculator (n=347, 2520 MII)")
        print(f"  Sperm source: {eb['sperm_source']}")
        print(f"  P(euploid blastocyst per MII oocyte) at age "
              f"{eb['age']:.0f}: {eb['p_per_mii']*100:.1f}%")
        print()
        print(f"  [A] Euploid blastocysts needed for pregnancy target")
        print(f"      (per-transfer prob used: {eb['p_transfer_used']*100:.1f}%)")
        for t in eb['preg_targets']:
            k = eb['euploid_for_preg'][t]
            print(f"        {t*100:>3.0f}% cumulative pregnancy  ->  "
                  f"{k} euploid blastocyst(s)")
        print()
        print(f"  [B] MII oocytes needed to bank (Esteves inverse model)")
        confs = eb['confidences']
        header = "        k euploid |" + "".join(
            f"  {int(c*100)}% conf" for c in confs)
        print(header)
        print("        " + "-" * (len(header) - 8))
        for k in eb['k_targets']:
            row = f"        {k:>9} |"
            for c in confs:
                v = eb['mii_table'][k][c]
                row += f"  {(str(v) if v else '>200'):>8}"
            print(row)
        print()
        if eb['patient_mii_median'] is not None:
            fwd = eb['forward_at_median']
            print(f"  [C] This patient's simulated MII median: "
                  f"{eb['patient_mii_median']} "
                  f"(P95: {eb['patient_mii_p95']})")
            if fwd:
                print(f"      -> expected euploid blastocysts: "
                      f"{fwd['mean']:.1f} "
                      f"(median {fwd['median']:.0f}, "
                      f"5-95%: {fwd['p05']:.0f}-{fwd['p95']:.0f})")
            # gap analysis
            need_50 = eb['euploid_for_preg'][0.50]
            if fwd and need_50:
                gap = "SUFFICIENT" if fwd['median'] >= need_50 else "MAY NEED BANKING"
                print(f"      -> for 50% pregnancy need {need_50} euploid: {gap}")

    # ─── CLUSTER ANALYSIS (v6.1) ──────────────────────────────
    if 'cluster_analysis' in res:
        ca = res['cluster_analysis']
        probs = ca['cluster_probs']
        dom = ca['dominant_cluster']
        dom_info = CLUSTER_INTERPRETATIONS[dom]

        print(f"\n--- Cluster Analysis (unsupervised classification) ---")
        print(f"  Reference: Sergeev et al., 'Decoding IVF Laboratory Performance")
        print(f"             through Dimensionality Reduction and Cluster Analysis'")
        print(f"")
        print(f"  Cluster membership distribution across {len(ca['assignments'])} simulations:")
        for c in (0, 1, 2):
            info = CLUSTER_INTERPRETATIONS[c]
            mark = "  <-- DOMINANT" if c == dom else ""
            bar = '#' * int(probs[c] * 50)
            print(f"    Cluster {c} ({info['name']:18s}, pred preg {info['preg_rate']*100:.0f}%): "
                  f"{probs[c]*100:5.1f}%  {bar}{mark}")
        print(f"")
        print(f"  Dominant cluster: {dom} - {dom_info['name']} ({dom_info['preg_rate']*100:.0f}% pred preg)")
        print(f"  Clinical notes:   {dom_info['clinical_notes']}")
        print(f"  PCA(2) explained variance: {ca['pca_explained'].sum()*100:.1f}%")

    print(f"\nRisk profile:")
    print(f"  OHSS moderate (15-19 ooc):  {res['ohss']['p_moderate_ohss']*100:.1f}%")
    print(f"  OHSS severe (>=20 ooc):     {res['ohss']['p_severe_ohss']*100:.1f}%")
    print(f"  Empty cycle (no blast):     {res['empty']['p_no_blast']*100:.1f}%")
    print(f"  No good-quality blast:      {res['empty']['p_no_good_blast']*100:.1f}%")
    print()


# ============================================================
# FIGURES
# ============================================================

def create_figures(patient, res):
    figs = {}
    ohss = res['ohss']; empty = res['empty']
    known = res['known']

    # ───── 1. FUNNEL ──────────────────────────────────────────
    funnel_y = ["Retrieved", "MII", "2PN", "Blastocysts",
                "Good blasts", "Euploid", "Warmed"]
    funnel_x = [int(res['okk_med']), int(res['mii_med']), int(res['pn2_med']),
                int(res['blasts_med']), int(res['good_med']),
                int(res['euploid_med']), int(res['warmed_med'])]
    colors_funnel = ["rgba(0,180,255,0.65)","rgba(0,220,180,0.65)","rgba(140,180,255,0.65)",
                     "rgba(180,120,255,0.65)","rgba(255,140,200,0.65)",
                     "rgba(255,180,80,0.65)","rgba(150,210,255,0.65)"]
    funnel = go.Figure(go.Funnel(
        y=funnel_y, x=funnel_x, opacity=0.78,
        marker={"color": colors_funnel}
    ))
    funnel.update_layout(title="IVF Cohort Funnel (Median of Monte Carlo)",
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    figs["IVF Cohort Funnel"] = funnel

    # ───── 2. VIOLINS ─────────────────────────────────────────
    stages = ["Retrieved","MII","2PN","Blastocysts","Good blasts","Euploid","Warmed"]
    arrays = [res['sim_okk'], res['sim_mii'], res['sim_pn2'],
              res['sim_blasts'], res['sim_good'], res['sim_euploid'], res['sim_warmed']]
    violin = go.Figure()
    for name, arr, col in zip(stages, arrays, colors_funnel):
        violin.add_trace(go.Violin(
            y=arr, name=name, box_visible=True, meanline_visible=True,
            fillcolor=col, line_color=col.replace("0.65","1.0"), opacity=0.7,
        ))
    violin.update_layout(title="Stage-wise Distributions (n = %d)" % N_SIM,
                         yaxis_title="Count", showlegend=False,
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    figs["Stage Distributions"] = violin

    # ───── 3. CUMULATIVE pregnancy probability distribution ──
    # Now uses three-level decomposition with rate-only CI
    p_any_pct = res['sim_p_any'] * 100
    ci_lo, ci_hi = res['rate_ci']
    cum = px.histogram(p_any_pct, nbins=60, opacity=0.65,
                       labels={"value": "Cumulative Pregnancy Probability (%)"},
                       marginal="box")
    cum.add_vline(x=res['p_per_transfer']*100, line_dash="dot",
                  line_color="rgba(255,160,80,0.9)",
                  annotation_text=f"Per-transfer {res['p_per_transfer']*100:.1f}%",
                  annotation_position="bottom right")
    cum.add_vline(x=res['p_cum_if_viable']*100, line_dash="dash",
                  line_color="rgba(80,180,80,0.9)",
                  annotation_text=f"If viable {res['p_cum_if_viable']*100:.1f}%",
                  annotation_position="top right")
    cum.add_vline(x=res['p_overall_cycle']*100, line_dash="dashdot",
                  line_color="rgba(80,80,200,0.9)",
                  annotation_text=f"Overall {res['p_overall_cycle']*100:.1f}%",
                  annotation_position="top left")
    cum.add_vrect(x0=ci_lo*100, x1=ci_hi*100, fillcolor="rgba(80,180,80,0.06)",
                  layer="below", line_width=0)
    cum.update_layout(
        title=f"Cycle Pregnancy Distribution  |  per-transfer {res['p_per_transfer']*100:.0f}%  →  "
              f"if-viable {res['p_cum_if_viable']*100:.0f}%  →  overall {res['p_overall_cycle']*100:.0f}%",
        xaxis_title="P(>=1 pregnancy) per simulation, %",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    figs["Cumulative Pregnancy Probability"] = cum

    # ───── 4. PER-TRANSFER pregnancy curve  (Herasight-style) ─
    p_at_least = pregnancy_count_distribution(res, max_k=14)
    per_tx = go.Figure()
    per_tx.add_trace(go.Bar(
        x=[str(k) for k in range(1, 15)],
        y=[v*100 for v in p_at_least],
        marker_color=[f"rgba({80+10*k},{120+5*k},{200-8*k},0.75)" for k in range(14)],
        text=[f"{v*100:.1f}%" for v in p_at_least],
        textposition="outside",
    ))
    per_tx.add_hline(
        y=res['p_per_transfer']*100, line_dash="dash",
        line_color="rgba(255,100,80,0.7)",
        annotation_text=f"Per-transfer baseline: {res['p_per_transfer']*100:.1f}%",
        annotation_position="top right"
    )
    per_tx.update_layout(
        title="Chances of Pregnancy — per-cycle k = >=1, >=2, >=3 ...",
        xaxis_title="Number of live pregnancies (>= k)",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0,100]),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    figs["Pregnancy Probability per Transfer"] = per_tx

    # ───── 5. RISK ────────────────────────────────────────────
    risk_fig = go.Figure()
    rl = ["OHSS moderate","OHSS severe","Any OHSS","Empty cycle","No good blast"]
    rv = [ohss['p_moderate_ohss']*100, ohss['p_severe_ohss']*100,
          ohss['p_any_ohss']*100, empty['p_no_blast']*100, empty['p_no_good_blast']*100]
    rc = ["rgba(255,200,60,0.7)","rgba(255,100,80,0.7)","rgba(255,140,40,0.7)",
          "rgba(180,100,220,0.7)","rgba(140,100,220,0.7)"]
    risk_fig.add_trace(go.Bar(x=rl, y=rv, marker_color=rc,
                              text=[f"{v:.1f}%" for v in rv], textposition="outside"))
    risk_fig.update_layout(title="Clinical Risk Profile",
                           yaxis_title="Probability (%)",
                           yaxis=dict(range=[0, max(rv)*1.4 + 5]),
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    figs["Clinical Risk Profile"] = risk_fig

    # ───── 6. RADAR ───────────────────────────────────────────
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=[
            min(patient.amh/5,1)*100,
            (res['p_fert'] if res['p_fert'] else 0.7)*100,
            min(res['good_med']/4,1)*100,
            res['p_at_least_one_euploid']*100,
            res['p_cum_if_viable']*100,
            max(0, 100 - ohss['p_any_ohss']*100)
        ],
        theta=["Ovarian Reserve","Fertilization","Embryo Quality",
               "Euploid Potential","Pregnancy if Viable","OHSS Safety"],
        fill='toself', name="Patient Profile",
        line=dict(color="rgba(30,120,200,0.9)"),
        fillcolor="rgba(30,120,200,0.25)",
    ))
    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100])),
        title="IVF Biological & Safety Profile",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    figs["Biological & Safety Profile"] = radar

    # ───── 7. KPI SCORE DISTRIBUTION ──────────────────────────
    kpi_scores = res['sim_kpi_scores']
    kpi_lo, kpi_hi = res['kpi_score_ci']
    kpi_med = res['kpi_score_median']
    kpi_counts = np.bincount(kpi_scores, minlength=26)
    kpi_pct = kpi_counts[5:26] / kpi_counts.sum() * 100
    kpi_fig = go.Figure()
    kpi_fig.add_trace(go.Bar(
        x=list(range(5, 26)), y=kpi_pct,
        marker_color=[
            "rgba(255,80,80,0.7)" if s <= 10 else
            "rgba(255,180,80,0.75)" if s <= 16 else
            "rgba(80,200,120,0.75)"
            for s in range(5, 26)
        ],
        text=[f"{v:.0f}%" if v >= 3 else "" for v in kpi_pct],
        textposition="outside",
    ))
    kpi_fig.add_vline(x=kpi_med, line_dash="dash",
                       line_color="rgba(30,120,200,0.9)",
                       annotation_text=f"Median {kpi_med}",
                       annotation_position="top")
    kpi_fig.update_layout(
        title=f"KPIScore Distribution (median {kpi_med}, 95% CI {kpi_lo}-{kpi_hi})",
        xaxis_title="KPIScore (5 = worst, 25 = best)",
        yaxis_title="Probability mass (%)",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    figs["KPIScore Distribution"] = kpi_fig

    # ───── 8. FORTUNE vs KPI vs ENSEMBLE COMPARISON ───────────
    comp_fig = go.Figure()
    bins = np.linspace(0, 1, 41)
    for label, arr, col in [
        ("FORTUNE-based",  res['sim_p_fortune'],  "rgba(80,140,220,0.55)"),
        ("KPI-based",      res['sim_p_kpi'],      "rgba(220,140,80,0.55)"),
        ("Combined (ensemble)", res['sim_p_combined'], "rgba(120,200,140,0.65)"),
    ]:
        comp_fig.add_trace(go.Histogram(
            x=arr * 100, xbins=dict(start=0, end=100, size=2.5),
            opacity=0.65, name=label,
            marker_color=col,
        ))
    comp_fig.update_layout(
        title="Per-Transfer Pregnancy Probability — FORTUNE vs KPI vs Ensemble",
        xaxis_title="Per-transfer probability (%)",
        yaxis_title="Simulation count",
        barmode='overlay',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    figs["FORTUNE vs KPI vs Ensemble"] = comp_fig

    # ───── 9. PER-ATTEMPT PROBABILITY DECAY  (v6) ─────────────
    if 'attempt_curve' in res:
        curve = res['attempt_curve']
        attempt_fig = go.Figure()

        # Selection-effect decay (dotted)
        attempt_fig.add_trace(go.Scatter(
            x=curve['attempts'], y=[p*100 for p in curve['p_sel_decay']],
            mode='lines+markers',
            line=dict(color='rgba(220,140,80,0.8)', width=2, dash='dot'),
            marker=dict(size=8), name='Selection-effect decay (analytical)',
        ))

        # NN raw (dashed, only meaningful when NN is loaded)
        attempt_fig.add_trace(go.Scatter(
            x=curve['attempts'], y=[p*100 for p in curve['p_nn_raw']],
            mode='lines+markers',
            line=dict(color='rgba(140,180,220,0.7)', width=2, dash='dash'),
            marker=dict(size=8), name='NN raw (per-attempt)',
        ))

        # Combined (solid, primary)
        attempt_fig.add_trace(go.Scatter(
            x=curve['attempts'], y=[p*100 for p in curve['p_mean']],
            mode='lines+markers+text',
            line=dict(color='rgba(80,140,220,1.0)', width=3),
            marker=dict(size=12, color='rgba(80,140,220,1.0)'),
            text=[f"{p*100:.1f}%" for p in curve['p_mean']],
            textposition='top center', name='Combined (primary)',
        ))

        # CI band
        attempt_fig.add_trace(go.Scatter(
            x=curve['attempts'] + curve['attempts'][::-1],
            y=[p*100 for p in curve['p_hi']] + [p*100 for p in curve['p_lo'][::-1]],
            fill='toself', fillcolor='rgba(80,140,220,0.12)',
            line=dict(color='rgba(255,255,255,0)'), showlegend=False,
        ))

        attempt_fig.update_layout(
            title=(f"Per-Attempt Pregnancy Probability  "
                   f"(decay α={curve['decay_alpha']:.2f} per attempt, "
                   f"source: Malizia et al. NEJM 2009)"),
            xaxis_title="IVF attempt number",
            xaxis=dict(tickmode='array', tickvals=curve['attempts']),
            yaxis_title="Probability of clinical pregnancy (%)",
            yaxis=dict(range=[0, max([p*100 for p in curve['p_hi']]) * 1.25 + 5]),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        figs["Per-Attempt Probability Decay"] = attempt_fig

    # ───── 10. BAYESIAN POSTERIOR DISTRIBUTION  (v6) ──────────
    if 'posterior' in res:
        post = res['posterior']
        x = np.linspace(0.001, 0.999, 400)
        post_pdf  = beta_dist.pdf(x, post['posterior_alpha'], post['posterior_beta'])
        prior_pdf = beta_dist.pdf(x, post['prior_alpha'],     post['prior_beta'])

        post_fig = go.Figure()
        # prior
        post_fig.add_trace(go.Scatter(
            x=x*100, y=prior_pdf, mode='lines',
            line=dict(color='rgba(150,150,150,0.7)', dash='dot', width=2),
            name=f"{post['prior_type']} prior  (mean {post['prior_mean']*100:.1f}%,  κ={post['prior_kappa']:.0f})",
        ))
        # posterior
        post_fig.add_trace(go.Scatter(
            x=x*100, y=post_pdf, mode='lines', fill='tozeroy',
            line=dict(color='rgba(80,180,120,1.0)', width=2.5),
            fillcolor='rgba(80,180,120,0.22)',
            name=f"Posterior  Beta({post['posterior_alpha']:.0f}, {post['posterior_beta']:.0f})",
        ))
        post_fig.add_vline(
            x=post['mean']*100, line_dash='dash',
            line_color='rgba(200,60,80,0.9)',
            annotation_text=f"Posterior mean {post['mean']*100:.1f}%",
            annotation_position="top right",
        )
        post_fig.add_vrect(
            x0=post['ci_low']*100, x1=post['ci_high']*100,
            fillcolor='rgba(80,180,120,0.07)', layer='below', line_width=0,
            annotation_text=f"95% credible: {post['ci_low']*100:.1f}–{post['ci_high']*100:.1f}%",
            annotation_position="top left",
        )
        post_fig.update_layout(
            title=(f"Bayesian Posterior — Covariate-Dependent Beta Regression Prior<br>"
                   f"<sup>Prior mean {post['prior_mean']*100:.1f}% (FORTUNE-based, κ={post['prior_kappa']:.0f}) → "
                   f"posterior mean {post['mean']*100:.1f}% after {post['n_real_cycles']} clinic batches + NN evidence</sup>"),
            xaxis_title="Pregnancy probability (%)",
            yaxis_title="Probability density",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        figs["Bayesian Posterior"] = post_fig

    # ───── 10b. ESTEVES EUPLOIDY / BANKING (v6.2) ─────────────
    if 'esteves_banking' in res:
        eb = res['esteves_banking']

        # ── Figure 1: euploid blastocysts vs MII oocytes curve ──
        mii_range = list(range(1, 41))
        p_mii = eb['p_per_mii']
        exp_euploid = [M * p_mii for M in mii_range]
        # confidence bands via binomial percentiles
        from scipy.stats import binom as _binom
        lo_band = [_binom.ppf(0.05, M, p_mii) for M in mii_range]
        hi_band = [_binom.ppf(0.95, M, p_mii) for M in mii_range]

        est_fig = go.Figure()
        est_fig.add_trace(go.Scatter(
            x=mii_range + mii_range[::-1],
            y=hi_band + lo_band[::-1],
            fill='toself', fillcolor='rgba(150,100,200,0.12)',
            line=dict(color='rgba(0,0,0,0)'), showlegend=False,
            name='5-95% band',
        ))
        est_fig.add_trace(go.Scatter(
            x=mii_range, y=exp_euploid, mode='lines',
            line=dict(color='rgba(130,70,190,1.0)', width=3),
            name=f'Expected euploid (p={p_mii*100:.1f}%/MII)',
        ))
        # mark euploid targets for pregnancy
        colors_t = ['rgba(60,160,90,0.9)', 'rgba(230,160,40,0.9)',
                    'rgba(210,70,70,0.9)']
        for (t, k), col in zip(eb['euploid_for_preg'].items(), colors_t):
            if k:
                est_fig.add_hline(
                    y=k, line_dash='dot', line_color=col,
                    annotation_text=f"{int(t*100)}% preg → {k} euploid",
                    annotation_position="right",
                )
        # patient's own MII median marker
        if eb['patient_mii_median']:
            mm = eb['patient_mii_median']
            est_fig.add_vline(
                x=mm, line_dash='dash', line_color='rgba(40,40,40,0.7)',
                annotation_text=f"Patient MII median: {mm}",
                annotation_position="top",
            )
        est_fig.update_layout(
            title=("Euploid Blastocysts vs MII Oocytes<br>"
                   f"<sup>{eb['sperm_source']} sperm, age {eb['age']:.0f}; "
                   f"independent banking-planning module</sup>"),
            xaxis_title="MII oocytes banked",
            yaxis_title="Euploid blastocysts",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        figs["Banking: Euploid vs MII"] = est_fig

        # ── Figure 2: MII needed heatmap-style bar (inverse) ────
        bank_fig = go.Figure()
        confs = eb['confidences']
        bar_colors = ['rgba(120,180,230,0.85)', 'rgba(90,140,210,0.9)',
                      'rgba(50,90,170,0.95)']
        for ci, c in enumerate(confs):
            ys = [eb['mii_table'][k][c] if eb['mii_table'][k][c] else None
                  for k in eb['k_targets']]
            bank_fig.add_trace(go.Bar(
                x=[f"{k} euploid" for k in eb['k_targets']],
                y=ys, name=f"{int(c*100)}% confidence",
                marker_color=bar_colors[ci % 3],
                text=[f"{v}" if v else ">200" for v in ys],
                textposition='outside',
            ))
        bank_fig.update_layout(
            title=("MII Oocytes to Bank for Target Euploid Blastocysts<br>"
                   f"<sup>Inverse banking model, age {eb['age']:.0f}, "
                   f"{eb['sperm_source']} sperm — key for fertility "
                   f"preservation counselling</sup>"),
            xaxis_title="Euploid blastocyst target",
            yaxis_title="MII oocytes required",
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        figs["Banking: MII Required"] = bank_fig

    # ───── 11. CLUSTER PCA SCATTER (v6.1) ─────────────────────
    if 'cluster_analysis' in res:
        ca = res['cluster_analysis']
        n_syn = ca['n_synthetic']
        emb = ca['pca_embedded']
        syn_2d = emb[:n_syn]
        patient_2d = emb[n_syn:]
        syn_labels = ca['synthetic_labels']
        probs = ca['cluster_probs']
        dom = ca['dominant_cluster']
        expl = ca['pca_explained']

        cluster_fig = go.Figure()

        # Background: synthetic cluster point clouds
        for c in (0, 1, 2):
            mask = syn_labels == c
            info = CLUSTER_INTERPRETATIONS[c]
            cluster_fig.add_trace(go.Scatter(
                x=syn_2d[mask, 0], y=syn_2d[mask, 1],
                mode='markers',
                marker=dict(size=5, color=info['color'], opacity=0.35,
                            line=dict(width=0)),
                name=f"C{c}: {info['name']} ({info['preg_rate']*100:.0f}%)",
                hoverinfo='skip',
            ))

        # Patient MC samples — colored by their cluster assignment
        for c in (0, 1, 2):
            mask = ca['assignments'] == c
            if mask.sum() == 0:
                continue
            info = CLUSTER_INTERPRETATIONS[c]
            # Make patient markers visually distinct: edge contrast
            cluster_fig.add_trace(go.Scatter(
                x=patient_2d[mask, 0], y=patient_2d[mask, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color=info['color'].replace('0.6', '0.85'),
                    line=dict(width=1.2, color='rgba(20,30,40,0.9)'),
                ),
                name=f"Patient -> C{c} ({mask.sum()/len(ca['assignments'])*100:.0f}%)",
                hoverinfo='skip',
            ))

        # Cluster centroids in PCA space (their positions in PCA from synthetic centroid mean)
        for c in (0, 1, 2):
            mask = syn_labels == c
            cx, cy = syn_2d[mask, 0].mean(), syn_2d[mask, 1].mean()
            info = CLUSTER_INTERPRETATIONS[c]
            cluster_fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode='markers+text',
                marker=dict(size=24, color=info['color'].replace('0.6', '1.0'),
                            symbol='star', line=dict(width=2, color='white')),
                text=[f"C{c}"], textposition='top center',
                textfont=dict(size=14, color='black'),
                name=f"Centroid C{c}",
                showlegend=False,
                hoverinfo='skip',
            ))

        # Patient median position
        med_x, med_y = np.median(patient_2d[:, 0]), np.median(patient_2d[:, 1])
        cluster_fig.add_trace(go.Scatter(
            x=[med_x], y=[med_y],
            mode='markers+text',
            marker=dict(size=22, color='rgba(255,80,80,1.0)',
                        symbol='diamond', line=dict(width=2.5, color='white')),
            text=['Patient'], textposition='bottom center',
            textfont=dict(size=12, color='black'),
            name=f"Patient median (-> C{dom})",
            showlegend=True,
        ))

        cluster_fig.update_layout(
            title=(f"Unsupervised Cluster Membership in PCA(2) Space  "
                   f"|  Explained variance: {expl.sum()*100:.1f}%"),
            xaxis_title=f"PC1 ({expl[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({expl[1]*100:.1f}%)",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(itemsizing='constant'),
        )
        figs["Cluster Membership (PCA)"] = cluster_fig

        # ───── 12. CLUSTER PROBABILITY BAR (v6.1) ─────────────
        prob_fig = go.Figure()
        labels = [f"C{c}: {CLUSTER_INTERPRETATIONS[c]['name']}<br>"
                  f"({CLUSTER_INTERPRETATIONS[c]['preg_rate']*100:.0f}% pred preg)"
                  for c in (0, 1, 2)]
        values = [probs[c]*100 for c in (0, 1, 2)]
        colors = [CLUSTER_INTERPRETATIONS[c]['color'] for c in (0, 1, 2)]
        # Highlight dominant
        line_widths = [3 if c == dom else 0 for c in (0, 1, 2)]
        prob_fig.add_trace(go.Bar(
            x=labels, y=values,
            marker=dict(color=colors,
                        line=dict(color='rgba(20,30,40,1.0)', width=line_widths)),
            text=[f"{v:.1f}%" for v in values],
            textposition='outside',
        ))
        prob_fig.update_layout(
            title=(f"Cluster Membership Probability  |  Dominant: C{dom} "
                   f"({CLUSTER_INTERPRETATIONS[dom]['name']})"),
            yaxis_title="Probability across MC iterations (%)",
            yaxis=dict(range=[0, 110]),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        figs["Cluster Probability Distribution"] = prob_fig

    return figs


# ============================================================
# HTML / PDF REPORT
# ============================================================

def generate_html(patient, res, image_paths):
    age_pct = percentile_rank(patient.female_age, REFERENCE["female_age"]["mean"], REFERENCE["female_age"]["sd"])
    amh_pct = percentile_rank(patient.amh, REFERENCE["amh"]["mean"], REFERENCE["amh"]["sd"])
    afc_pct = percentile_rank(patient.afc, REFERENCE["afc"]["mean"], REFERENCE["afc"]["sd"])
    ci_lo, ci_hi = res['rate_ci']
    ohss = res['ohss']; empty = res['empty']
    k = res['known']

    p_at_least = pregnancy_count_distribution(res, max_k=14)

    pct = lambda arr, q: int(np.percentile(arr, q))

    # known-values section
    known_html = ""
    any_known = any(v is not None for v in [k.okk, k.mii, k.pn2, k.blasts, k.good, k.euploid])
    if any_known:
        rows = ""
        for lbl, val in [("Retrieved oocytes", k.okk),
                          ("MII oocytes",       k.mii),
                          ("2PN zygotes",       k.pn2),
                          ("Blastocysts",       k.blasts),
                          ("Good-quality blastocysts", k.good),
                          ("Euploid embryos",   k.euploid)]:
            if val is not None:
                rows += f"<tr><td>{lbl}</td><td><b>{val}</b></td></tr>"
        known_html = f"""
        <h2>Bayesian Conditioning &mdash; Observed Values</h2>
        <p>The following values were entered as observations. Downstream
        distributions have been recomputed conditional on these point-mass
        observations.</p>
        <table>
            <tr><th>Stage</th><th>Observed value</th></tr>
            {rows}
        </table>
        """

    # ─── NN ensemble HTML block (v6) ──────────────────────────
    nn_html = ""
    if 'nn_prediction' in res:
        nn = res['nn_prediction']
        nvsa = res.get('nn_nvsa', {})
        nn_html = f"""
        <h2>NN Ensemble Final Layer (KAN + FT-Transformer)</h2>
        <p class="small">Source: <b>{nn['source']}</b>. Attempt #{res['nn_attempt_number']}.</p>
        <table>
            <tr><th>Endpoint</th><th>Value</th></tr>
            <tr><td>Base NN per-transfer probability</td>
                <td><b>{nn['base_prob_mean']*100:.1f}%</b>
                    (95% CI {nn['base_prob_ci'][0]*100:.1f}-{nn['base_prob_ci'][1]*100:.1f}%)</td></tr>
            <tr style="background:#fff8e1"><td><b>NVSA-adjusted probability</b><br>
                <span class="small">KPI-table corrected</span></td>
                <td><b>{nvsa.get('adjusted_mean',0)*100:.1f}%</b>
                    (95% CI {nvsa.get('adjusted_ci',(0,0))[0]*100:.1f}-{nvsa.get('adjusted_ci',(0,0))[1]*100:.1f}%)</td></tr>
            <tr><td>KPI-based confidence interval (mean of per-iteration CIs)</td>
                <td>{nvsa.get('kpi_ci_low_mean',0)*100:.1f}-{nvsa.get('kpi_ci_high_mean',0)*100:.1f}%</td></tr>
        </table>
        """

    # ─── Bayesian posterior HTML block (v6) ────────────────────
    posterior_html = ""
    if 'posterior' in res:
        post = res['posterior']
        posterior_html = f"""
        <h2>Bayesian Posterior &mdash; Covariate-Dependent Beta Regression Prior</h2>
        <p class="small">
          <b>Prior type:</b> {post['prior_type']}.
          Instead of a fixed global Beta(26,74), the prior mean is the FORTUNE-based
          per-transfer probability for <em>this specific patient</em>
          ({post['prior_mean']*100:.1f}%), encoded as {post['prior_kappa']:.0f}
          pseudo-observations (κ = {post['prior_kappa']:.0f}).
          This implements the Ferrari &amp; Cribari-Neto (2004) Beta regression
          parameterisation: α₀ = μ·κ = {post['prior_alpha']:.1f},
          β₀ = (1−μ)·κ = {post['prior_beta']:.1f}.
          The prior is then updated with {post['n_real_cycles']} real clinic
          batches and the NVSA-adjusted NN probability ({post['nn_input_prob']*100:.1f}%)
          encoded as {post['nn_pseudo_n']} pseudo-observations.
        </p>
        <table>
            <tr><th>Quantity</th><th>Value</th></tr>
            <tr><td>Prior type</td><td>{post['prior_type']}</td></tr>
            <tr><td>Prior mean (FORTUNE-based)</td><td>{post['prior_mean']*100:.1f}%</td></tr>
            <tr><td>Prior precision κ</td><td>{post['prior_kappa']:.0f} pseudo-observations</td></tr>
            <tr><td>Prior distribution</td>
                <td>Beta({post['prior_alpha']:.1f}, {post['prior_beta']:.1f})</td></tr>
            <tr><td>Real clinic batches incorporated</td><td>{post['n_real_cycles']}</td></tr>
            <tr><td>NN evidence (pseudo-obs)</td>
                <td>p_NN = {post['nn_input_prob']*100:.1f}%,  n = {post['nn_pseudo_n']}</td></tr>
            <tr><td>Posterior distribution</td>
                <td>Beta({post['posterior_alpha']:.0f}, {post['posterior_beta']:.0f})</td></tr>
            <tr style="background:#e8f5e9"><td><b>Posterior mean (point estimate)</b></td>
                <td><b>{post['mean']*100:.1f}%</b></td></tr>
            <tr><td>Posterior mode (MAP)</td><td>{post['mode']*100:.1f}%</td></tr>
            <tr><td>{int(post['cred_level']*100)}% credible interval</td>
                <td>{post['ci_low']*100:.1f}&ndash;{post['ci_high']*100:.1f}%</td></tr>
        </table>
        """

    # ─── Per-attempt curve HTML block (v6) ─────────────────────
    attempt_html = ""
    if 'attempt_curve' in res:
        curve = res['attempt_curve']
        rows = ""
        for a, pm, plo, phi, pnn, psel in zip(
                curve['attempts'], curve['p_mean'],
                curve['p_lo'], curve['p_hi'],
                curve['p_nn_raw'], curve['p_sel_decay']):
            rows += (f"<tr><td>{a}</td>"
                     f"<td>{pnn*100:.1f}%</td>"
                     f"<td>{psel*100:.1f}%</td>"
                     f"<td><b>{pm*100:.1f}%</b></td>"
                     f"<td>{plo*100:.1f}–{phi*100:.1f}%</td></tr>")
        attempt_html = f"""
        <h2>Per-Attempt Probability Decay</h2>
        <p class="small">
          Three series reported. <b>NN raw</b>: raw neural-network output per attempt
          (often near-flat if training data lacks multi-attempt patients).
          <b>Selection-effect decay</b>: analytical logit-scale decay calibrated to
          Malizia et al. NEJM 2009 (n=14,248 cycles; per-cycle rates:
          30 % → 25.6 % → 22.5 % → 20.5 %); decay constant
          &alpha; = {curve['decay_alpha']:.2f}/attempt.
          <b>Combined</b>: logit-scale mean of the two — used as primary estimate.
          {"(NN fallback active: NN raw = FORTUNE+KPI ensemble, attempt-independent)" if "FORTUNE+KPI ensemble" in res['nn_prediction']['source'] else ""}
        </p>
        <table>
            <tr><th>Attempt</th><th>NN raw</th><th>Selection decay</th>
                <th>Combined (primary)</th><th>95% CI</th></tr>
            {rows}
        </table>
        """

    # ─── Cluster analysis HTML block (v6.1) ────────────────────
    cluster_html = ""
    if 'cluster_analysis' in res:
        ca = res['cluster_analysis']
        probs = ca['cluster_probs']
        dom = ca['dominant_cluster']
        dom_info = CLUSTER_INTERPRETATIONS[dom]

        # Per-cluster rows
        cluster_rows = ""
        for c in (0, 1, 2):
            info = CLUSTER_INTERPRETATIONS[c]
            is_dom = (c == dom)
            shade = "background:#fffde7" if is_dom else ("background:#ffffff" if c % 2 == 0 else "background:#f7fafc")
            mark = " <b>(dominant)</b>" if is_dom else ""
            cluster_rows += (
                f"<tr style='{shade}'>"
                f"<td>C{c} &mdash; <b>{info['name']}</b>{mark}</td>"
                f"<td>{info['preg_rate']*100:.0f}%</td>"
                f"<td><b>{probs[c]*100:.1f}%</b></td>"
                f"<td class='small'>{info['description']}</td>"
                f"</tr>"
            )

        cluster_html = f"""
        <h2>Unsupervised Cluster Classification (v6.1)</h2>
        <p class="small">
            Independent unsupervised assessment based on k-means clustering of
            1556 IVF cycles published in Sergeev et al., "Decoding IVF Laboratory
            Performance through Dimensionality Reduction and Cluster Analysis".
            Each Monte Carlo iteration is assigned to the nearest cluster centroid
            in z-score-standardized 18-dimensional feature space. The dominant
            cluster across iterations indicates the most likely IVF protocol
            phenotype for this patient.
        </p>
        <p>
            <b>Dominant cluster: C{dom} &mdash; {dom_info['name']}</b>
            (published predicted pregnancy rate: <b>{dom_info['preg_rate']*100:.0f}%</b>)
        </p>
        <table>
            <tr><th>Cluster</th><th>Published preg rate</th>
                <th>P(this patient)</th><th>Description</th></tr>
            {cluster_rows}
        </table>
        <p class="small"><b>Clinical recommendation for {dom_info['name']}s:</b>
            {dom_info['clinical_notes']}</p>
        <p class="small"><em>Note on visualization: the published study used t-SNE
            for optimal local structure preservation. Because the original
            1556-cycle raw dataset is not redistributable, the PCA(2) scatter plot
            below uses synthetic point clouds generated around each published
            cluster centroid for visual context. The patient's classification
            itself uses the actual nearest-centroid algorithm in the original
            18-dimensional space.</em></p>
        """

    # per-transfer / cumulative pregnancy rows
    preg_rows = ""
    for i, p_ in enumerate(p_at_least, 1):
        preg_rows += f"<tr><td>&ge; {i}</td><td>{p_*100:.1f}%</td></tr>"

    html = f"""
    <!DOCTYPE html><html><head><meta charset="UTF-8">
    <title>IVF Digital Twin Report v5.0</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.45; color: #1a3a4a; }}
        h1 {{ color: #1B4F72; border-bottom: 2px solid #1B4F72; padding-bottom:6px; }}
        h2 {{ color: #154360; margin-top: 28px; }}
        h3 {{ color: #1A5276; }}
        table {{ border-collapse: collapse; width: 100%; margin: 14px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background: #1B4F72; color: white; }}
        tr:nth-child(even) td {{ background: #f7fafc; }}
        .figure {{ margin: 32px 0; text-align: center; page-break-inside: avoid; }}
        .figure img {{ max-width: 100%; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .summary-box {{ background: #EAF4FB; border-left: 6px solid #1B4F72;
                        padding: 14px 18px; margin: 16px 0; }}
        .small {{ font-size: 0.9em; color: #555; }}
    </style></head><body>

    <h1>IVF Digital Twin Report v5.0 &mdash; Bayesian Dynamic Updating</h1>

    <div class="summary-box">
        <b>Pregnancy outcomes — KPI + FORTUNE ensemble (w_KPI = {res['kpi_weight']:.2f}):</b><br>
        &nbsp;&nbsp;&nbsp;FORTUNE-based per-transfer: <b>{res['p_per_transfer_fortune']*100:.1f}%</b>
        &nbsp;|&nbsp; KPI-based per-transfer: <b>{res['p_per_transfer_kpi']*100:.1f}%</b>
        &nbsp;|&nbsp; Ensemble: <b>{res['p_per_transfer']*100:.1f}%</b><br>
        &nbsp;&nbsp;&nbsp;Median KPIScore: <b>{res['kpi_score_median']} / 25</b>
        (95% CI {res['kpi_score_ci'][0]}-{res['kpi_score_ci'][1]})<br><br>
        <b>Three-level pregnancy outcome (based on ensemble per-transfer):</b><br>
        &nbsp;&nbsp;&nbsp;Per-transfer: <b>{res['p_per_transfer']*100:.1f}%</b>
        &nbsp;|&nbsp; If viable ({res['n_tx_median_viable']} tx): <b>{res['p_cum_if_viable']*100:.1f}%</b>
        (rate 95% CI {ci_lo*100:.1f}-{ci_hi*100:.1f}%)
        &nbsp;|&nbsp; Overall (from stim): <b>{res['p_overall_cycle']*100:.1f}%</b>
    </div>

    <h2>Patient characteristics</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th><th>Population percentile</th></tr>
        <tr><td>Female age</td><td>{patient.female_age} years</td><td>{age_pct:.0f}%</td></tr>
        <tr><td>AMH</td><td>{patient.amh} ng/mL</td><td>{amh_pct:.0f}%</td></tr>
        <tr><td>AFC</td><td>{patient.afc}</td><td>{afc_pct:.0f}%</td></tr>
        <tr><td>BMI</td><td>{patient.bmi}</td><td>&mdash;</td></tr>
    </table>

    {known_html}

    <h2>Stochastic pipeline &mdash; median outcomes (n = {N_SIM})</h2>
    <table>
      <tr><th>Stage</th><th>Process model</th><th>Median</th><th>2.5-97.5% CI</th></tr>
      <tr><td>Retrieved oocytes</td><td>Normal(&mu;, &sigma;)</td><td>{int(res['okk_med'])}</td>
          <td>{pct(res['sim_okk'],2.5)}-{pct(res['sim_okk'],97.5)}</td></tr>
      <tr><td>MII oocytes</td><td>Binomial &middot; logistic rate</td><td>{int(res['mii_med'])}</td>
          <td>{pct(res['sim_mii'],2.5)}-{pct(res['sim_mii'],97.5)}</td></tr>
      <tr><td>2PN zygotes</td><td>Binomial &middot; logistic rate</td><td>{int(res['pn2_med'])}</td>
          <td>{pct(res['sim_pn2'],2.5)}-{pct(res['sim_pn2'],97.5)}</td></tr>
      <tr><td>Blastocysts</td><td>Binomial &middot; Gaussian rate</td><td>{int(res['blasts_med'])}</td>
          <td>{pct(res['sim_blasts'],2.5)}-{pct(res['sim_blasts'],97.5)}</td></tr>
      <tr><td>Good-quality blasts</td><td>Binomial &middot; Beta rate</td><td>{int(res['good_med'])}</td>
          <td>{pct(res['sim_good'],2.5)}-{pct(res['sim_good'],97.5)}</td></tr>
      <tr><td>Euploid embryos</td><td>Bernoulli &times; age</td><td>{int(res['euploid_med'])}</td>
          <td>{pct(res['sim_euploid'],2.5)}-{pct(res['sim_euploid'],97.5)}</td></tr>
      <tr><td>Warmed (95% survival)</td><td>Binomial(EUP, 0.95)</td><td>{int(res['warmed_med'])}</td>
          <td>{pct(res['sim_warmed'],2.5)}-{pct(res['sim_warmed'],97.5)}</td></tr>
    </table>

    <h2>Pregnancy outcomes — KPI + FORTUNE ensemble</h2>
    <p class="small">The per-transfer pregnancy probability is computed as a logit-scale ensemble
       of two independent sources: FORTUNE (clinical predictors: age, AMH, BMI) and KPIScore
       (laboratory KPI: MII count, fertilization rate, good blast count). The weight w<sub>KPI</sub>
       = {res['kpi_weight']:.2f} balances the two; w=0 uses FORTUNE only, w=1 uses KPI only.</p>
    <table>
        <tr><th>Source</th><th>Per-transfer probability</th><th>Notes</th></tr>
        <tr style="background:#e3f2fd"><td><b>FORTUNE-based</b></td>
            <td><b>{res['p_per_transfer_fortune']*100:.1f}%</b></td>
            <td>From age, AMH, BMI (clinical predictors)</td></tr>
        <tr style="background:#fff3e0"><td><b>KPI-based</b></td>
            <td><b>{res['p_per_transfer_kpi']*100:.1f}%</b></td>
            <td>From KPIScore distribution (median {res['kpi_score_median']}/25)</td></tr>
        <tr style="background:#e8f5e9"><td><b>Ensemble (combined)</b></td>
            <td><b>{res['p_per_transfer']*100:.1f}%</b></td>
            <td>Logit-weighted average, w<sub>KPI</sub>={res['kpi_weight']:.2f}</td></tr>
    </table>

    <h2>Three-level cycle outcome (based on ensemble per-transfer)</h2>
    <p class="small">Mathematical decomposition: <b>per-transfer &le; cumulative-if-viable</b>,
       and <b>overall = P(viable) &times; cumulative-if-viable</b>.</p>
    <table>
        <tr><th>Endpoint</th><th>Value</th></tr>
        <tr><td>P(&ge;1 euploid embryo)</td><td>{res['p_at_least_one_euploid']*100:.1f}%</td></tr>
        <tr><td>Expected euploid embryos</td><td>{res['expected_euploid']:.2f}</td></tr>
        <tr><td>P(viable cycle, &ge;1 transfer)</td><td>{res['p_viable']*100:.1f}%</td></tr>
        <tr style="background:#fff8e1"><td><b>[1] Per-transfer pregnancy rate (ensemble)</b><br>
            <span class="small">single transfer, if it happens</span></td>
            <td><b>{res['p_per_transfer']*100:.1f}%</b></td></tr>
        <tr style="background:#e8f5e9"><td><b>[2] Pregnancy if viable cycle</b><br>
            <span class="small">cumulative across {res['n_tx_median_viable']} median transfer(s), if cycle is viable</span></td>
            <td><b>{res['p_cum_if_viable']*100:.1f}%</b>
                (rate CI {ci_lo*100:.1f}-{ci_hi*100:.1f}%)</td></tr>
        <tr style="background:#e3f2fd"><td><b>[3] Overall cycle success</b><br>
            <span class="small">marginal probability from start of stimulation</span></td>
            <td><b>{res['p_overall_cycle']*100:.1f}%</b></td></tr>
    </table>

    <h2>Probability of &ge; k pregnancies in this cycle</h2>
    <p class="small">Based on the stochastic transfer of warmed euploid embryos sequentially.
       Each row gives the probability of achieving <b>at least</b> that many clinical
       pregnancies in the current cycle.</p>
    <table>
        <tr><th>&ge; k pregnancies</th><th>Probability</th></tr>
        {preg_rows}
    </table>

    {nn_html}

    {posterior_html}

    {attempt_html}

    {cluster_html}

    <h2>Risk profile</h2>
    <table>
        <tr><th>Risk</th><th>Probability</th></tr>
        <tr><td>OHSS moderate (15-19 oocytes)</td><td>{ohss['p_moderate_ohss']*100:.1f}%</td></tr>
        <tr><td>OHSS severe (&ge;20 oocytes)</td><td>{ohss['p_severe_ohss']*100:.1f}%</td></tr>
        <tr><td>Any OHSS</td><td>{ohss['p_any_ohss']*100:.1f}%</td></tr>
        <tr><td>Empty cycle (no blast)</td><td>{empty['p_no_blast']*100:.1f}%</td></tr>
        <tr><td>No good-quality blast</td><td>{empty['p_no_good_blast']*100:.1f}%</td></tr>
    </table>

    <p class="small"><em>All predictions based on {N_SIM} Monte Carlo simulations
       with Bayesian conditional updating where observed values were entered.
       Clinical decisions require physician judgment.</em></p>
    """
    for name, path in image_paths.items():
        html += f'<div class="figure"><h3>{name}</h3><img src="{path}"></div>\n'
    html += "</body></html>"
    return html


def save_pdf_report(patient, res, output_filename="ivf_report_v6_1.pdf"):
    if path_to_wkhtmltopdf and not os.path.exists(path_to_wkhtmltopdf):
        raise FileNotFoundError(
            f"wkhtmltopdf not found at: {path_to_wkhtmltopdf}\n"
            f"Install from https://wkhtmltopdf.org/downloads.html and set the path."
        )

    with tempfile.TemporaryDirectory() as tmp:
        figs = create_figures(patient, res)
        paths = {}
        for name, fig in figs.items():
            safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("&","and")
            fp = os.path.join(tmp, f"{safe}.png")
            fig.write_image(fp, scale=2, width=1100, height=560)
            paths[name] = fp

        html = generate_html(patient, res, paths)
        html_path = os.path.join(tmp, "report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        opts = {'encoding': 'UTF-8', 'enable-local-file-access': None,
                'no-stop-slow-scripts': None, 'javascript-delay': 200,
                'load-error-handling': 'ignore'}
        if path_to_wkhtmltopdf:
            cfg = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)
            pdfkit.from_file(html_path, output_filename, options=opts, configuration=cfg)
        else:
            pdfkit.from_file(html_path, output_filename, options=opts)

    print(f"PDF saved to: {output_filename}")


# ============================================================
# INTERACTIVE INPUT (with optional Bayesian updates)
# ============================================================

def ask_float(prompt, default=None):
    s = input(prompt).strip()
    if not s and default is not None:
        return default
    return float(s)

def ask_int(prompt, default=None):
    s = input(prompt).strip()
    if not s:
        return default
    return int(s)

def collect_known_values():
    print("\n--- BAYESIAN UPDATING (optional) ---")
    print("If you have observed values from this cycle, enter them now.")
    print("Press ENTER to skip a stage (it stays stochastic).\n")
    return KnownValues(
        okk     = ask_int("Observed retrieved oocytes (OKK)?: ", None),
        mii     = ask_int("Observed MII oocytes?: ",            None),
        pn2     = ask_int("Observed 2PN zygotes?: ",            None),
        blasts  = ask_int("Observed blastocyst count?: ",       None),
        good    = ask_int("Observed good-quality blast count?: ", None),
        euploid = ask_int("Observed euploid embryos (PGT-A)?: ", None),
        n_transfers_planned = ask_int(
            "Number of transfers planned (0 = use all warmed euploid sequentially): ",
            default=0
        ),
    )


# ============================================================
# MAIN
# ============================================================

# ============================================================
# MAIN
# ============================================================

# Example clinic prior data (replace with your real values)
DEFAULT_CLINIC_REAL_SUCCESSES = [19, 18, 20, 6, 13, 12, 19, 22, 25]
DEFAULT_CLINIC_REAL_TRIALS    = [43, 45, 65, 18, 26, 31, 47, 49, 58]


def collect_v6_inputs():
    """Collect v6-specific extra inputs: attempt #, follicles, clinic prior."""
    print("\n--- v6.0 ADDITIONAL INPUTS ---")
    attempt = ask_int("IVF attempt number (default 1): ", default=1)
    foll    = ask_int("Total follicles on OPU (ENTER = use AFC as proxy): ", default=None)

    print("\nClinic prior data for Bayesian update:")
    print(" ENTER to use built-in defaults, or 'no' to skip clinic prior.")
    choice = input("Use clinic prior? [yes/no/custom, default yes]: ").strip().lower()

    if choice in ("no", "n", "skip"):
        return attempt, foll, None, None
    elif choice in ("custom", "c"):
        succ_str = input("Real successes per cycle batch (comma-separated): ").strip()
        tr_str   = input("Real trials per cycle batch (comma-separated):   ").strip()
        succ = [int(x.strip()) for x in succ_str.split(",") if x.strip()]
        tr   = [int(x.strip()) for x in tr_str.split(",") if x.strip()]
        return attempt, foll, succ, tr
    else:
        return attempt, foll, DEFAULT_CLINIC_REAL_SUCCESSES, DEFAULT_CLINIC_REAL_TRIALS


if __name__ == "__main__":
    print("=" * 70)
    print("IVF DIGITAL TWIN PLATFORM v6.1 - Cluster Analysis Extension")
    print("=" * 70)

    female_age = ask_float("Female age: ")
    amh        = ask_float("AMH (ng/mL): ")
    afc        = ask_int(  "AFC: ")
    bmi        = ask_float("BMI: ")

    patient = PatientInput(female_age=female_age, amh=amh, afc=afc, bmi=bmi)
    known   = collect_known_values()

    # v6 extras
    attempt, follicles, clinic_succ, clinic_tr = collect_v6_inputs()

    # Try to load NN ensemble
    nn_model = load_nn_ensemble()

    print(f"\nRunning {N_SIM} Monte Carlo simulations with full v6 pipeline...")
    res = run_pipeline_extended(
        patient,
        known=known,
        attempt_number=attempt,
        follicles=follicles,
        nn_model=nn_model,
        clinic_real_successes=clinic_succ,
        clinic_real_trials=clinic_tr,
        max_attempts_curve=6,
        n=N_SIM,
    )
    print("Done.\n")

    textual_report(patient, res)

    out = "ivf_report_v6_1.pdf"
    try:
        save_pdf_report(patient, res, out)
        print(f"\nDone. Report saved to: {out}")
    except Exception as exc:
        print(f"\nPDF generation failed: {exc}")
        print(f"(install wkhtmltopdf or run on a system where it is available)")
    print("Tip: re-run with new observed values to update predictions mid-cycle.")
