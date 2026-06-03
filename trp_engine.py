# ============================================================
# TRP ENGINE — Total Reproductive Potential Module
# IVF Digital Twin Platform, add-on v1.0
#
# Computes the couple's full reproductive timeline:
#   • AMH decline trajectory (log-linear, individual variability)
#   • MC fan of possible future cycle sequences
#   • Importance resampling conditioned on past cycle response
#   • Cumulative P(≥1 pregnancy) over N years / N cycles
#   • Window-of-opportunity metrics (P10/P50/P90)
#   • Plotly figures for Streamlit tab
#
# INTEGRATION — one line in app.py (after loading ivf_digital_twin):
#   from trp_engine import compute_trp, build_trp_tab
#
# STANDALONE demo:
#   python trp_engine.py
#
# No changes to ivf_digital_twin.py, app.py, or any other module.
# Self-contained: FORTUNE oracle re-implemented inline (2 equations).
# Optionally accepts a p_oracle_fn callback for KAT-based estimates.
#
# Literature basis:
#   AMH decline: Dölleman et al. Hum Reprod 2013; Tehrani et al. 2011
#   AFC-AMH:     Scheffer et al. Fertil Steril 2009
#   Cum. LBR:    McLernon et al. BMJ 2016; Malizia et al. NEJM 2009
#   Selection:   Malizia et al. NEJM 2009 (alpha = 0.08/attempt)
#   FORTUNE:     Calhaz-Jorge et al. Hum Reprod 2015
# ============================================================

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY = True
except ImportError:
    _PLOTLY = False


# ──────────────────────────────────────────────────────────────
# 1. DATA STRUCTURES
# ──────────────────────────────────────────────────────────────

@dataclass
class PastCycle:
    """
    Summary of one completed IVF cycle.
    At minimum provide age_at_cycle + okk_actual.
    All other fields are optional but improve conditioning.
    """
    age_at_cycle:   float                  # patient age at that cycle
    okk_actual:     Optional[int]  = None  # retrieved oocytes (actual)
    blasts_actual:  Optional[int]  = None  # total blastocysts
    good_actual:    Optional[int]  = None  # good-quality blastocysts
    outcome:        Optional[int]  = None  # 1 = clinical pregnancy / 0 = no
    amh_at_cycle:   Optional[float] = None # AMH at that time (ng/mL)
    afc_at_cycle:   Optional[int]   = None # AFC at that time
    cycle_index:    int             = 1    # 1-based attempt number


@dataclass
class TRPInput:
    """
    All inputs for the TRP simulation.
    """
    # ── current patient state ──────────────────────────────
    age:            float          # current age
    amh:            float          # current AMH (ng/mL)
    afc:            int            # current AFC
    bmi:            float  = 24.0
    sperm_source:   str    = "ejaculate"   # "ejaculate" | "donor" | "surgical"

    # ── past attempts (already completed) ─────────────────
    past_cycles:    List[PastCycle] = field(default_factory=list)

    # ── planning horizon ───────────────────────────────────
    max_future_cycles:  int   = 6      # max additional cycles willing to do
    desired_children:   int   = 1      # 1 or 2
    cycle_interval_mo:  float = 3.0    # months between cycles

    # ── clinical thresholds (window closure rules) ─────────
    amh_min:        float = 0.10   # ng/mL — below this: no stimulation
    age_max:        float = 45.0   # hard biological cutoff
    p_min_per_cycle: float = 0.03  # below this: clinically futile

    # ── MC parameters ──────────────────────────────────────
    n_trajectories: int   = 5_000
    seed:           int   = 42

    # ── calibrated base probability (from KAT pipeline) ──────
    # Pass res["p_overall_cycle"] from the main Digital Twin calculation.
    # If set, TRP uses this as the absolute anchor for cycle 0 and applies
    # relative FORTUNE corrections for future cycles — much more accurate
    # than raw FORTUNE absolute values.
    # If None, raw FORTUNE is used as fallback (conservative but uncalibrated).
    p_base_override: Optional[float] = field(default=None, repr=False)

    # ── optional external KAT oracle ───────────────────────
    # If provided, called as: p_oracle_fn(age, amh, afc, bmi, attempt) -> float
    # If None, FORTUNE formula is used (fast, ~same accuracy for TRP)
    p_oracle_fn: Optional[Callable] = field(default=None, repr=False)


@dataclass
class TRPResult:
    """Full TRP output bundle."""
    # ── cumulative probability curves ──────────────────────
    p_cum_by_cycle:   np.ndarray    # shape (max_future_cycles,)  P(≥1 success)
    p_cum_ci_lo:      np.ndarray    # P10
    p_cum_ci_hi:      np.ndarray    # P90

    # ── per-cycle probability (median trajectory) ──────────
    p_per_cycle_median: np.ndarray  # p(success) for cycle 1..N

    # ── window metrics ─────────────────────────────────────
    window_cycles_p10:  float       # pessimistic cycles within patient horizon
    window_cycles_p50:  float       # median cycles within patient horizon
    window_cycles_p90:  float       # optimistic cycles within patient horizon
    window_years_p10:   float       # biological window P10 (years to AMH_min/age_max)
    window_years_p50:   float       # biological window P50
    window_years_p90:   float       # biological window P90

    # ── summary scalars ────────────────────────────────────
    p_success_total:        float   # P(≥1 success within horizon)
    p_window_closes_first:  float   # P(window closes before success)
    expected_cycles_to_success: float

    # ── trajectory fan (for visualization) ─────────────────
    fan_success_times:  np.ndarray  # cycle index of success, NaN if none
    fan_window_sizes:   np.ndarray  # available cycles within patient horizon
    fan_window_years:   np.ndarray  # biological window in years per trajectory
    amh_trajectories:   np.ndarray  # shape (n_traj_sample, n_steps) for plot

    # ── AMH decline model parameters ───────────────────────
    k_median:   float
    k_p10:      float
    k_p90:      float

    # ── conditioning info ──────────────────────────────────
    response_ratio:     Optional[float]  # actual/predicted OKK ratio
    n_past_cycles:      int
    conditioning_used:  bool

    # ── input echo ─────────────────────────────────────────
    inp: TRPInput = field(repr=False)


# ──────────────────────────────────────────────────────────────
# 2. PROBABILITY ORACLES (inline, no external dependency)
# ──────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(np.clip(x, -30, 30))))


def _fortune_p_transfer(age: float, amh: float, bmi: float) -> float:
    """
    FORTUNE logit model → per-transfer live-birth probability.
    Calhaz-Jorge et al. Hum Reprod 2015; calibrated to SART 2018-2022.
    """
    z_a = (age - 36.3) / 5.5
    z_f = (math.log(amh + 0.1) - math.log(2.1)) / 1.2
    z_b = (bmi - 24.0) / 4.2
    logit = 0.40 - 0.55 * z_a + 0.15 * z_f - 0.20 * z_b
    return float(np.clip(_sigmoid(logit), 0.03, 0.90))


def _selection_decay(p_base: float, attempt: int, alpha: float = 0.08) -> float:
    """
    Logit-scale selection-effect decay (Malizia et al. NEJM 2009).
    ~8% relative decline per attempt in the mid-probability range.
    """
    if attempt <= 1 or p_base <= 0 or p_base >= 1:
        return float(np.clip(p_base, 0.01, 0.99))
    logit = math.log(p_base / (1.0 - p_base))
    return float(_sigmoid(logit - alpha * (attempt - 1)))


def _p_cycle(age: float, amh: float, bmi: float, attempt: int,
             p_oracle_fn: Optional[Callable]) -> float:
    """Single per-cycle probability via oracle or FORTUNE."""
    if p_oracle_fn is not None:
        try:
            p = float(p_oracle_fn(age, amh, 0, bmi, attempt))
            return float(np.clip(p, 0.01, 0.97))
        except Exception:
            pass
    p = _fortune_p_transfer(age, amh, bmi)
    return _selection_decay(p, attempt)


def _p_cycle_calibrated(age_now: float, amh_now: float, bmi: float,
                         step: int,
                         p_base: float,
                         age_base: float, amh_base: float,
                         decay_alpha: float = 0.08) -> float:
    """
    Per-cycle probability anchored to the KAT-calibrated p_base.

    Uses FORTUNE *relatively* (as a delta correction for biological
    change since the base cycle) rather than in absolute terms:

        logit(p_s) = logit(p_base)
                   + [logit_F(age_s, AMH_s) - logit_F(age_0, AMH_0)]
                   - alpha * step

    step=0  → returns p_base exactly (no correction needed)
    step>0  → applies age+AMH delta + selection decay

    This preserves the correct absolute calibration from the pipeline
    (p_overall_cycle) while correctly modelling how the probability
    changes as the patient ages and AMH declines.
    """
    if step == 0:
        return float(np.clip(p_base, 0.01, 0.97))

    p_b = float(np.clip(p_base, 0.01, 0.99))
    logit_base = math.log(p_b / (1.0 - p_b))

    # FORTUNE delta (relative change only, not absolute value)
    p_f_now  = _fortune_p_transfer(age_now,  amh_now,  bmi)
    p_f_base = _fortune_p_transfer(age_base, amh_base, bmi)
    p_f_now  = float(np.clip(p_f_now,  0.01, 0.99))
    p_f_base = float(np.clip(p_f_base, 0.01, 0.99))
    delta_logit_fortune = (math.log(p_f_now  / (1.0 - p_f_now)) -
                           math.log(p_f_base / (1.0 - p_f_base)))

    # Selection decay for additional attempts beyond the base
    logit_new = logit_base + delta_logit_fortune - decay_alpha * step

    return float(_sigmoid(logit_new))


# ──────────────────────────────────────────────────────────────
# 3. AMH DECLINE MODEL
# ──────────────────────────────────────────────────────────────

# Log-linear decline: AMH(t) = AMH_0 * exp(-k * t_years)
# k distribution from Dölleman et al. Hum Reprod 2013 +
# Tehrani et al. Fertil Steril 2011 (longitudinal cohorts).
# Median k ≈ 0.07 /year; large inter-individual σ.
_AMH_K_MU_LOG   = math.log(0.07)   # lognormal mean of log(k)
_AMH_K_SIGMA_LOG = 0.45            # lognormal sd of log(k); wide IQ range

def _afc_from_amh(amh: float, age: float) -> float:
    """
    AFC ≈ f(AMH, age). Scheffer et al. Fertil Steril 2009 approximation.
    Returns estimated AFC as float.
    """
    # Scheffer 2009 + van Rooij 2002 calibration
    afc = 3.0 + 5.5 * math.sqrt(max(amh, 0.01)) - 0.10 * max(age - 33, 0)
    return max(float(afc), 1.0)


def _amh_at_t(amh_0: float, k: float, t_years: float) -> float:
    """AMH at time t_years given individual decline rate k."""
    return float(amh_0 * math.exp(-k * t_years))


# ──────────────────────────────────────────────────────────────
# 4. RESPONSE RATIO — conditioning weight
# ──────────────────────────────────────────────────────────────

def _compute_response_ratio(past_cycles: List[PastCycle],
                             bmi: float) -> Optional[float]:
    """
    Ratio of actual OKK to model-predicted OKK across past cycles.
    response_ratio > 1 → better-than-expected responder.
    response_ratio < 1 → worse-than-expected.
    None if no past OKK data available.
    """
    ratios = []
    for pc in past_cycles:
        if pc.okk_actual is None:
            continue
        # Predict OKK at that time: use AMH at that cycle if known,
        # otherwise use age-only FORTUNE (rough but consistent).
        amh_then = pc.amh_at_cycle if pc.amh_at_cycle is not None else 2.0
        # Expected OKK: calibrated log-linear model to SART/HFEA cohorts.
        # Coefficients: AMH 2 ng/mL age 36 → ~11 eggs;
        #               AMH 0.5 → ~4-5 eggs; AMH 5 → ~16 eggs.
        # E[OKK] = exp(1.925 + 0.64*log(AMH+0.1) - 0.030*(age-36))
        log_mu = (1.925
                  + 0.64 * math.log(amh_then + 0.1)
                  - 0.030 * (pc.age_at_cycle - 36.0))
        expected_okk = max(math.exp(log_mu), 1.0)
        ratios.append(pc.okk_actual / expected_okk)

    if not ratios:
        return None
    # Geometric mean of ratios (more robust than arithmetic for this)
    return float(math.exp(np.mean(np.log(np.clip(ratios, 0.05, 5.0)))))


def _importance_weights(sim_okk: np.ndarray,
                         response_ratio: float,
                         pred_okk: float,
                         sigma_ratio: float = 0.35) -> np.ndarray:
    """
    Weight each MC sample by how consistent it is with the
    patient's observed response_ratio.
    w_i ∝ N(sim_okk_i | mu=ratio*pred_okk, sigma=sigma_ratio*pred_okk)
    """
    mu    = response_ratio * pred_okk
    sigma = max(sigma_ratio * pred_okk, 1.0)
    log_w = -0.5 * ((sim_okk - mu) / sigma) ** 2
    w = np.exp(log_w - log_w.max())
    w_sum = w.sum()
    if w_sum < 1e-12:
        return np.ones(len(sim_okk)) / len(sim_okk)
    return w / w_sum


# ──────────────────────────────────────────────────────────────
# 5. MAIN MC SIMULATION
# ──────────────────────────────────────────────────────────────

def compute_trp(inp: TRPInput) -> TRPResult:
    """
    Run the TRP Monte Carlo simulation.

    For each of N trajectories:
      1. Draw individual AMH decline rate k ~ LogNormal
      2. For each future cycle step:
         a. Update age and AMH
         b. Check window-closure conditions
         c. Evaluate per-cycle success probability
         d. Sample outcome ~ Bernoulli(p)
         e. Stop on success or window closure
      3. Apply importance resampling if past_cycles provided

    Returns TRPResult with all metrics and plot data.
    """
    rng = np.random.default_rng(inp.seed)
    N   = inp.n_trajectories
    M   = inp.max_future_cycles
    dt  = inp.cycle_interval_mo / 12.0   # cycle spacing in years

    # ── 4a. Draw decline rates ──────────────────────────────
    k_samples = rng.lognormal(
        mean=_AMH_K_MU_LOG,
        sigma=_AMH_K_SIGMA_LOG,
        size=N
    )
    k_median = float(np.median(k_samples))
    k_p10    = float(np.percentile(k_samples, 10))
    k_p90    = float(np.percentile(k_samples, 90))

    # ── 4b. Compute response_ratio for conditioning ─────────
    n_past = len(inp.past_cycles)
    rr = _compute_response_ratio(inp.past_cycles, inp.bmi)
    conditioning_used = rr is not None

    # Expected OKK at current parameters (for resampling weight)
    log_mu_now = (1.925
                  + 0.64 * math.log(inp.amh + 0.1)
                  - 0.030 * (inp.age - 36.0))
    pred_okk_now = max(math.exp(log_mu_now), 1.0)

    # Number of past attempts (total, including current)
    n_past_total = n_past + 1  # current = attempt n_past + 1

    # ── p_base: anchored probability for cycle 0 ───────────
    # Use KAT-calibrated p_overall_cycle if provided, else FORTUNE.
    _p_base = inp.p_base_override  # None → FORTUNE fallback
    _use_calibrated = (_p_base is not None and 0.01 <= _p_base <= 0.99)
    if _use_calibrated:
        _p_base = float(np.clip(_p_base, 0.01, 0.99))

    # ── 4c. Per-trajectory simulation ──────────────────────
    # Arrays: each row = one trajectory
    success_at  = np.full(N, np.nan)   # cycle index (1-based) of success
    window_size       = np.zeros(N, dtype=int)   # cycles within patient horizon
    bio_window_years  = np.zeros(N, dtype=float) # biological window in years
    # AMH matrix for plot (sub-sample 200 trajectories)
    n_plot = min(200, N)
    amh_mat = np.zeros((n_plot, M + 1))
    amh_mat[:, 0] = inp.amh

    # Compute weights for importance resampling (based on OKK at step 0)
    if conditioning_used:
        # Sample "hypothetical OKK" for step-0 from the ZINB-like distribution
        sim_okk_0 = rng.poisson(pred_okk_now, size=N).astype(float)
        weights = _importance_weights(sim_okk_0, rr, pred_okk_now)
    else:
        weights = np.ones(N) / N

    for i in range(N):
        k_i    = k_samples[i]
        avail  = 0      # biological window: cycles available before closure
        first_success_set = False

        # ── Pass 1: count biological window (ignore success/failure) ──
        for step in range(M):
            t_years = step * dt
            age_now = inp.age + t_years
            amh_now = _amh_at_t(inp.amh, k_i, t_years)
            if i < n_plot:
                amh_mat[i, step + 1] = amh_now
            if amh_now < inp.amh_min:
                break
            if age_now >= inp.age_max:
                break
            attempt_abs = n_past_total + step
            if _use_calibrated:
                p = _p_cycle_calibrated(
                    age_now, amh_now, inp.bmi,
                    step=step,
                    p_base=_p_base,
                    age_base=inp.age,
                    amh_base=inp.amh,
                )
            else:
                p = _p_cycle(age_now, amh_now, inp.bmi,
                             attempt_abs, inp.p_oracle_fn)
            if p < inp.p_min_per_cycle:
                break
            avail += 1
        window_size[i] = avail

        # ── Pass 1b: biological window in years (scan up to 30yr) ────
        # Independent of max_future_cycles — asks "when does the
        # biological window actually close?" for THIS trajectory.
        # Scanned at monthly resolution; stops at amh_min OR age_max.
        _bio_win_yr = 30.0  # default: open beyond scan horizon
        for _mo in range(1, 361):           # 1 month .. 30 years
            _t  = _mo / 12.0
            _an = _amh_at_t(inp.amh, k_i, _t)
            if _an < inp.amh_min:
                _bio_win_yr = _t
                break
            if inp.age + _t >= inp.age_max:
                _bio_win_yr = _t
                break
        bio_window_years[i] = _bio_win_yr

        # ── Pass 2: simulate outcomes within the window ──────────────
        for step in range(avail):
            t_years = step * dt
            age_now = inp.age + t_years
            amh_now = _amh_at_t(inp.amh, k_i, t_years)
            attempt_abs = n_past_total + step
            if _use_calibrated:
                p = _p_cycle_calibrated(
                    age_now, amh_now, inp.bmi,
                    step=step,
                    p_base=_p_base,
                    age_base=inp.age,
                    amh_base=inp.amh,
                )
            else:
                p = _p_cycle(age_now, amh_now, inp.bmi,
                             attempt_abs, inp.p_oracle_fn)
            outcome = rng.binomial(1, p)
            if outcome == 1:
                success_at[i] = step + 1   # 1-based cycle index
                first_success_set = True
                break

    # ── 4d. Importance resampling ───────────────────────────
    if conditioning_used:
        resample_idx = rng.choice(N, size=N, replace=True, p=weights)
        success_at        = success_at[resample_idx]
        window_size       = window_size[resample_idx]
        bio_window_years  = bio_window_years[resample_idx]

    # ── 4e. Compute cumulative probability curves ───────────
    p_cum        = np.zeros(M)
    p_cum_lo     = np.zeros(M)
    p_cum_hi     = np.zeros(M)
    p_per_median = np.zeros(M)

    for c in range(1, M + 1):
        # Fraction of trajectories that succeeded by cycle c
        success_by_c = np.nansum(success_at <= c) / N
        p_cum[c - 1] = success_by_c

        # Bootstrap CI (P10/P90) via 200 resamples
        boot = rng.choice(N, size=(200, N), replace=True)
        boot_p = np.array([
            np.nansum(success_at[b] <= c) / N for b in boot
        ])
        p_cum_lo[c - 1] = float(np.percentile(boot_p, 10))
        p_cum_hi[c - 1] = float(np.percentile(boot_p, 90))

    # Per-cycle marginal p (median trajectory)
    for c in range(1, M + 1):
        t_years = (c - 1) * dt
        age_c   = inp.age + t_years
        amh_c   = _amh_at_t(inp.amh, k_median, t_years)
        att_c   = n_past_total + (c - 1)
        if _use_calibrated:
            p_per_median[c - 1] = _p_cycle_calibrated(
                age_c, amh_c, inp.bmi,
                step=c - 1,
                p_base=_p_base,
                age_base=inp.age,
                amh_base=inp.amh,
            )
        else:
            p_per_median[c - 1] = _p_cycle(age_c, amh_c, inp.bmi,
                                            att_c, inp.p_oracle_fn)

    # ── 4f. Window metrics ──────────────────────────────────
    ws_valid = window_size[window_size > 0]
    if len(ws_valid) == 0:
        ws_valid = np.array([0])
    win_p10 = float(np.percentile(ws_valid, 10))
    win_p50 = float(np.percentile(ws_valid, 50))
    win_p90 = float(np.percentile(ws_valid, 90))
    win_years_p50 = win_p50 * dt    # cycles → years within horizon

    # Biological window percentiles (in years, from monthly scan)
    bw_p10 = float(np.percentile(bio_window_years, 10))
    bw_p50 = float(np.percentile(bio_window_years, 50))
    bw_p90 = float(np.percentile(bio_window_years, 90))

    # ── 4g. Summary scalars ─────────────────────────────────
    p_success_total = float(np.nansum(~np.isnan(success_at)) / N)
    p_window_first  = float(np.nansum(np.isnan(success_at)) / N)

    valid_success = success_at[~np.isnan(success_at)]
    exp_cycles = float(np.mean(valid_success)) if len(valid_success) > 0 else np.nan

    return TRPResult(
        p_cum_by_cycle      = p_cum,
        p_cum_ci_lo         = p_cum_lo,
        p_cum_ci_hi         = p_cum_hi,
        p_per_cycle_median  = p_per_median,
        window_cycles_p10   = win_p10,
        window_cycles_p50   = win_p50,
        window_cycles_p90   = win_p90,
        window_years_p10    = bw_p10,
        window_years_p50    = bw_p50,
        window_years_p90    = bw_p90,
        p_success_total     = p_success_total,
        p_window_closes_first = p_window_first,
        expected_cycles_to_success = exp_cycles,
        fan_success_times   = success_at,
        fan_window_sizes    = window_size,
        fan_window_years    = bio_window_years,
        amh_trajectories    = amh_mat,
        k_median            = k_median,
        k_p10               = k_p10,
        k_p90               = k_p90,
        response_ratio      = rr,
        n_past_cycles       = n_past,
        conditioning_used   = conditioning_used,
        inp                 = inp,
    )


# ──────────────────────────────────────────────────────────────
# 6. PLOTLY FIGURES
# ──────────────────────────────────────────────────────────────

# Pipeline-matched palette (same as app.py C dict + constants)
_TEAL   = "#78AAA5"   # C["teal"]
_AMBER  = "#D9A36A"   # C["orange"]
_RED    = "#C98282"   # C["red"]
_PURPLE = "#A792C6"   # C["purple"]
_GRAY   = "#71808C"   # C["grey"]
_BLUE   = "#6F93B7"   # C["blue"]
_GREEN  = "#8DBA8D"   # C["green"]
_BGCOL  = "rgba(0,0,0,0)"
_PLOT_BG = "rgba(0,0,0,0)"
_GRID   = "rgba(115,132,145,0.16)"
_AXIS   = "rgba(115,132,145,0.42)"
_FONT_D = dict(family="Inter, Arial, sans-serif", size=12, color="#243746")
_HATCHES = ["/", "\\\\", "x", "-", "|", "+", "."]


def _base_layout(**extra):
    kw = dict(
        paper_bgcolor=_BGCOL,
        plot_bgcolor=_PLOT_BG,
        font=_FONT_D,
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="rgba(196,211,222,0.95)",
            font=dict(family="Inter, Arial, sans-serif", size=12, color="#243746"),
        ),
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=1.04, yanchor="bottom",
            bgcolor="rgba(255,255,255,0.72)",
            bordercolor="rgba(196,211,222,0.80)",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(t=70, b=55, l=60, r=40),
    )
    kw.update(extra)
    return kw


def _ax():
    return dict(
        showline=True, linecolor=_AXIS, linewidth=1,
        gridcolor=_GRID, zeroline=False,
        tickfont=dict(size=11, color="#526473"),
        title_font=dict(size=12, color="#405565"),
    )


def _fig_cumulative(res: TRPResult) -> "go.Figure":
    """Cumulative probability of ≥1 clinical pregnancy vs cycle number."""
    if not _PLOTLY:
        return None

    M      = res.inp.max_future_cycles
    cycles = list(range(1, M + 1))
    p_cum  = res.p_cum_by_cycle * 100
    p_lo   = res.p_cum_ci_lo    * 100
    p_hi   = res.p_cum_ci_hi    * 100

    fig = go.Figure()

    # P10–P90 band
    fig.add_trace(go.Scatter(
        x=cycles + cycles[::-1],
        y=list(p_hi) + list(p_lo[::-1]),
        fill="toself",
        fillcolor="rgba(120,170,165,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="P10–P90",
        hoverinfo="skip",
    ))

    # Conditioning label
    cond_label = (
        f"С учётом истории (RR={res.response_ratio:.2f})"
        if res.conditioning_used else "По популяционной модели"
    )

    # Main curve
    fig.add_trace(go.Scatter(
        x=cycles, y=list(p_cum),
        mode="lines+markers",
        line=dict(color=_TEAL, width=2.5),
        marker=dict(size=7, color=_TEAL),
        name=cond_label,
    ))

    # 90% threshold
    fig.add_hline(y=90, line_dash="dot", line_color=_AMBER,
                  annotation_text="90%", annotation_position="right")

    # No window vline on cumulative figure (window is now in years, different axis)

    fig.update_layout(
        title="Кумулятивная вероятность ≥1 беременности",
        xaxis_title="Число будущих циклов",
        yaxis_title="P(≥1 беременность), %",
        yaxis=dict(range=[0, 100], **_ax()),
        xaxis=dict(tickvals=cycles, **_ax()),
        **_base_layout(),
    )
    return fig


def _fig_amh_trajectories(res: TRPResult) -> "go.Figure":
    """AMH decline fan with window-closure thresholds."""
    if not _PLOTLY:
        return None

    inp   = res.inp
    M     = inp.max_future_cycles
    dt    = inp.cycle_interval_mo / 12.0
    times = [round(step * dt, 2) for step in range(M + 1)]

    fig = go.Figure()

    # Fan: first 80 trajectories (thin, semi-transparent)
    n_fan = min(80, res.amh_trajectories.shape[0])
    for i in range(n_fan):
        traj = res.amh_trajectories[i, :]
        # truncate at window closure
        amh_vals = [float(v) for v in traj]
        fig.add_trace(go.Scatter(
            x=times, y=amh_vals,
            mode="lines",
            line=dict(color="rgba(120,170,165,0.10)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # Median trajectory
    amh_med = [_amh_at_t(inp.amh, res.k_median, t) for t in times]
    amh_p10 = [_amh_at_t(inp.amh, res.k_p10,    t) for t in times]
    amh_p90 = [_amh_at_t(inp.amh, res.k_p90,    t) for t in times]

    fig.add_trace(go.Scatter(
        x=times, y=amh_p90,
        mode="lines", line=dict(color=_TEAL, dash="dot", width=1.5),
        name="P90 (медленное снижение)", opacity=0.75,
    ))
    fig.add_trace(go.Scatter(
        x=times, y=amh_med,
        mode="lines+markers",
        line=dict(color=_BLUE, width=2.5),
        marker=dict(size=6, color=_BLUE),
        name="P50 медиана",
    ))
    fig.add_trace(go.Scatter(
        x=times, y=amh_p10,
        mode="lines", line=dict(color=_RED, dash="dot", width=1.5),
        name="P10 (быстрое снижение)", opacity=0.75,
    ))

    # AMH_min threshold
    fig.add_hline(y=inp.amh_min,
                  line_dash="dash", line_color=_RED, line_width=1.5,
                  annotation_text=f"АМГ_min = {inp.amh_min} нг/мл",
                  annotation_position="right")

    # Cycle markers on x-axis
    fig.update_layout(
        title="Прогноз снижения АМГ — веер индивидуальных траекторий",
        xaxis_title="Лет от текущего момента",
        yaxis_title="АМГ (нг/мл)",
        xaxis=_ax(),
        yaxis=_ax(),
        **_base_layout(),
    )
    return fig


def _fig_window_distribution(res: TRPResult) -> "go.Figure":
    """
    Per-cycle pregnancy probability chart.

    Uses p_per_cycle_median from the MC simulation — this already
    encodes both biological drift (age + AMH decline via FORTUNE delta)
    and selection-effect decay (-0.08 per step, Malizia 2009).

    A second reference line shows the pure biological component
    (no selection decay) so the doctor can see both effects separately.
    Y-axis is auto-scaled to the actual data range for clear visibility.
    """
    if not _PLOTLY:
        return None

    inp  = res.inp
    M    = inp.max_future_cycles
    dt   = inp.cycle_interval_mo / 12.0
    ages = [round(inp.age + s * dt, 1) for s in range(M)]

    # ── Main curve: p_per_cycle_median from MC (full model) ──────────
    p_mc = [v * 100 for v in res.p_per_cycle_median]

    # ── Reference: biological drift only (no selection decay) ────────
    p_base = inp.p_base_override
    use_cal = (p_base is not None and 0.01 <= p_base <= 0.99)

    p_bio = []
    if use_cal:
        _p_f_base = float(np.clip(
            _fortune_p_transfer(inp.age, inp.amh, inp.bmi), 0.01, 0.99))
        _lf_base = math.log(_p_f_base / (1.0 - _p_f_base))
        _lp_base = math.log(p_base / (1.0 - p_base))
        for s in range(M):
            t     = s * dt
            amh_s = _amh_at_t(inp.amh, res.k_median, t)
            age_s = inp.age + t
            p_fn  = float(np.clip(
                _fortune_p_transfer(age_s, amh_s, inp.bmi), 0.01, 0.99))
            delta = math.log(p_fn / (1.0 - p_fn)) - _lf_base
            p_bio.append(_sigmoid(_lp_base + delta) * 100)

    fig = go.Figure()

    # ── Bars: full model (selection + biological) ─────────────────────
    fig.add_trace(go.Bar(
        x=list(range(1, M + 1)),
        y=p_mc,
        name="P за цикл (модель)",
        marker=dict(
            color=f"rgba(111,147,183,0.65)",
            line=dict(color="rgba(111,147,183,0.90)", width=1.2),
            pattern=dict(shape="/", solidity=0.12,
                         fgcolor="rgba(75,92,105,0.22)",
                         bgcolor="rgba(255,255,255,0)"),
        ),
        text=[f"{v:.1f}%" for v in p_mc],
        textposition="outside",
        customdata=ages,
        hovertemplate="Цикл %{x}<br>Возраст: %{customdata}<br>P = %{y:.1f}%<extra></extra>",
    ))

    # ── Reference line: biological only ──────────────────────────────
    if p_bio:
        fig.add_trace(go.Scatter(
            x=list(range(1, M + 1)),
            y=p_bio,
            mode="lines+markers",
            line=dict(color=_GRAY, width=1.5, dash="dot"),
            marker=dict(size=5, color=_GRAY, opacity=0.75),
            opacity=0.80,
            name="Только биологический спад (без эффекта отбора)",
        ))

    # ── Y-axis: auto-scale with 15% headroom ─────────────────────────
    all_vals = p_mc + (p_bio if p_bio else [])
    y_max = max(all_vals) * 1.22
    y_min = max(0.0, min(all_vals) * 0.75)

    # ── X-axis: cycle number + age labels ────────────────────────────
    tick_labels = [f"Цикл {s+1}<br><sup>возраст {ages[s]:.0f}</sup>"
                   for s in range(M)]

    # Anchor info
    anchor_txt = (f"Якорь KAT: {p_base*100:.1f}%"
                  if use_cal else "Якорь: FORTUNE (без калибровки)")

    fig.update_layout(
        title=(f"Вероятность беременности за каждый будущий цикл<br>"
               f"<sup>{anchor_txt} · отбор α=0.08 · k_AMG={res.k_median:.3f}/год</sup>"),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(1, M + 1)),
            ticktext=tick_labels,
            **_ax(),
        ),
        yaxis=dict(
            title="P(беременность за 1 цикл), %",
            range=[y_min, y_max],
            **_ax(),
        ),
        barmode="group",
        **_base_layout(),
    )
    return fig


def _fig_trajectory_fan(res: TRPResult) -> "go.Figure":
    """
    Per-cycle success probability on median trajectory,
    with cumulative overlay.
    """
    if not _PLOTLY:
        return None

    M      = res.inp.max_future_cycles
    cycles = list(range(1, M + 1))
    p_per  = res.p_per_cycle_median * 100
    p_cum  = res.p_cum_by_cycle     * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=cycles, y=list(p_per),
        name="P беременности за цикл (медиана)",
        marker=dict(
            color="rgba(120,170,165,0.65)",
            line=dict(color="rgba(120,170,165,0.90)", width=1.2),
            pattern=dict(shape="\\", solidity=0.12,
                         fgcolor="rgba(75,92,105,0.22)",
                         bgcolor="rgba(255,255,255,0)"),
        ),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=cycles, y=list(p_cum),
        mode="lines+markers",
        line=dict(color=_PURPLE, width=2.5),
        marker=dict(size=7, color=_PURPLE, opacity=0.85),
        opacity=0.90,
        name="Кумулятивная P (правая ось)",
    ), secondary_y=True)

    fig.update_layout(
        title="Шанс за цикл vs кумулятивная вероятность",
        xaxis=_ax(),
        **_base_layout(),
    )
    fig.update_yaxes(title_text="P за 1 цикл, %",   secondary_y=False,
                     range=[0, 80], **_ax())
    fig.update_yaxes(title_text="Кумулятивная P, %", secondary_y=True,
                     range=[0, 100], **_ax())
    return fig


def build_figures(res: TRPResult) -> Dict[str, "go.Figure"]:
    """Return all four Plotly figures as a dict."""
    return {
        "cumulative":    _fig_cumulative(res),
        "amh":           _fig_amh_trajectories(res),
        "window":        _fig_window_distribution(res),
        "per_cycle":     _fig_trajectory_fan(res),
    }


# ──────────────────────────────────────────────────────────────
# 7. STREAMLIT INTEGRATION (zero changes to app.py internals)
# ──────────────────────────────────────────────────────────────

def build_trp_tab(res: TRPResult, theme_fn=None) -> None:
    """
    Render a complete TRP tab in Streamlit.
    Call this from app.py inside a `with tab_trp:` block.

    Usage in app.py:
        from trp_engine import compute_trp, build_trp_tab, TRPInput, PastCycle
        # ... collect inputs from Streamlit widgets ...
        trp_inp = TRPInput(age=age, amh=amh, afc=afc, bmi=bmi, ...)
        trp_res = compute_trp(trp_inp)
        with tab_trp:
            build_trp_tab(trp_res)
    """
    try:
        import streamlit as st
    except ImportError:
        raise RuntimeError("Streamlit not available; use compute_trp() directly.")

    inp = res.inp

    # ── KPI cards ───────────────────────────────────────────
    st.subheader("Совокупный репродуктивный потенциал (TRP)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "P(≥1 беременность за горизонт)",
            f"{res.p_success_total * 100:.1f}%",
            help="Вероятность хотя бы одной клинической беременности за горизонт планирования",
        )
    with c2:
        st.metric(
            "Биологическое окно (P50)",
            f"{res.window_years_p50:.1f} лет",
            delta=f"P10={res.window_years_p10:.1f} / P90={res.window_years_p90:.1f} лет",
            help="Лет до закрытия биологического окна (АМГ < порога или возраст_max). "
                 "Рассчитывается независимо от горизонта планирования пары.",
        )
    with c3:
        if not math.isnan(res.expected_cycles_to_success):
            st.metric(
                "Ожидаемо циклов до беременности",
                f"{res.expected_cycles_to_success:.1f}",
                help="Среднее число попыток до первой беременности (по успешным траекториям)",
            )
        else:
            st.metric("Ожидаемо циклов до беременности", "—")
    with c4:
        st.metric(
            "P(окно закроется первым)",
            f"{res.p_window_closes_first * 100:.1f}%",
            delta="риск",
            delta_color="inverse",
            help="Вероятность того, что биологическое окно закроется до наступления беременности",
        )

    # ── Conditioning notice ─────────────────────────────────
    if res.conditioning_used:
        rr = res.response_ratio
        label = (
            "Слабый ответчик" if rr < 0.75 else
            "Сильный ответчик" if rr > 1.25 else
            "Типичный ответ"
        )
        st.info(
            f"**Кондиционирование активно.** "
            f"Response ratio = {rr:.2f} ({label}). "
            f"MC-траектории взвешены по истории {res.n_past_cycles} предыдущих попыток."
        )
    else:
        st.info(
            "Данные предыдущих циклов не указаны — используется популяционная модель. "
            "Добавьте данные прошлых попыток для patient-specific прогноза."
        )

    # ── Figures ─────────────────────────────────────────────
    figs = build_figures(res)
    _th = theme_fn if callable(theme_fn) else (lambda f: f)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Кумулятивная P", "AMГ траектории",
        "Спад P по возрасту", "P за цикл"
    ])
    with tab1:
        st.plotly_chart(_th(figs["cumulative"]), use_container_width=True)
    with tab2:
        st.plotly_chart(_th(figs["amh"]),        use_container_width=True)
    with tab3:
        st.plotly_chart(_th(figs["window"]),     use_container_width=True)
    with tab4:
        st.plotly_chart(_th(figs["per_cycle"]),  use_container_width=True)

    # ── Interpretation text ─────────────────────────────────
    with st.expander("Интерпретация результатов"):
        st.markdown(f"""
**Спад вероятности.** График "Спад P по возрасту" показывает три сценария
снижения АМГ (P10 / P50 / P90 по скорости k). Разброс траекторий невелик
({(res.window_years_p90 - res.window_years_p10):.1f} лет между P10 и P90)
потому что модель FORTUNE возраст-доминирована — возраст является
главным предиктором, АМГ добавляет поправку.
Возрастной горизонт до закрытия окна: {res.window_years_p50:.1f} лет (до возраста {res.inp.age_max:.0f}).

**Риск закрытия раньше беременности** = **{res.p_window_closes_first*100:.1f}%**.
{"Этот показатель выше 30% — целесообразно обсудить тактику freeze-all или донорские ооциты." if res.p_window_closes_first > 0.30 else "Риск в пределах нормы для данного профиля."}

**Снижение АМГ.** Медианный темп k = {res.k_median:.3f}/год
(диапазон P10–P90: {res.k_p10:.3f}–{res.k_p90:.3f}).
{"Индивидуальная вариабельность велика — регулярный мониторинг АМГ важен для актуализации прогноза." if res.k_p90 / max(res.k_p10, 0.001) > 3 else ""}

**Методология.** {res.inp.n_trajectories:,} MC-траектории. Оракул вероятности: FORTUNE
{f"+ кондиционирование (response_ratio={res.response_ratio:.2f})" if res.conditioning_used else "(без кондиционирования)"}.
АМГ-снижение: лог-линейная модель k~LogNormal по Dölleman et al. Hum Reprod 2013.
        """)


# ──────────────────────────────────────────────────────────────
# 8. STREAMLIT INPUT WIDGETS HELPER
# ──────────────────────────────────────────────────────────────

def build_trp_inputs(current_age: float, current_amh: float,
                      current_afc: int, current_bmi: float,
                      p_base: Optional[float] = None) -> TRPInput:
    """
    Render TRP input expander in Streamlit and return TRPInput.

    Parameters
    ----------
    p_base : float or None
        KAT-calibrated per-cycle pregnancy probability (p_overall_cycle
        from the main Digital Twin pipeline). If provided, TRP uses this
        as the absolute anchor instead of raw FORTUNE values.

    Usage in app.py:
        trp_inp = build_trp_inputs(age, amh, afc, bmi,
                                   p_base=res.get("p_overall_cycle"))
        trp_res = compute_trp(trp_inp)
        build_trp_tab(trp_res)
    """
    try:
        import streamlit as st
    except ImportError:
        raise RuntimeError("Streamlit not available.")

    with st.expander("Настройки TRP — репродуктивный горизонт", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            max_cycles = st.slider(
                "Максимум будущих попыток", 1, 10, 6, key="trp_max_cycles"
            )
            desired_ch = st.radio(
                "Желаемых детей", [1, 2], horizontal=True, key="trp_children"
            )
            interval_mo = st.slider(
                "Интервал между циклами (мес)", 2, 12, 3, key="trp_interval"
            )
        with col2:
            amh_min = st.number_input(
                "Порог АМГ (нг/мл)", value=0.10, step=0.05,
                format="%.2f", key="trp_amh_min"
            )
            age_max = st.number_input(
                "Возрастной лимит", value=45.0, step=0.5,
                format="%.1f", key="trp_age_max"
            )
            n_traj = st.select_slider(
                "Точность (траектории)", [1000, 2000, 5000, 10000],
                value=5000, key="trp_n_traj"
            )

        st.markdown("**Предыдущие попытки** (опционально)")
        n_past = st.number_input(
            "Количество прошлых циклов", 0, 5, 0, key="trp_n_past"
        )

        past_cycles = []
        for i in range(int(n_past)):
            with st.container():
                st.caption(f"Попытка #{i+1}")
                pc1, pc2, pc3, pc4, pc5 = st.columns(5)
                with pc1:
                    age_pc = st.number_input(
                        "Возраст", 20.0, 50.0, float(current_age - (int(n_past)-i)*1.0),
                        step=0.5, key=f"trp_pc_age_{i}"
                    )
                with pc2:
                    okk_pc = st.number_input(
                        "ОКК факт", 0, 60, 8, key=f"trp_pc_okk_{i}"
                    )
                with pc3:
                    bl_pc = st.number_input(
                        "Бластоцисты", 0, 30, 2, key=f"trp_pc_bl_{i}"
                    )
                with pc4:
                    amh_pc = st.number_input(
                        "АМГ (тогда)", 0.0, 20.0, float(current_amh),
                        step=0.1, key=f"trp_pc_amh_{i}"
                    )
                with pc5:
                    out_pc = st.selectbox(
                        "Исход", ["Нет данных", "Успех", "Неудача"],
                        key=f"trp_pc_out_{i}"
                    )
                outcome_val = (
                    None if out_pc == "Нет данных" else
                    1    if out_pc == "Успех" else 0
                )
                past_cycles.append(PastCycle(
                    age_at_cycle   = float(age_pc),
                    okk_actual     = int(okk_pc),
                    blasts_actual  = int(bl_pc),
                    outcome        = outcome_val,
                    amh_at_cycle   = float(amh_pc),
                    cycle_index    = i + 1,
                ))

    return TRPInput(
        age               = float(current_age),
        amh               = float(current_amh),
        afc               = int(current_afc),
        bmi               = float(current_bmi),
        past_cycles       = past_cycles,
        max_future_cycles = int(max_cycles),
        desired_children  = int(desired_ch),
        cycle_interval_mo = float(interval_mo),
        amh_min           = float(amh_min),
        age_max           = float(age_max),
        n_trajectories    = int(n_traj),
        p_base_override   = float(p_base) if p_base is not None else None,
    )


# ──────────────────────────────────────────────────────────────
# 9. QUICK INTEGRATION SNIPPET (copy to app.py)
# ──────────────────────────────────────────────────────────────

_INTEGRATION_SNIPPET = '''
# ── TRP TAB — add after existing tabs in app.py ──────────────
# (1) Import at top of app.py (after ivf_digital_twin is loaded):
#     from trp_engine import compute_trp, build_trp_tab, build_trp_inputs
#     from trp_engine import TRPInput, PastCycle
#
# (2) Add a tab:
#     tab_main, tab_trp, ... = st.tabs(["Основной", "TRP", ...])
#
# (3) Inside tab_trp:
#     with tab_trp:
#         trp_inp = build_trp_inputs(age, amh, afc, bmi)
#         if st.button("Рассчитать TRP", key="run_trp"):
#             with st.spinner("MC-симуляция..."):
#                 trp_res = compute_trp(trp_inp)
#             st.session_state["trp_res"] = trp_res
#         if "trp_res" in st.session_state:
#             build_trp_tab(st.session_state["trp_res"])
# ─────────────────────────────────────────────────────────────
'''


# ──────────────────────────────────────────────────────────────
# 10. CLI DEMO
# ──────────────────────────────────────────────────────────────

def _demo():
    """Quick self-test & printout."""
    print("=" * 60)
    print("TRP ENGINE — demo run")
    print("=" * 60)

    # Scenario 1: 36-year-old, AMH 1.8, no history
    # p_base=0.38 simulates a typical KAT p_overall_cycle output
    inp1 = TRPInput(age=36, amh=1.8, afc=10, bmi=23.5,
                    max_future_cycles=6, n_trajectories=3000, seed=7,
                    p_base_override=0.38)
    r1 = compute_trp(inp1)
    print(f"\n[Scenario 1] Age=36, AMH=1.8, no history")
    print(f"  P(≥1 pregnancy in 6 cycles) = {r1.p_success_total*100:.1f}%")
    print(f"  Median window             = {r1.window_cycles_p50:.0f} cycles")
    print(f"  P(window closes first)    = {r1.p_window_closes_first*100:.1f}%")
    print(f"  E[cycles to success]      = {r1.expected_cycles_to_success:.1f}")
    print(f"  Cumulative P by cycle:")
    for c, p in enumerate(r1.p_cum_by_cycle, 1):
        bar = "█" * int(p * 30)
        print(f"    Cycle {c}: {p*100:5.1f}% {bar}")

    # Scenario 2: same patient, 2 past cycles — poor responder
    past = [
        PastCycle(age_at_cycle=34.5, okk_actual=4, blasts_actual=1,
                  outcome=0, amh_at_cycle=2.2, cycle_index=1),
        PastCycle(age_at_cycle=35.5, okk_actual=3, blasts_actual=0,
                  outcome=0, amh_at_cycle=1.9, cycle_index=2),
    ]
    inp2 = TRPInput(age=36, amh=1.8, afc=10, bmi=23.5,
                    past_cycles=past,
                    max_future_cycles=6, n_trajectories=3000, seed=7,
                    p_base_override=0.38)
    r2 = compute_trp(inp2)
    print(f"\n[Scenario 2] Same patient + 2 poor cycles (RR={r2.response_ratio:.2f})")
    print(f"  P(≥1 pregnancy in 6 cycles) = {r2.p_success_total*100:.1f}%  (conditioned)")
    print(f"  Median window             = {r2.window_cycles_p50:.0f} cycles")
    print(f"  P(window closes first)    = {r2.p_window_closes_first*100:.1f}%")
    print(f"  Cumulative P by cycle:")
    for c, p in enumerate(r2.p_cum_by_cycle, 1):
        bar = "█" * int(p * 30)
        print(f"    Cycle {c}: {p*100:5.1f}% {bar}")

    # Scenario 3: 40-year-old, DOR (AMH 0.4)
    # p_base=0.22 simulates KAT output for DOR patient
    inp3 = TRPInput(age=40, amh=0.4, afc=4, bmi=24.0,
                    max_future_cycles=4, n_trajectories=3000, seed=7,
                    p_base_override=0.22)
    r3 = compute_trp(inp3)
    print(f"\n[Scenario 3] Age=40, AMH=0.4 (DOR)")
    print(f"  P(≥1 pregnancy in 4 cycles) = {r3.p_success_total*100:.1f}%")
    print(f"  Median window             = {r3.window_cycles_p50:.0f} cycles")
    print(f"  P(window closes first)    = {r3.p_window_closes_first*100:.1f}%")
    print(f"  k_median={r3.k_median:.3f}/yr; window_p10={r3.window_cycles_p10:.0f}")

    print("\n" + "=" * 60)
    print("Integration snippet for app.py:")
    print(_INTEGRATION_SNIPPET)


if __name__ == "__main__":
    _demo()
