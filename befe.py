"""
BEFE — Bayesian Evidence Fusion Engine  (Layer 7 of the IVF Digital Twin)
=========================================================================

L7 turns the upstream layers into a formal  Prior -> Evidence -> Posterior
update, performed as conjugate Gaussian pooling in logit space.

    Mechanistic prior      L1  (Monte-Carlo embryological cascade)
    Empirical evidence     L2  (FORTUNE + KPI ensemble)
                           L3  (KAT: DNN-KAN + FT-Transformer)
                           L6  (GAT graph transformer)
    Verification           L5  (CSDI diffusion; KS vs MC) -> modulates the PRIOR
    Context                L4  (cluster phenotype), graph neighbourhood (L6)

Two-level fusion
----------------
Level 1 — Evidence fusion (empirical models only):
    l_emp  = sum(tau_i * logit(p_i)) / sum(tau_i)        for i in {L2, KAT, GAT}
    tau_emp = sum(tau_i)                                  ("strength of evidence")
    P_predictive = sigmoid(l_emp)

Level 2 — Posterior update (prior x evidence):
    l_post = (tau_prior * l_prior + tau_emp * l_emp) / (tau_prior + tau_emp)
    var_post = 1 / (tau_prior + tau_emp)
    Posterior = sigmoid(l_post)

Because precision adds, weak evidence (low tau_emp) makes the posterior shrink
toward the mechanistic prior — the desired clinical fallback. The relative pull
(tau_prior vs tau_emp) is reported so the clinician sees how much of the answer
came from mechanism vs data.

Dual OOD
--------
OOD_clinical    (Age, AMH, AFC, BMI)        -> deflates EVIDENCE precision
OOD_embryology  (OCC, MII, 2PN, Blast, KPI) -> deflates PRIOR precision
OOD_final = max(OOD_clinical, OOD_embryology) -> alert flag + reliability cap

This separation lets the report state e.g. "clinically typical, lab-unusual".

NOTE: every constant below is a PRIOR and must be calibrated on held-out data.
After fusion, re-check final-layer calibration (ECE) — pooling calibrated
experts does not guarantee a calibrated posterior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
#  Tunable priors  (CALIBRATE before deployment)
# --------------------------------------------------------------------------- #

TAU_PRIOR_BASE = 1.0                      # base precision of the L1 mechanistic prior
TAU_BASE_EVIDENCE = {                     # base precision of each empirical expert
    "KAT": 2.4,   # L3 neural ensemble — best calibrated to our cohort (low ECE)
    "GAT": 1.0,   # L6 graph transformer — useful context, calibrated via ensemble
}

RELIABILITY_WEIGHTS = {                   # must sum to 1.0
    "consensus": 0.40,
    "diffusion": 0.30,
    "graph": 0.20,
    "cluster": 0.10,
}

OOD_ALPHA = 0.01                          # chi-square tail for the OOD threshold
OOD_DEFLATE_K = 1.5                       # how hard OOD shrinks precision
_EPS = 1e-6


# --------------------------------------------------------------------------- #
#  Numeric helpers
# --------------------------------------------------------------------------- #

def _clip_p(p: float) -> float:
    return float(min(max(p, _EPS), 1.0 - _EPS))

def logit(p: float) -> float:
    p = _clip_p(p)
    return math.log(p / (1.0 - p))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------- #
#  Graph confidence primitives (from L6 attention)
# --------------------------------------------------------------------------- #

def effective_neighbour_count(attention: Sequence[float]) -> float:
    """N_eff = 1 / sum(a_i^2): inverse-Simpson participation ratio of attention."""
    a = np.asarray(attention, dtype=float)
    s = a.sum()
    if s <= 0:
        return 0.0
    a = a / s
    return float(1.0 / np.sum(a ** 2))

def attention_entropy(attention: Sequence[float], normalised: bool = True) -> float:
    """Normalised Shannon entropy in [0,1]; 1 = broad support, 0 = collapsed."""
    a = np.asarray(attention, dtype=float)
    s = a.sum()
    if s <= 0 or a.size < 2:
        return 0.0
    a = np.clip(a / s, _EPS, 1.0)
    h = -np.sum(a * np.log(a))
    return float(h / math.log(a.size)) if normalised else float(h)

def neighbour_variance(neighbour_probs: Sequence[float]) -> float:
    n = np.asarray(neighbour_probs, dtype=float)
    return float(np.var(n)) if n.size else 0.25


# --------------------------------------------------------------------------- #
#  Input containers
# --------------------------------------------------------------------------- #

@dataclass
class ExpertOutputs:
    """Mechanistic prior + empirical predictors."""
    P_L1: float                 # mechanistic prior = MC + FORTUNE/KPI per-transfer
    P_KAT: float                # empirical evidence: L3 KAT neural ensemble
    P_GAT: float                # empirical evidence: L6 graph transformer

@dataclass
class UncertaintyContext:
    CI_L1_width: float = 0.30          # width of the L1 prior CI (prob units)
    posterior_variance: float = 0.02   # L3 Bayesian posterior variance
    MC_variance: float = 0.02          # KAT MC-dropout variance

@dataclass
class DiffusionContext:
    """L5 CSDI verification; lower KS = better MC/CSDI agreement."""
    KS_pregnancy: float = 0.0
    KS_blast: float = 0.0
    KS_TGBDR: float = 0.0
    available: bool = True
    @property
    def agreement_score(self) -> float:
        mean_ks = float(np.mean([self.KS_pregnancy, self.KS_blast, self.KS_TGBDR]))
        return float(min(max(1.0 - mean_ks, 0.0), 1.0))

@dataclass
class ClusterContext:
    cluster_id: int
    cluster_label: str = ""
    distance_to_centroid: float = 1.0
    cluster_probability: float = 1.0
    typical_distance: float = 1.0

@dataclass
class GraphContext:
    attention: Sequence[float] = field(default_factory=list)
    neighbour_probs: Sequence[float] = field(default_factory=list)
    available: bool = True
    @property
    def n_eff(self) -> float:        return effective_neighbour_count(self.attention)
    @property
    def entropy(self) -> float:      return attention_entropy(self.attention)
    @property
    def var_neighbors(self) -> float: return neighbour_variance(self.neighbour_probs)


# --------------------------------------------------------------------------- #
#  Dual OOD (Mahalanobis on clinical and embryological subspaces)
# --------------------------------------------------------------------------- #

def fit_gaussian(train_matrix: np.ndarray, reg: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Fit mean and (regularised) inverse covariance for one feature subspace.
    Use OFFLINE on the training cohort; store the returned arrays.
    """
    X = np.asarray(train_matrix, dtype=float)
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov = np.atleast_2d(cov) + reg * np.eye(X.shape[1])
    return mu, np.linalg.pinv(cov)

def _ood_threshold(df: int) -> float:
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(1.0 - OOD_ALPHA, df))
    except Exception:
        return float(df + 3.0 * math.sqrt(2.0 * df))

@dataclass
class _Subspace:
    """One OOD subspace: the patient vector + the fitted training stats."""
    features: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None
    cov_inv: Optional[np.ndarray] = None

    def score(self) -> Tuple[bool, float, float]:
        """Return (is_ood, ratio, distance). ratio = d^2 / threshold; >1 => OOD."""
        if self.features is None or self.mean is None or self.cov_inv is None:
            return False, 0.0, 0.0
        diff = np.asarray(self.features, float) - np.asarray(self.mean, float)
        d2 = float(diff.T @ self.cov_inv @ diff)
        ratio = d2 / max(_ood_threshold(diff.shape[0]), _EPS)
        return ratio > 1.0, ratio, math.sqrt(max(d2, 0.0))

@dataclass
class OODContext:
    clinical: _Subspace = field(default_factory=_Subspace)       # Age, AMH, AFC, BMI
    embryology: _Subspace = field(default_factory=_Subspace)     # OCC, MII, 2PN, Blast, KPI

def _deflate(ratio: float) -> float:
    """Map an OOD ratio (1 = at threshold) to a precision multiplier in (0,1]."""
    excess = max(ratio - 1.0, 0.0)
    return 1.0 / (1.0 + OOD_DEFLATE_K * excess)


# --------------------------------------------------------------------------- #
#  Trust functions  (confidence -> multiplier in (0,1])
# --------------------------------------------------------------------------- #

def _trust_from_ci_width(width: float, scale: float = 4.0) -> float:
    return math.exp(-scale * max(width, 0.0))

def _trust_from_variance(var: float, scale: float = 8.0) -> float:
    return 1.0 / (1.0 + scale * max(var, 0.0))

def _graph_trust(g: GraphContext, target_neighbours: float = 10.0) -> float:
    s_neff = 1.0 - math.exp(-g.n_eff / max(target_neighbours, _EPS))
    s_var = 1.0 - min(g.var_neighbors / 0.25, 1.0)
    s_ent = g.entropy
    return float((max(s_neff, _EPS) * max(s_var, _EPS) * max(s_ent, _EPS)) ** (1 / 3))

def _cluster_certainty(c: ClusterContext) -> float:
    closeness = math.exp(-c.distance_to_centroid / max(c.typical_distance, _EPS))
    return float(min(max(0.5 * closeness + 0.5 * c.cluster_probability, 0.0), 1.0))


# --------------------------------------------------------------------------- #
#  Result container
# --------------------------------------------------------------------------- #

@dataclass
class BEFEResult:
    posterior: float                 # final pregnancy probability
    ci_low: float
    ci_high: float
    ci_source: str                   # "beta-posterior" | "logit-pool"
    p_predictive: float              # Level-1 empirical evidence (L3/L6)
    p_prior: float                   # L1 mechanistic prior
    prior_pull: float                # share of the posterior from the prior (0..1)
    evidence_pull: float             # share from empirical evidence (0..1)
    reliability: int                 # 0..100
    reliability_band: str
    consensus: str
    n_eff: float
    diffusion_agreement: float
    diffusion_available: bool
    graph_available: bool
    cluster_label: str
    ood_clinical: bool
    ood_embryology: bool
    ood_final: bool
    ood_note: str
    evidence_weights: dict           # normalised contribution within Level 1
    raw_experts: dict


# --------------------------------------------------------------------------- #
#  The engine
# --------------------------------------------------------------------------- #

class BayesianEvidenceFusionEngine:
    """BEFE — two-level (Prior -> Evidence -> Posterior) trust-weighted fusion."""

    def __init__(self, tau_prior_base: float = TAU_PRIOR_BASE,
                 tau_base_evidence: Optional[dict] = None):
        self.tau_prior_base = tau_prior_base
        self.tau_base_evidence = dict(tau_base_evidence or TAU_BASE_EVIDENCE)

    # -- per-expert trust -------------------------------------------------- #
    def _evidence_trusts(self, unc: UncertaintyContext, graph: GraphContext) -> dict:
        return {
            "KAT": _trust_from_variance(unc.posterior_variance + unc.MC_variance),
            "GAT": _graph_trust(graph),
        }

    # -- reliability ------------------------------------------------------- #
    @staticmethod
    def _consensus_score(active_probs) -> float:
        """Spread among the ACTIVE empirical predictors. Tight -> ~1.
        With <2 active experts there is no disagreement to measure -> 1.0."""
        p = np.asarray(list(active_probs), dtype=float)
        if p.size < 2:
            return 1.0
        return float(1.0 - min(np.std(p) / 0.5, 1.0))

    def _reliability(self, cons_val, diff_eff, graph_eff, cluster) -> float:
        w = RELIABILITY_WEIGHTS
        return 100.0 * (
            w["consensus"] * cons_val
            + w["diffusion"] * diff_eff
            + w["graph"] * graph_eff
            + w["cluster"] * _cluster_certainty(cluster)
        )

    @staticmethod
    def _ood_note(c: bool, em: bool) -> str:
        if not c and not em:
            return "No"
        if c and not em:
            return "Clinically atypical, embryologically typical"
        if em and not c:
            return "Clinically typical, embryologically atypical"
        return "Atypical in both clinical and embryological space"

    # -- main fusion ------------------------------------------------------- #
    def predict(
        self,
        experts: ExpertOutputs,
        uncertainty: UncertaintyContext,
        diffusion: DiffusionContext,
        cluster: ClusterContext,
        graph: GraphContext,
        ood: Optional[OODContext] = None,
    ) -> BEFEResult:
        ood = ood or OODContext()
        ood_c, ratio_c, _ = ood.clinical.score()
        ood_e, ratio_e, _ = ood.embryology.score()
        ratio_final = max(ratio_c, ratio_e)
        ood_final = ratio_final > 1.0

        clinical_mult = _deflate(ratio_c)     # hits the empirical evidence
        embryo_mult = _deflate(ratio_e)       # hits the mechanistic prior

        # ---- effective verification signals (neutral 0.5 if not run) ---- #
        diff_eff = diffusion.agreement_score if diffusion.available else 0.5
        graph_eff = _graph_trust(graph) if graph.available else 0.5

        # ---- PRIOR precision (L1), verified by L5 diffusion -------------- #
        diff_mult = 0.5 + diff_eff                           # in [0.5, 1.5]
        tau_prior = (self.tau_prior_base
                     * _trust_from_ci_width(uncertainty.CI_L1_width)
                     * diff_mult
                     * embryo_mult)
        tau_prior = max(tau_prior, _EPS)
        l_prior = logit(experts.P_L1)

        # ---- LEVEL 1: empirical evidence fusion (KAT, GAT) -------------- #
        trusts = self._evidence_trusts(uncertainty, graph)
        p_emp = {"KAT": experts.P_KAT, "GAT": experts.P_GAT}
        taus = {k: max(self.tau_base_evidence[k] * trusts[k], _EPS) for k in p_emp}
        tau_emp_raw = sum(taus.values())
        active = [p_emp[k] for k in p_emp if taus[k] > 1e-3]   # models actually present
        if tau_emp_raw <= 1e-3:
            l_emp, tau_emp_raw = l_prior, _EPS                 # no evidence -> prior only
        else:
            l_emp = sum(taus[k] * logit(p_emp[k]) for k in p_emp) / tau_emp_raw
        p_predictive = sigmoid(l_emp)
        tau_emp = tau_emp_raw * clinical_mult     # clinical OOD weakens the evidence

        # ---- LEVEL 2: posterior update ---------------------------------- #
        tau_post = tau_prior + tau_emp
        l_post = (tau_prior * l_prior + tau_emp * l_emp) / tau_post
        sd_post = math.sqrt(1.0 / tau_post)
        posterior = sigmoid(l_post)
        ci_low = sigmoid(l_post - 1.96 * sd_post)
        ci_high = sigmoid(l_post + 1.96 * sd_post)

        # ---- reliability + bands ---------------------------------------- #
        cons_val = self._consensus_score(active if active else [experts.P_KAT])
        reliability = self._reliability(cons_val, diff_eff, graph_eff, cluster)
        if ood_final:
            # BUG FIX (was: reliability = min(reliability, 49.0)).
            # The hard cap at 49 produced a CONSTANT score for every OOD patient,
            # masking genuine variation in model agreement. Replaced with a
            # proportional soft penalty keyed on ratio_final:
            #   ratio = 1.0  → no penalty (patient right at the OOD boundary)
            #   ratio = 2.0  → ~18 % reduction
            #   ratio ≥ 3.0  → ~35 % reduction (floor: 65 % of raw score)
            # The OOD alert is still raised in the result (ood_final = True) and
            # shown prominently in the UI — the reliability score now reflects
            # BOTH actual model agreement AND OOD severity simultaneously.
            ood_factor = max(1.0 - 0.175 * min(ratio_final - 1.0, 2.0), 0.65)
            reliability = reliability * ood_factor
        reliability = int(round(min(max(reliability, 0.0), 100.0)))
        # [CALIB] Label boundaries relaxed so the continuous scores (shown as-is
        # in the UI) map onto realistic bands instead of reading "Low" almost
        # always. The numeric reliability / cons_val are NOT changed — only the
        # category cut-points. OOD detection and the raw scores are untouched.
        rel_band = ("High" if reliability >= 70
                    else "Moderate" if reliability >= 45 else "Low")
        consensus = ("High" if cons_val >= 0.75
                     else "Moderate" if cons_val >= 0.50 else "Low")

        _tau_sum = sum(taus.values())
        ev_weights = {k: round(v / _tau_sum, 3) for k, v in taus.items()}

        return BEFEResult(
            posterior=posterior, ci_low=ci_low, ci_high=ci_high,
            ci_source="logit-pool",
            p_predictive=p_predictive, p_prior=experts.P_L1,
            prior_pull=tau_prior / tau_post, evidence_pull=tau_emp / tau_post,
            reliability=reliability, reliability_band=rel_band, consensus=consensus,
            n_eff=(graph.n_eff if graph.available else float('nan')),
            diffusion_agreement=diffusion.agreement_score,
            diffusion_available=diffusion.available, graph_available=graph.available,
            cluster_label=cluster.cluster_label or f"cluster {cluster.cluster_id}",
            ood_clinical=ood_c, ood_embryology=ood_e, ood_final=ood_final,
            ood_note=self._ood_note(ood_c, ood_e),
            evidence_weights=ev_weights, raw_experts={"P_L1": experts.P_L1, **p_emp},
        )

    # -- physician-facing report ------------------------------------------ #
    @staticmethod
    def format_report(r: BEFEResult) -> str:
        def pct(x): return f"{round(100 * x)}%"
        if not r.diffusion_available:
            diff_word = "n/a (not run)"
        else:
            diff_word = ("Excellent" if r.diffusion_agreement >= 0.9
                         else "Good" if r.diffusion_agreement >= 0.7 else "Weak")
        if not r.graph_available:
            sim_word, neff_txt = "n/a (not run)", "-"
        else:
            sim_word = ("Strong" if r.n_eff >= 20
                        else "Moderate" if r.n_eff >= 8 else "Limited")
            neff_txt = f"{r.n_eff:.0f}"
        lines = [
            "=" * 52,
            "BEFE — BAYESIAN EVIDENCE FUSION  (Digital Twin L7)",
            "=" * 52, "",
            f"Mechanistic prior  (L1):        {pct(r.p_prior)}",
            f"Empirical evidence (L3/L6):     {pct(r.p_predictive)}   [P_predictive]",
            "-" * 52,
            f"Posterior probability:          {pct(r.posterior)}",
            f"95% CI:                         {pct(r.ci_low)}-{pct(r.ci_high)}",
            f"Fusion pull:                    evidence {round(100*r.evidence_pull)}%"
            f" / prior {round(100*r.prior_pull)}%",
            "",
            f"Reliability:                    {r.reliability}/100  ({r.reliability_band})",
            f"Consensus (empirical):          {r.consensus}",
            f"Patient similarity:             {sim_word} (N_eff = {neff_txt})",
            f"Diffusion agreement (L5):       {diff_word}",
            f"Cluster:                        {r.cluster_label}",
            "",
            f"OOD clinical:                   {'YES' if r.ood_clinical else 'No'}",
            f"OOD embryology:                 {'YES' if r.ood_embryology else 'No'}",
            f"OOD final (max):                {'YES — ' + r.ood_note if r.ood_final else 'No'}",
        ]
        lines.append("=" * 52)
        return "\n".join(lines)


BEFE = BayesianEvidenceFusionEngine   # alias


# --------------------------------------------------------------------------- #
#  Demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Offline: fit the two OOD subspaces on a training cohort -------------- #
    clin_train = rng.normal([34, 2.0, 12, 24], [4, 0.8, 4, 3], size=(1500, 4))  # Age,AMH,AFC,BMI
    embryo_train = rng.normal([10, 8, 6, 3, 70], [3, 2.5, 2, 1.5, 12],
                              size=(1500, 5))                                    # OCC,MII,2PN,Blast,KPI
    clin_mu, clin_ci = fit_gaussian(clin_train)
    emb_mu, emb_ci = fit_gaussian(embryo_train)

    engine = BEFE()

    # Case: clinically typical, embryologically unusual ------------------- #
    res = engine.predict(
        experts=ExpertOutputs(P_L1=0.38, P_KAT=0.47, P_GAT=0.45),
        uncertainty=UncertaintyContext(CI_L1_width=0.15,
                                       posterior_variance=0.012, MC_variance=0.010),
        diffusion=DiffusionContext(KS_pregnancy=0.05, KS_blast=0.06, KS_TGBDR=0.05),
        cluster=ClusterContext(cluster_id=2, cluster_label="High responder phenotype",
                               distance_to_centroid=0.7, cluster_probability=0.85),
        graph=GraphContext(
            attention=[0.05]*20, neighbour_probs=[0.44, 0.46, 0.43, 0.45, 0.44]),
        ood=OODContext(
            clinical=_Subspace(features=[33, 2.1, 13, 23], mean=clin_mu, cov_inv=clin_ci),
            embryology=_Subspace(features=[3, 1, 1, 0, 22],  # very atypical lab profile
                                 mean=emb_mu, cov_inv=emb_ci)),
    )
    print(engine.format_report(res))
    print("\nLevel-1 evidence weights:", res.evidence_weights)
