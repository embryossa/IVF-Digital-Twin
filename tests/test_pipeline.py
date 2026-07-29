# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
IVF Digital Twin v6.2 — Test Suite
Run: python -m pytest tests/ -v
"""
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

# Import the pipeline as a real module. It used to be loaded via
# exec(open(...).read()), which left __file__ undefined inside the module and
# broke collection outright, and only worked when pytest ran from the repo
# root. The path is derived from this file, so cwd no longer matters.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

warnings.filterwarnings("ignore")  # suppress NN-not-found notices

from ivf_digital_twin import (  # noqa: E402
    CLUSTER_INTERPRETATIONS,
    KnownValues,
    PatientInput,
    bayesian_posterior_pregnancy,
    run_pipeline,
    run_pipeline_extended,
    stage1_oocytes,
)

N_SIM = 500   # smaller N for speed during testing

# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def standard_patient():
    return PatientInput(female_age=35, amh=2.5, afc=15, bmi=23.0)

@pytest.fixture
def poor_patient():
    return PatientInput(female_age=42, amh=0.5, afc=6, bmi=25.0)

@pytest.fixture
def high_patient():
    return PatientInput(female_age=28, amh=4.5, afc=28, bmi=22.0)

@pytest.fixture
def young_patient():
    return PatientInput(female_age=21, amh=3.5, afc=20, bmi=22.0)


# ============================================================
# STAGE 1 — OKK DISTRIBUTION
# ============================================================

class TestStage1:
    def test_output_shape(self, standard_patient):
        np.random.seed(42)
        okk = stage1_oocytes(standard_patient, KnownValues(), n=N_SIM)
        assert len(okk) == N_SIM

    def test_known_value_conditioning(self, standard_patient):
        np.random.seed(42)
        known = KnownValues(okk=10)
        okk = stage1_oocytes(standard_patient, known, n=N_SIM)
        assert np.all(okk == 10), "All iterations must equal the observed value"

    def test_support_bounds(self, poor_patient):
        """OKK lies in [0, 50].

        Zero is a legitimate outcome, not a clipping failure: the v6.2 ZINB
        model has a Bernoulli zero-inflation component representing cancelled
        cycles and empty-follicle punctures. The v6.1 truncated Normal clipped
        to [1, 50] and formally forbade them, which is exactly the limitation
        ZINB was introduced to fix.
        """
        np.random.seed(42)
        okk = stage1_oocytes(poor_patient, KnownValues(), n=N_SIM)
        assert np.all(okk >= 0), "Oocyte count cannot be negative"
        assert np.all(okk <= 50), "Oocyte count must be at most 50"

    def test_structural_zeros_occur_for_poor_responder(self, poor_patient):
        """The zero-inflation component must actually fire.

        A poor responder (age 42, AMH 0.5, AFC 6) should produce some
        cancelled cycles. If this ever returns no zeros, the ZINB has silently
        degenerated to a plain Negative Binomial.
        """
        np.random.seed(42)
        okk = stage1_oocytes(poor_patient, KnownValues(), n=N_SIM)
        assert (okk == 0).any(), "zero-inflation component never fired"

    def test_no_structural_zeros_when_okk_observed(self, poor_patient):
        """Conditioning on an observed retrieval overrides zero-inflation."""
        np.random.seed(42)
        okk = stage1_oocytes(poor_patient, KnownValues(okk=8), n=N_SIM)
        assert np.all(okk == 8)

    def test_high_responder_larger_than_poor(self, high_patient, poor_patient):
        np.random.seed(42)
        okk_high = stage1_oocytes(high_patient, KnownValues(), n=N_SIM)
        okk_poor = stage1_oocytes(poor_patient, KnownValues(), n=N_SIM)
        assert np.median(okk_high) > np.median(okk_poor)


# ============================================================
# PIPELINE — ORDERING AND CONSTRAINTS
# ============================================================

class TestPipeline:
    def test_mii_le_okk(self, standard_patient):
        np.random.seed(42)
        res = run_pipeline(standard_patient, KnownValues(), n=N_SIM)
        assert np.all(res["sim_mii"] <= res["sim_okk"]), "MII cannot exceed OKK"

    def test_pn2_le_mii(self, standard_patient):
        np.random.seed(42)
        res = run_pipeline(standard_patient, KnownValues(), n=N_SIM)
        assert np.all(res["sim_pn2"] <= res["sim_mii"]), "2PN cannot exceed MII"

    def test_euploid_le_good(self, standard_patient):
        np.random.seed(42)
        res = run_pipeline(standard_patient, KnownValues(), n=N_SIM)
        assert np.all(res["sim_euploid"] <= res["sim_good"]), "Euploid cannot exceed good"

    def test_warmed_le_euploid(self, standard_patient):
        np.random.seed(42)
        res = run_pipeline(standard_patient, KnownValues(), n=N_SIM)
        assert np.all(res["sim_warmed"] <= res["sim_euploid"]), "Warmed cannot exceed euploid"

    def test_probabilities_in_unit_interval(self, standard_patient):
        np.random.seed(42)
        res = run_pipeline(standard_patient, KnownValues(), n=N_SIM)
        assert 0 <= res["p_per_transfer"] <= 1
        assert 0 <= res["p_cum_if_viable"] <= 1
        assert 0 <= res["p_overall_cycle"] <= 1
        assert 0 <= res["p_viable"] <= 1

    def test_three_level_ordering(self, standard_patient):
        """per_transfer <= cum_if_viable; overall <= cum_if_viable."""
        np.random.seed(42)
        res = run_pipeline(standard_patient, KnownValues(), n=N_SIM)
        assert res["p_per_transfer"] <= res["p_cum_if_viable"] + 0.01
        assert res["p_overall_cycle"] <= res["p_cum_if_viable"] + 0.01

    def test_three_level_ordering_poor_responder(self, poor_patient):
        """Critical: must hold even for poor responders where p_viable << 1."""
        np.random.seed(42)
        res = run_pipeline(poor_patient, KnownValues(), n=N_SIM)
        assert res["p_per_transfer"] <= res["p_cum_if_viable"] + 0.01
        assert res["p_overall_cycle"] <= res["p_cum_if_viable"] + 0.01

    def test_poor_responder_lower_euploid(self, standard_patient, poor_patient):
        np.random.seed(42)
        res_std = run_pipeline(standard_patient, KnownValues(), n=N_SIM)
        res_poor = run_pipeline(poor_patient, KnownValues(), n=N_SIM)
        assert res_std["expected_euploid"] > res_poor["expected_euploid"]


# ============================================================
# BAYESIAN CONDITIONAL UPDATING
# ============================================================

class TestBayesianUpdating:
    def test_known_okk_narrows_variance(self, standard_patient):
        np.random.seed(42)
        res_prior = run_pipeline(standard_patient, KnownValues(), n=N_SIM)
        res_post  = run_pipeline(standard_patient, KnownValues(okk=12), n=N_SIM)
        # Variance in downstream should be narrower after conditioning
        # (2PN variance is the clearest proxy)
        assert np.std(res_post["sim_pn2"]) <= np.std(res_prior["sim_pn2"]) + 1

    def test_known_euploid_bounds_warmed(self, standard_patient):
        """Post-thaw survivors are bounded by the euploid count.

        warmed ~ Binomial(euploid, 0.95), so with euploid = 3 the support is
        {0, 1, 2, 3} and the mean is 2.85. The previous assertion required
        warmed >= 2, which is not an invariant: P(warmed <= 1) = 0.0073, so it
        failed on roughly 4 of 500 iterations. Bounds are asserted exactly;
        the survival rate is asserted distributionally.
        """
        np.random.seed(42)
        res = run_pipeline(standard_patient, KnownValues(euploid=3), n=N_SIM)
        warmed = res["sim_warmed"]
        assert np.all(warmed >= 0)
        assert np.all(warmed <= 3), "cannot thaw more embryos than exist"
        # 95% survival per blastocyst (Coello et al. 2021) → mean 2.85 of 3
        assert 2.7 <= warmed.mean() <= 3.0

    def test_all_stages_known(self, standard_patient):
        """Full conditioning should give deterministic embryo counts."""
        np.random.seed(42)
        kv = KnownValues(okk=12, mii=10, pn2=7, blasts=5, good=4, euploid=3)
        res = run_pipeline(standard_patient, kv, n=N_SIM)
        assert np.all(res["sim_okk"] == 12)
        assert np.all(res["sim_mii"] == 10)
        assert np.all(res["sim_pn2"] == 7)
        assert np.all(res["sim_blasts"] == 5)
        assert np.all(res["sim_good"] == 4)
        assert np.all(res["sim_euploid"] == 3)


# ============================================================
# KPI SCORE
# ============================================================

class TestKPIScore:
    def test_kpi_range(self, standard_patient):
        np.random.seed(42)
        res = run_pipeline(standard_patient, KnownValues(), n=N_SIM)
        scores = res["sim_kpi_scores"]
        assert np.all(scores >= 5)
        assert np.all(scores <= 25)

    def test_high_responder_higher_kpi(self, standard_patient, poor_patient):
        np.random.seed(42)
        res_std  = run_pipeline(standard_patient, KnownValues(), n=N_SIM)
        res_poor = run_pipeline(poor_patient,     KnownValues(), n=N_SIM)
        assert np.median(res_std["sim_kpi_scores"]) > np.median(res_poor["sim_kpi_scores"])


# ============================================================
# BAYESIAN POSTERIOR
# ============================================================

class TestBayesianPosterior:
    def test_posterior_mean_in_unit_interval(self):
        post = bayesian_posterior_pregnancy(0.45)
        assert 0 < post["mean"] < 1

    def test_credible_interval_ordered(self):
        post = bayesian_posterior_pregnancy(0.45)
        assert post["ci_low"] < post["mean"] < post["ci_high"]

    def test_posterior_dampens_optimistic_nn(self):
        """High NN probability should be pulled toward clinic's historical rate."""
        post_high = bayesian_posterior_pregnancy(0.90)
        post_low  = bayesian_posterior_pregnancy(0.10)
        assert post_high["mean"] > post_low["mean"]  # ordering preserved
        assert post_high["mean"] < 0.90               # pulled down from ceiling

    def test_with_real_data(self):
        post = bayesian_posterior_pregnancy(
            0.55,
            real_successes=[19, 18, 20],
            real_trials   =[43, 45, 65],
        )
        assert post["n_real_cycles"] == 3
        assert 0 < post["mean"] < 1


# ============================================================
# CLUSTER CLASSIFIER
# ============================================================

class TestClusterClassifier:
    def _run(self, patient):
        np.random.seed(42)
        res = run_pipeline_extended(
            patient, known=KnownValues(),
            attempt_number=1, max_attempts_curve=2,
        )
        return res["cluster_analysis"]

    def test_probs_sum_to_one(self, standard_patient):
        ca = self._run(standard_patient)
        assert abs(sum(ca["cluster_probs"].values()) - 1.0) < 0.001

    def test_standard_responder_dominant_c0(self, standard_patient):
        ca = self._run(standard_patient)
        assert ca["dominant_cluster"] == 0, "Standard patient should land in C0"

    def test_poor_responder_dominant_c1(self, poor_patient):
        ca = self._run(poor_patient)
        assert ca["dominant_cluster"] == 1, "Poor responder should land in C1"

    def test_high_responder_dominant_c2(self, high_patient):
        ca = self._run(high_patient)
        assert ca["dominant_cluster"] == 2, "High responder should land in C2"

    def test_pca_shape(self, standard_patient):
        ca = self._run(standard_patient)
        n_syn  = ca["n_synthetic"]
        n_total = n_syn + len(ca["patient_features"])
        assert ca["pca_embedded"].shape == (n_total, 2)

    def test_explained_variance_reasonable(self, standard_patient):
        ca = self._run(standard_patient)
        # PCA(2) should explain at least 40% in this low-rank synthetic cloud
        assert ca["pca_explained"].sum() > 0.40


# ============================================================
# RISK METRICS
# ============================================================

class TestRiskMetrics:
    def test_ohss_risk_higher_for_high_responder(self, high_patient, poor_patient):
        np.random.seed(42)
        res_high = run_pipeline(high_patient, KnownValues(), n=N_SIM)
        res_poor = run_pipeline(poor_patient, KnownValues(), n=N_SIM)
        assert res_high["ohss"]["p_any_ohss"] > res_poor["ohss"]["p_any_ohss"]

    def test_empty_cycle_higher_for_poor_responder(self, poor_patient, standard_patient):
        np.random.seed(42)
        res_poor = run_pipeline(poor_patient,     KnownValues(), n=N_SIM)
        res_std  = run_pipeline(standard_patient, KnownValues(), n=N_SIM)
        assert res_poor["empty"]["p_no_blast"] > res_std["empty"]["p_no_blast"]


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python", "-m", "pytest", __file__, "-v"])
