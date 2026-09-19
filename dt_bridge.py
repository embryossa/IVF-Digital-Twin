# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""Data flow of IVF Digital Twin 7.1 for the Streamlit interface (app.py).

The 7.1 clinic build computes one case in a fixed order, and every output
(screen, PDF, history, batch) reads the same result:

  1. ivf_core.predict_single_patient      L1-L6: pipeline, KAT, cluster, CSDI
                                          (with applicability check), GAT, Esteves
  2. befe_batch_utils.compute_l7_posterior L7 BEFE on the transfer view
  3. presentation.clinical_summary        headline contract (per transfer, or
                                          0 for the current cycle when no
                                          transfer is possible)
  4. embryology.anchored_cycle_probability cycle probability consistent with
                                          the final per-transfer probability
  5. trp_anchor                           starting probability of the TRP plan

This module reproduces that order without the desktop licence, storage and
HTTP layers, so app.py no longer assembles the layers itself. Centre
calibration of the clinic build is not part of this repository.
"""
import os

import numpy as np

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Results of the current cycle; a plan of new cycles must not inherit them.
CYCLE_OBSERVATIONS = ('known_okk', 'known_mii', 'known_pn2', 'known_blasts',
                      'known_good', 'known_euploid', 'follicles')

DEFAULT_RELIABILITY = {'moderate': 45, 'high': 70}


def reliability_band(score, high, moderate):
    return 'High' if score >= high else 'Moderate' if score >= moderate else 'Low'


def compute(patient, clinic_successes=None, clinic_trials=None, reliability=None):
    """Full L1-L7 calculation of one case.

    patient: keyword arguments of ivf_core.predict_single_patient
    (age, amh, afc, bmi, attempt, follicles, sperm_source, known_*, n_sim, seed).
    Returns the core result extended with the L7 fusion, the clinical
    headline and the cycle probability.
    """
    from ivf_core import predict_single_patient
    from befe_batch_utils import compute_l7_posterior
    from presentation import clinical_summary
    from embryology import anchored_cycle_probability

    patient = dict(patient)
    raw = predict_single_patient(**patient, clinic_successes=clinic_successes or None,
                                 clinic_trials=clinic_trials or None)
    probability, fusion, mapping = compute_l7_posterior(
        raw, age=patient['age'], amh=patient['amh'], afc=patient['afc'],
        bmi=patient['bmi'], base_dir=_BASE_DIR)
    if fusion is not None:
        # One set of reliability thresholds for every output of this case.
        limits = reliability or DEFAULT_RELIABILITY
        fusion.reliability_band = reliability_band(fusion.reliability, limits['high'],
                                                   limits['moderate'])
    res = raw['res']
    summary = clinical_summary({'patient': patient, 'headline': probability})
    cycle_probability = (0.0 if summary['no_transfer_confirmed'] else anchored_cycle_probability(
        res['sim_p_combined'], res['sim_n_tx_good'], res['sim_n_tx_fair'], summary['probability']))
    return {**raw,
            'patient': patient,
            'clinic_successes': clinic_successes,
            'clinic_trials': clinic_trials,
            'headline': probability,
            'fusion': fusion,
            'fusion_mapping': mapping or {},
            'clinical_summary': summary,
            'cycle_probability': cycle_probability,
            'per_transfer': res.get('p_per_transfer_if_transfer'),
            'p_cancel_risk': float(np.mean(np.asarray(res['sim_okk']) == 0)) if len(res.get('sim_okk', [])) else None}


def trp_anchor(result):
    """Starting per-cycle probability for the TRP plan, from the full stack.

    TRP extrapolates one absolute level over time, so the anchor is the
    per-cycle probability shown for the case (BEFE/L7 carried from the transfer
    to the cycle). Only a cycle whose transfer is confirmed impossible is
    replaced: its zero must not zero every planned cycle, so the anchor is
    recomputed for a fresh cycle without the current observations.
    Returns (probability, source); the value is cached in the result.
    """
    current = result.get('cycle_probability')
    if current:
        return float(current), 'current-L1-L7'
    if result.get('trp_anchor') is None:
        fresh = {**result['patient'], **{k: None for k in CYCLE_OBSERVATIONS}}
        prospective = compute(fresh, result.get('clinic_successes'), result.get('clinic_trials'))
        value = prospective.get('cycle_probability')
        # Without L7 the mechanistic cycle value is the only anchor left.
        result['trp_anchor'] = ((float(value), 'prospective-L1-L7') if value else
                                (float(prospective['res']['p_overall_cycle']), 'prospective-L1-L4'))
    return result['trp_anchor']
