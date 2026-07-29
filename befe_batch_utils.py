# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
Headless BEFE (L7) helpers for batch scripts.

Keeps batch output aligned with the Streamlit app: the headline prediction is
BEFE.posterior when L7 is available, with graceful fallback to p_overall_cycle.
"""

from __future__ import annotations

import glob
import json
import os
from functools import lru_cache

import numpy as np


def _f(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


@lru_cache(maxsize=8)
def load_befe_ood_stats(base_dir: str):
    """Load optional BEFE OOD stats created by fit_befe_ood.py."""
    path = os.path.join(base_dir, "models", "befe_ood_stats.npz")
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=True)
        return {
            "clinical_mu": z["clinical_mu"],
            "clinical_cov_inv": z["clinical_cov_inv"],
            "embryo_mu": z["embryo_mu"],
            "embryo_cov_inv": z["embryo_cov_inv"],
        }
    except Exception:
        return None


@lru_cache(maxsize=8)
def load_clinic_adaptation(base_dir: str):
    """Load the latest clinic adaptation JSON, matching app.py startup logic."""
    files = sorted(glob.glob(os.path.join(base_dir, "models", "clinic_adaptation_*.json")))
    if not files:
        return None
    try:
        with open(files[-1], encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def compute_l7_posterior(result: dict, *, age, amh, afc, bmi, base_dir: str):
    """
    Return (posterior, befe_result, mapping). posterior is None if BEFE is absent
    or cannot be computed for this row.
    """
    if not isinstance(result, dict):
        return None, None, {}

    try:
        from befe_app import build_befe_result
    except Exception:
        return None, None, {}

    res = result.get("res") or {}
    nn_pred = res.get("nn_prediction", {}) if isinstance(res, dict) else {}
    gnn_result = result.get("gnn_result") or {}

    p_kat_raw = result.get("p_kat_raw")
    p_gnn_raw = result.get("p_gnn_raw")
    ci_kat = nn_pred.get("base_prob_ci", (None, None)) if isinstance(nn_pred, dict) else (None, None)
    w_gnn = _f(gnn_result.get("w_gnn") if isinstance(gnn_result, dict) else None, 0.35)
    tau_kat_dyn = None

    adapt = load_clinic_adaptation(base_dir)
    if adapt:
        try:
            from calibrate_for_clinic import apply_clinic_calibration, compute_dynamic_tau_kat

            temp = adapt.get("temperature", {})
            if p_kat_raw is not None and abs(temp.get("T_kat", 1.0) - 1.0) > 0.01:
                p_kat_raw = apply_clinic_calibration(p_kat_raw, "T_kat", adapt)
            if p_gnn_raw is not None and abs(temp.get("T_gat", 1.0) - 1.0) > 0.01:
                p_gnn_raw = apply_clinic_calibration(p_gnn_raw, "T_gat", adapt)

            features = {
                "age": float(age),
                "amh": float(amh),
                "afc": float(afc),
                "bmi": float(bmi),
                "okk": float(res.get("okk_med", 0)),
                "mii": float(res.get("mii_med", 0)),
                "pn2": float(res.get("pn2_med", 0)),
                "blasts_total": float(res.get("blasts_med", 0)),
                "blasts_good": float(res.get("good_med", 0)),
            }
            tau_kat_dyn = compute_dynamic_tau_kat(features, adapt)
        except Exception:
            tau_kat_dyn = None

    try:
        befe_res, befe_map = build_befe_result(
            res,
            p_kat_raw=p_kat_raw,
            ci_kat=ci_kat,
            p_gnn_raw=p_gnn_raw,
            gnn_result=gnn_result,
            w_gnn=w_gnn,
            csdi_result=result.get("csdi_result"),
            age=float(age),
            amh=float(amh),
            afc=int(afc),
            bmi=float(bmi),
            ood_stats=load_befe_ood_stats(base_dir),
            tau_kat_override=tau_kat_dyn,
        )
    except Exception:
        return None, None, {}

    posterior = _f(getattr(befe_res, "posterior", None))
    if posterior is None:
        return None, befe_res, befe_map or {}
    return posterior, befe_res, befe_map or {}
