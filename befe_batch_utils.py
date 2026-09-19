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
    if not os.path.exists(path) and not os.path.exists(path + '.enc'):
        return None
    try:
        if os.path.exists(path + '.enc'):
            from crypt_engine import decrypt_model_to_stream
            z = np.load(decrypt_model_to_stream(path + '.enc'), allow_pickle=False)
        else:
            z = np.load(path, allow_pickle=False)
        version=int(z['schema_version']) if 'schema_version' in z else 1
        expected_clin=['age','afc'] if version in (2,3) else ['age','amh','afc','bmi']
        expected_emb=['okk','mii','pn2','blast'] if version in (2,3) else ['okk','mii','pn2','blast','kpi']
        if (version not in (1,2,3) or list(z['clinical_order']) != expected_clin or list(z['embryo_order']) != expected_emb):
            z.close()
            return None
        stats = {
            "clinical_mu": z["clinical_mu"],
            "clinical_cov_inv": z["clinical_cov_inv"],
            "embryo_mu": z["embryo_mu"],
            "embryo_cov_inv": z["embryo_cov_inv"],
        }
        if version in (2,3):
            stats.update(clinical_indices=[0,2],embryo_indices=[0,1,2,3],reference='GAT training cohort',unassessed=['amh','bmi','kpi'])
            for name in ('clinical','embryo'):
                threshold=float(z[name+'_threshold'])
                if not np.isfinite(threshold) or threshold<=0:z.close();return None
                stats[name+'_threshold']=threshold
        if version==3:
            mu=np.asarray(z['embryo_shrink_mu']);kappa=np.asarray(z['embryo_shrink_kappa'])
            if list(z['embryo_transform'])!=['log1p','eb_logit','eb_logit','eb_logit'] or mu.shape!=(3,) or kappa.shape!=(3,) or not np.isfinite(mu).all() or not np.isfinite(kappa).all() or (mu<=0).any() or (mu>=1).any() or (kappa<=0).any():
                z.close();return None
            stats.update(embryo_transform='v3',embryo_shrink_mu=mu,embryo_shrink_kappa=kappa,reference='GAT/KAT training cohort (protocols_15k), culture-observed')
        z.close()
    except Exception:
        return None
    return stats if _ood_stats_usable(stats) else None


def _ood_stats_usable(stats) -> bool:
    """Reject a fitted OOD file whose subspace has a collapsed dimension.

    Such a file marks every patient as out-of-distribution, which deflates the
    empirical-evidence precision inside BEFE to ~0 and turns L7 into a
    pass-through of the L1/L2 prior. Disabling OOD is the documented fallback
    (no crash, no reliability cap), so degrade to that instead.
    """
    import logging
    from befe import degenerate_ood_dimensions
    for name in ("clinical", "embryo"):
        bad = degenerate_ood_dimensions(stats[name + "_mu"], stats[name + "_cov_inv"])
        if bad:
            logging.error(
                "BEFE OOD disabled: %s subspace dimension(s) %s have no variation "
                "in the fitted cohort; refit with fit_befe_ood.py.", name, bad)
            return False
    return True


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

    # The final probability is per transfer: fuse the view restricted to
    # scenarios with a transfer when the core provides it.
    res = result.get("res_transfer") or result.get("res") or {}
    nn_pred = res.get("nn_prediction", {}) if isinstance(res, dict) else {}
    gnn_result = result.get("gnn_result") or {}

    p_kat_raw = result.get("p_kat_raw")
    p_gnn_raw = result.get("p_gnn_raw")
    ci_kat = nn_pred.get("base_prob_ci", (None, None)) if isinstance(nn_pred, dict) else (None, None)
    w_gnn = _f(gnn_result.get("w_gnn") if isinstance(gnn_result, dict) else None, 0.35)
    tau_kat_dyn = None

    # Clinic 7.1 uses only the fingerprint-bound final intercept in desktop.calibration.
    # Research layer-temperature / dynamic-tau adaptations are not clinical inputs.
    try:
        ood_stats=load_befe_ood_stats(base_dir)
        befe_res, befe_map = build_befe_result(
            res,
            p_kat_raw=p_kat_raw,
            ci_kat=ci_kat,
            p_gnn_raw=p_gnn_raw,
            gnn_result=gnn_result,
            w_gnn=w_gnn,
            csdi_result=result.get("csdi_result"),
            csdi_applicability=result.get("csdi_applicability"),
            age=float(age),
            amh=float(amh),
            afc=int(afc),
            bmi=float(bmi),
            ood_stats=ood_stats,
            tau_kat_override=tau_kat_dyn,
        )
        befe_map['ood_available']=ood_stats is not None
        befe_map['ood_reference']=ood_stats.get('reference','training cohort') if ood_stats else None
        befe_map['ood_unassessed']=ood_stats.get('unassessed',[]) if ood_stats else []
    except Exception:
        return None, None, {}

    posterior = _f(getattr(befe_res, "posterior", None))
    if posterior is None:
        return None, befe_res, befe_map or {}
    return posterior, befe_res, befe_map or {}


@lru_cache(maxsize=8)
def load_csdi_ood_stats(base_dir):
    from befe import degenerate_ood_dimensions
    path=os.path.join(base_dir,'models','csdi_ood_stats.npz')
    try:
        if os.path.exists(path+'.enc'):
            from crypt_engine import decrypt_model_to_stream
            source=decrypt_model_to_stream(path+'.enc')
        else:source=path
        with np.load(source,allow_pickle=False) as z:
            if int(z['schema_version'])!=1 or list(z['order'])!=['follicles','okk','mii','pn2','kpi'] or list(z['transform'])!=['log1p']*4+['identity'] or int(z['min_pn2'])!=1:return None
            stats={k:z[k].copy() for k in ('mu','cov_inv')};stats['threshold']=float(z['threshold'])
            if stats['mu'].shape!=(5,) or stats['cov_inv'].shape!=(5,5) or degenerate_ood_dimensions(stats['mu'],stats['cov_inv']) or not np.isfinite(stats['threshold']) or stats['threshold']<=0:return None
            return stats
    except (OSError,ValueError,KeyError,TypeError):return None


def assess_csdi(patient,base_dir):
    stats=load_csdi_ood_stats(base_dir)
    values=np.array([patient[k] for k in ('Количество фолликулов','Число ОКК','Число инсеминированных','2 pN','KPIScore')],dtype=float)
    valid=np.isfinite(values).all() and min(values)>=0 and values[3]>=1
    ratio=None
    if valid and stats:
        x=np.r_[np.log1p(values[:4]),values[4]];d=x-stats['mu']
        ratio=float(d@stats['cov_inv']@d/stats['threshold'])
    ood=bool(ratio is not None and ratio>1)
    # Fail closed: without the fitted applicability statistics CSDI cannot be
    # shown to lie inside its training data, so it does not enter the fusion.
    reason='no_2pn' if not valid else 'unchecked' if stats is None else 'outside_training' if ood else 'in_support'
    return {'available':stats is not None,'in_support':bool(valid),'ratio':ratio,'ood':ood,
            'used_in_fusion':bool(valid and stats is not None and not ood),'reason':reason,
            'reference':'CSDI training cohort (all_df_with_KPI)'}
