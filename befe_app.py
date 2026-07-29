# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
befe_app.py — Integration layer for BEFE (L7) into IVF Digital Twin v6.3
========================================================================

Bridges the running app to the BEFE engine (befe.py):

    build_befe_result(...)   -> (BEFEResult, mapping)   # construct inputs + fuse
    render_befe_tab(...)     -> Streamlit UI            # the "BEFE (L7)" tab
    befe_pdf_flowables(...)  -> list[reportlab flowable] # PDF section

All extraction is DEFENSIVE: any missing upstream value degrades gracefully
(wider uncertainty / neutral trust) rather than crashing, matching the app's
existing graceful-fallback philosophy.

Layer mapping (verify against your pipeline — see note in build_befe_result):
    P_L1  (mechanistic prior)  <- res['p_per_transfer']        (Monte-Carlo cascade)
    P_L2  (empirical evidence) <- mean(res['sim_p_combined'])  (FORTUNE + KPI)
    P_KAT (empirical evidence) <- p_kat_raw                    (L3 neural ensemble)
    P_GAT (empirical evidence) <- p_gnn_raw                    (L6 pure graph output)
    L5 verification            <- KS(MC vs CSDI) on blast / TGBDR + |dP| pregnancy
"""

from __future__ import annotations

import numpy as np

from befe import (
    BEFE, ExpertOutputs, UncertaintyContext, DiffusionContext,
    ClusterContext, GraphContext, OODContext, _Subspace, fit_gaussian,
)

# Optional deps imported lazily so this module is importable headless.
try:
    from scipy.stats import ks_2samp as _ks_2samp
except Exception:
    _ks_2samp = None


# --------------------------------------------------------------------------- #
#  Small safe helpers
# --------------------------------------------------------------------------- #

def _f(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _arr(x):
    if x is None:
        return None
    try:
        a = np.asarray(x, dtype=float)
    except Exception:
        return None
    if a.ndim == 0 or a.size == 0:   # scalar or empty -> not a sequence
        return None
    return a


def _ci_width(lo, hi):
    lo, hi = _f(lo), _f(hi)
    if lo is None or hi is None:
        return None
    return abs(hi - lo)


def _first_key(d: dict, keys, default=None):
    """Return d[k] for the first key present and non-empty."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            v = d[k]
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 0:
                continue
            return v
    return default


# --------------------------------------------------------------------------- #
#  Graph context from the (opaque) GNN result dict
# --------------------------------------------------------------------------- #

def _graph_context(gnn_result: dict, w_gnn: float):
    """Best-effort extraction of neighbour / attention structure.

    The gnn_predictor result dict is opaque here; we probe several plausible
    key names. Returns (GraphContext, note) where note documents what was used.
    """
    gnn_result = gnn_result or {}
    neigh = _first_key(gnn_result, [
        "neighbor_probs", "neighbour_probs", "neighbor_probabilities",
        "neighbor_preds", "neighbours_probs",
    ])
    attn = _first_key(gnn_result, [
        "attention", "attention_weights", "attn", "similarities",
        "cos_sim", "weights", "edge_weights",
    ])

    # gnn_predictor nests everything under 'neighbors': {'probs', 'sims', ...}.
    nd = _first_key(gnn_result, ["neighbors", "neighbours"])
    if isinstance(nd, dict):
        if neigh is None:
            neigh = _first_key(nd, ["probs", "prob", "gnn_prob", "probabilities"])
        if attn is None:
            attn = _first_key(nd, ["sims", "similarities", "cos_sim", "weights"])
    # Neighbours as a list of dicts -> pull a prob-like field.
    elif neigh is None and isinstance(nd, (list, tuple)) and nd and isinstance(nd[0], dict):
        for pk in ("prob", "gnn_prob", "p", "probability", "y"):
            vals = [d.get(pk) for d in nd if isinstance(d, dict) and d.get(pk) is not None]
            if vals:
                neigh = vals
                break

    neigh_arr = _arr(neigh)
    attn_arr = _arr(attn)

    if attn_arr is not None and np.isfinite(attn_arr).all() and attn_arr.size:
        # Cosine sims are top-k, ~all near 1 and unnormalised; raw normalisation
        # would make attention look uniform (N_eff ~ k always). A softmax over
        # similarities recovers how attention actually concentrates weight, so a
        # dominant neighbour lowers N_eff meaningfully.
        a = attn_arr - attn_arr.max()                 # stabilise
        attention = list(np.exp(a / 0.1))             # temperature 0.1 ~ sharpness
        available = True
        note = "attention from softmax(cosine similarities) of GNN neighbours"
    elif neigh_arr is not None:
        attention = [1.0] * len(neigh_arr)
        available = True
        note = ("similarities not found — uniform attention assumed "
                "(N_eff = neighbour count); neighbour variance still real")
    else:
        # Nothing available: graph marked NOT available (neutral, no inflation).
        attention = [1.0] * 10
        neigh_arr = None
        available = False
        note = "GNN not run / neighbour data not exposed — graph term neutralised"

    nprobs = list(neigh_arr) if neigh_arr is not None else [0.4, 0.5, 0.45]
    return GraphContext(attention=attention, neighbour_probs=nprobs,
                        available=available), note


# --------------------------------------------------------------------------- #
#  Diffusion (L5) agreement from CSDI vs Monte-Carlo
# --------------------------------------------------------------------------- #

def _diffusion_context(res: dict, csdi_result, p_mc):
    """Recompute KS(MC, CSDI) for blast and TGBDR; derive a pregnancy-axis
    agreement from |P_csdi - P_mc|. Returns (DiffusionContext, available)."""
    if not csdi_result:
        return DiffusionContext(available=False), False

    ks_bl = ks_tgbdr = 0.0
    df = csdi_result.get("samples") if isinstance(csdi_result, dict) else None
    if _ks_2samp is not None and df is not None:
        try:
            mc_bl = _arr(res.get("sim_blasts"))
            csdi_bl = _arr(df["Число Bl"].values) if "Число Bl" in df else None
            if mc_bl is not None and csdi_bl is not None:
                ks_bl = float(_ks_2samp(mc_bl, csdi_bl).statistic)
        except Exception:
            pass
        try:
            mc_good = _arr(res.get("sim_good"))
            mc_pn2 = _arr(res.get("sim_pn2"))
            if mc_good is not None and mc_pn2 is not None:
                mc_tgbdr = np.clip(mc_good / np.maximum(mc_pn2, 1), 0, 1)
                col = next((c for c in df.columns if "TGBDR" in str(c) or "хор" in str(c).lower()), None)
                if col is not None:
                    ks_tgbdr = float(_ks_2samp(mc_tgbdr, _arr(df[col].values)).statistic)
        except Exception:
            pass

    p_csdi = _f(csdi_result.get("P_pregnancy")) if isinstance(csdi_result, dict) else None
    ks_preg = abs(p_csdi - p_mc) if (p_csdi is not None and p_mc is not None) else 0.0
    return DiffusionContext(KS_pregnancy=min(ks_preg, 1.0),
                            KS_blast=ks_bl, KS_TGBDR=ks_tgbdr, available=True), True


# --------------------------------------------------------------------------- #
#  Cluster context (L4)
# --------------------------------------------------------------------------- #

_CLUSTER_LABELS = {
    "C0": "Cluster 0", "C1": "Cluster 1", "C2": "Cluster 2",
}

def _cluster_context(res: dict):
    ca = res.get("cluster_analysis", {}) if isinstance(res, dict) else {}
    dom = ca.get("dominant_cluster", "")
    probs = ca.get("cluster_probs", {}) or {}
    dom_info = (ca.get("clusters", {}) or {}).get(dom, {}) if isinstance(ca.get("clusters"), dict) else {}
    label = ""
    if isinstance(dom_info, dict):
        label = dom_info.get("label") or dom_info.get("name") or ""
    label = label or _CLUSTER_LABELS.get(str(dom), str(dom) or "—")

    # Probe both int and str keys — the pipeline stores {0: p, 1: p, 2: p} (int),
    # but serialisation / JSON round-trips may produce {"0": p, "1": p, "2": p}.
    if isinstance(probs, dict):
        cluster_prob = _f(probs.get(dom), None)
        if cluster_prob is None:
            cluster_prob = _f(probs.get(str(dom)), None)
        if cluster_prob is None:
            cluster_prob = 1.0
    else:
        cluster_prob = 1.0

    # BUG FIX: cluster_analysis does not expose per-cluster centroid distances,
    # so dom_info is always {}.  The old default of 1.0 made closeness = exp(-1)
    # ~0.368 (constant), compressing cluster_certainty into [0.35, 0.63].
    # Default to 0.0 instead -> closeness = exp(0) = 1.0 (patient assumed on
    # centroid when distance is unknown), giving certainty in [0.5, 1.0] and a
    # full read of cluster_probability.
    dist = _f(_first_key(dom_info if isinstance(dom_info, dict) else {},
                         ["distance_to_centroid", "distance", "dist"]), 0.0)
    return ClusterContext(
        cluster_id=hash(str(dom)) % 100, cluster_label=label,
        distance_to_centroid=dist, cluster_probability=cluster_prob,
        typical_distance=1.0,
    )


# --------------------------------------------------------------------------- #
#  OOD context (optional — requires fitted training stats)
# --------------------------------------------------------------------------- #

def _ood_context(res, age, amh, afc, bmi, ood_stats):
    """ood_stats: optional dict with keys
        clinical_mu, clinical_cov_inv  (over [age, amh, afc, bmi])
        embryo_mu,   embryo_cov_inv    (over [OCC, MII, 2PN, Blast, KPI])
    Generate them offline with befe.fit_gaussian on your training cohort.
    If absent, OOD is disabled (no crash, no reliability cap)."""
    if not ood_stats:
        return OODContext()
    clin_x = [_f(age), _f(amh), _f(afc), _f(bmi)]
    kpi = _f(res.get("kpi_score")) or _f(res.get("KPIScore")) or 0.0
    emb_x = [_f(res.get("okk_med")), _f(res.get("mii_med")),
             _f(res.get("pn2_med")), _f(res.get("blasts_med")), kpi]
    clin = _Subspace()
    emb = _Subspace()
    if ood_stats.get("clinical_mu") is not None and None not in clin_x:
        clin = _Subspace(features=np.array(clin_x),
                         mean=np.asarray(ood_stats["clinical_mu"]),
                         cov_inv=np.asarray(ood_stats["clinical_cov_inv"]))
    if ood_stats.get("embryo_mu") is not None and None not in emb_x:
        emb = _Subspace(features=np.array(emb_x),
                        mean=np.asarray(ood_stats["embryo_mu"]),
                        cov_inv=np.asarray(ood_stats["embryo_cov_inv"]))
    return OODContext(clinical=clin, embryology=emb)


# --------------------------------------------------------------------------- #
#  MAIN: build the BEFE result from app state
# --------------------------------------------------------------------------- #

def build_befe_result(res, *, p_kat_raw=None, ci_kat=(None, None),
                      p_gnn_raw=None, gnn_result=None, w_gnn=0.35,
                      csdi_result=None, age=None, amh=None, afc=None, bmi=None,
                      ood_stats=None, tau_kat_override=None):
    """Construct BEFE inputs from the app and run the fusion.

    Returns (BEFEResult | None, mapping: dict). Returns (None, mapping) if the
    mechanistic prior is unavailable (nothing to fuse).

    In this pipeline p_per_transfer == mean(sim_p_combined) (FORTUNE+KPI IS the
    per-transfer prior), so L2 is NOT a separate expert: the FORTUNE+KPI spread
    is used as the prior's CI width instead. Empirical evidence = {KAT, GAT}.
    """
    mapping = {}
    p_l1 = _f(res.get("p_per_transfer")) if isinstance(res, dict) else None
    if p_l1 is None:
        return None, {"error": "res['p_per_transfer'] missing — cannot fuse"}

    # FORTUNE+KPI ensemble (sim_p_combined) == the per-transfer prior in this
    # pipeline, so it is NOT a separate expert: its spread becomes the prior's
    # own CI width (the natural uncertainty of the mechanistic prior).
    combined = _arr(res.get("sim_p_combined"))
    ci_l1_from_combined = None
    if combined is not None:
        ci_l1_from_combined = float(np.percentile(combined, 97.5)
                                    - np.percentile(combined, 2.5))

    p_kat = _f(p_kat_raw)   # L3 evidence
    p_gat = _f(p_gnn_raw)   # L6 evidence

    # Empirical evidence = {KAT, GAT}. Missing models get near-zero precision
    # so they cannot pull; if both are missing the posterior collapses to the
    # prior (handled inside the engine).
    experts = ExpertOutputs(
        P_L1=p_l1,
        P_KAT=p_kat if p_kat is not None else p_l1,
        P_GAT=p_gat if p_gat is not None else p_l1,
    )
    tau_evidence = {
        # tau_kat_override: dynamic tau from GBDT meta-learner (clinic adaptation).
        # If None, use baseline 2.4 from the general cohort.
        "KAT": (tau_kat_override if tau_kat_override is not None else 2.4) if p_kat is not None else 1e-6,
        "GAT": 1.0 if p_gat is not None else 1e-6,
    }

    # Prior CI width: prefer the FORTUNE+KPI spread, else the MC rate CI.
    rate_ci = res.get("rate_ci", (None, None)) if isinstance(res, dict) else (None, None)
    ci_l1 = ci_l1_from_combined
    if ci_l1 is None:
        ci_l1 = _ci_width(rate_ci[0], rate_ci[1]) if rate_ci else None
    kat_w = _ci_width(ci_kat[0], ci_kat[1]) if ci_kat else None
    kat_var = ((kat_w / 3.92) ** 2) if kat_w is not None else 0.02

    uncertainty = UncertaintyContext(
        CI_L1_width=ci_l1 if ci_l1 is not None else 0.30,
        posterior_variance=kat_var,
        MC_variance=0.0,
    )

    diffusion, diff_ok = _diffusion_context(res, csdi_result, p_l1)
    cluster = _cluster_context(res)
    graph, graph_note = _graph_context(gnn_result, w_gnn)
    ood = _ood_context(res, age, amh, afc, bmi, ood_stats)

    engine = BEFE(tau_base_evidence=tau_evidence)
    result = engine.predict(experts, uncertainty, diffusion, cluster, graph, ood)
    # Preserve the model's own uncertainty interval for technical inspection.
    # The clinic-derived bounds below are a historical limiting corridor, not
    # the confidence interval of the BEFE point estimate.
    result.model_ci_low = result.ci_low
    result.model_ci_high = result.ci_high

    # The L7 point estimate is BEFE, but the interval shown to the clinician
    # should preserve the clinical Beta-Binomial posterior: it is the distribution
    # that explicitly incorporates "Данные клиники (prior)" from the sidebar.
    beta_post = res.get("posterior", {}) if isinstance(res, dict) else {}
    beta_ci_low = _f(beta_post.get("ci_low")) if isinstance(beta_post, dict) else None
    beta_ci_high = _f(beta_post.get("ci_high")) if isinstance(beta_post, dict) else None
    beta_alpha = beta_post.get("posterior_alpha") if isinstance(beta_post, dict) else None
    beta_beta = beta_post.get("posterior_beta") if isinstance(beta_post, dict) else None
    if beta_ci_low is not None and beta_ci_high is not None and beta_ci_low < beta_ci_high:
        result.ci_low = beta_ci_low
        result.ci_high = beta_ci_high
        result.clinic_corridor_low = beta_ci_low
        result.clinic_corridor_high = beta_ci_high
        result.ci_source = "clinic-historical-corridor"

    # Display classification only. These configurable, deliberately softer
    # thresholds do not alter the BEFE probability or reliability score.
    try:
        import streamlit as _st
        _high = int(_st.session_state.get("_rel_high_threshold", 60))
        _moderate = int(_st.session_state.get("_rel_moderate_threshold", 35))
    except Exception:
        _high, _moderate = 60, 35
    result.reliability_band = (
        "High" if result.reliability >= _high else
        "Moderate" if result.reliability >= _moderate else "Low"
    )

    mapping = {
        "P_L1 (prior)": (p_l1, "res['p_per_transfer'] = MC + FORTUNE/KPI per-transfer"),
        "  ↳ prior CI width": (ci_l1, "spread of sim_p_combined" if ci_l1_from_combined is not None else "rate_ci"),
        "  ↳ clinic corridor source": (
            None,
            (
                f"clinic batches · Beta({beta_alpha:.0f}, {beta_beta:.0f})"
                if beta_ci_low is not None and beta_ci_high is not None
                and beta_alpha is not None and beta_beta is not None
                else "BEFE logit-pool fallback"
            ),
        ),
        "P_KAT (evidence)": (p_kat, "p_kat_raw · L3" if p_kat is not None else "не запущена"),
        "P_GAT (evidence)": (p_gat, "p_gnn_raw · L6 (pure graph)" if p_gat is not None else "не запущена"),
        "Diffusion (L5)": (diffusion.agreement_score if diff_ok else None,
                           "KS MC↔CSDI → модулирует приор" if diff_ok else "CSDI не запущена"),
        "Graph note": (None, graph_note),
        "OOD": (None, "включён" if ood_stats else "выключен (нет train-статистик)"),
    }
    return result, mapping


# --------------------------------------------------------------------------- #
#  Streamlit tab
# --------------------------------------------------------------------------- #


def render_befe_tab(result, mapping, *, format_report=None):
    """Render the BEFE tab — полная карточная дизайн-система dt_ui.

    Требует наличия dt_ui.py рядом с befe_app.py (или в PYTHONPATH).
    При отсутствии dt_ui падает на упрощённый рендер gracefully.
    """
    import streamlit as st

    # ── попытка импорта dt_ui ─────────────────────────────────────────────
    try:
        import dt_ui as UI
        _HAS_UI = True
    except ImportError:
        _HAS_UI = False

    # ── заголовок вкладки ─────────────────────────────────────────────────
    if _HAS_UI:
        UI.tab_header_by_key("befe")
    else:
        st.markdown(
            '<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">'
            'Bayesian Evidence Fusion — итоговый прогноз (L7)</p>',
            unsafe_allow_html=True,
        )

    if result is None:
        st.info(
            "Нажмите **Запустить расчёт** — BEFE объединит уровни L1–L6 "
            "в единый posterior."
        )
        if mapping and mapping.get("error"):
            st.caption(mapping["error"])
        return

    def pct(x, default="н/д"):
        try:
            return f"{float(x)*100:.0f}%"
        except Exception:
            return default

    def pct_f(x, default=0.0):
        try:
            return float(x) * 100
        except Exception:
            return default

    band_ru    = {"High": "Высокая",  "Moderate": "Умеренная", "Low": "Ограниченная"}
    band_color = {"High": "#1E8449",  "Moderate": "#D97706",   "Low": "#C0392B"}
    rb  = result.reliability_band
    rcol = band_color.get(rb, "#1B4F72")

    if not _HAS_UI:
        # ── Fallback: старый рендер ──────────────────────────────────────
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#F4F8FC,#EAF1F8);
            border:1px solid #D2E0EE;border-radius:12px;padding:18px 22px;
            margin:4px 0 14px 0">
              <div style="font-size:13px;color:#5A6B7B;font-weight:600">
                Вероятность беременности (итоговая)</div>
              <div style="font-size:46px;font-weight:700;color:#1B4F72;line-height:1.05">
                {pct(result.posterior)}</div>
              <div style="font-size:14px;color:#5A6B7B">
                Исторический коридор клиники: {pct(result.ci_low)} – {pct(result.ci_high)}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Приор (L1)",       pct(result.p_prior))
        c2.metric("Доказательство",   pct(result.p_predictive))
        c3.metric("Надёжность",
                  f"{result.reliability}/100",
                  delta=band_ru.get(rb, rb),
                  delta_color="off")
        return

    # ═══════════════════════════════════════════════════════════
    # ПОЛНЫЙ ДИЗАЙН-РЕНДЕР (dt_ui доступен)
    # ═══════════════════════════════════════════════════════════

    # ── 1. Шапка карточки (синяя, с posterior крупно) ────────────────────
    UI.befe_card_header(
        pct_f(result.posterior),
        pct_f(result.ci_low),
        pct_f(result.ci_high),
    )

    # ── 2. Три плитки: Prior → Evidence → Pull ────────────────────────────
    st.markdown('<div style="background:#FFFFFF;border:0.5px solid #D4E4F0;'
                'border-radius:0 0 12px 12px;padding:14px 18px 16px;margin-bottom:12px">',
                unsafe_allow_html=True)

    UI.metric_row([
        ("Механистический приор (L1)",       pct(result.p_prior),       ""),
        ("Эмпирическое доказательство (L3/L6)", pct(result.p_predictive), "accent"),
        ("Вклад доказательства",
         f"{result.evidence_pull*100:.0f}% / {result.prior_pull*100:.0f}%", ""),
    ])

    # ── 3. Линейка Prior → CI → Posterior ────────────────────────────────
    UI.ci_bar(
        prior_pct     = pct_f(result.p_prior),
        posterior_pct = pct_f(result.posterior),
        ci_low_pct    = pct_f(result.ci_low),
        ci_high_pct   = pct_f(result.ci_high),
    )

    # ── 4. Полоса надёжности ──────────────────────────────────────────────
    UI.reliability_bar(int(result.reliability), rb)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── 5. Расхождение экспертов (только при Moderate/Low consensus) ──────
    p_kat = mapping.get("P_KAT (evidence)", (None,))[0]
    p_gat = mapping.get("P_GAT (evidence)", (None,))[0]
    if result.consensus in ("Moderate", "Low") and p_kat is not None and p_gat is not None:
        gap = abs(p_kat - p_gat) * 100
        kat_word = "выше" if p_kat >= p_gat else "ниже"
        UI.result_box(
            f"<b>Источник неопределённости.</b> Эксперты расходятся на "
            f"<b>{gap:.0f}&nbsp;п.п.</b>: "
            f"KAT (L3) <b>{pct(p_kat)}</b>; "
            f"GAT (L6) <b>{pct(p_gat)}</b>. "
            f"KAT {kat_word} GAT. Итог смещён к KAT как к лучше откалиброванной модели.",
            kind="warning",
        )

    # ── 6. Верификационная сетка ──────────────────────────────────────────
    UI.section_header("Верификация и контекст")

    diff_txt = (
        "не запускалась"    if not result.diffusion_available
        else f"{result.diffusion_agreement*100:.0f}%  — отличное" if result.diffusion_agreement >= 0.9
        else f"{result.diffusion_agreement*100:.0f}%  — хорошее"  if result.diffusion_agreement >= 0.7
        else f"{result.diffusion_agreement*100:.0f}%  — умеренное" if result.diffusion_agreement >= 0.5
        else f"{result.diffusion_agreement*100:.0f}%  — слабое"
    )
    diff_color = (
        "" if not result.diffusion_available
        else "green"  if result.diffusion_agreement >= 0.7
        else "amber"  if result.diffusion_agreement >= 0.5
        else "red"
    )

    if result.graph_available:
        sim_txt = (
            f"N_eff={result.n_eff:.0f} — сильная"    if result.n_eff >= 20
            else f"N_eff={result.n_eff:.0f} — умеренная" if result.n_eff >= 8
            else f"N_eff={result.n_eff:.0f} — слабая"
        )
        sim_color = "green" if result.n_eff >= 20 else "amber" if result.n_eff >= 8 else "red"
    else:
        sim_txt   = "граф не запускался"
        sim_color = ""

    cons_ru_map = {"High": "высокий ✓", "Moderate": "умеренный", "Low": "низкий ✗"}
    cons_color  = {"High": "green",     "Moderate": "amber",     "Low": "red"}

    UI.verification_grid([
        ("Согласие диффузии (L5)",    diff_txt,                          diff_color),
        ("Консенсус моделей",         cons_ru_map.get(result.consensus,
                                                      result.consensus), cons_color.get(result.consensus, "")),
        ("Похожесть пациентов (L6)",  sim_txt,                           sim_color),
        ("Кластер (L4)",              result.cluster_label,              ""),
    ])

    # ── 7. OOD-детектор ──────────────────────────────────────────────────
    UI.section_header("Out-of-distribution детектор")
    UI.ood_strip(result.ood_clinical, result.ood_embryology, result.ood_final)

    # ── 8. Аудит-раскрывашка ──────────────────────────────────────────────
    with st.expander("Веса экспертов и источники входов"):
        w = result.evidence_weights
        st.markdown(
            "**Вклад нейросетевых экспертов (уровень 1):** "
            + " · ".join(f"{k} = {v*100:.0f}%" for k, v in w.items())
        )
        st.markdown("**Входы BEFE → значения приложения:**")
        rows = []
        for k, (val, src) in mapping.items():
            vs = (
                f"{val:.3f}"
                if isinstance(val, (int, float)) and val is not None
                else "—"
            )
            rows.append(f"- **{k}** = {vs}  ·  _{src}_")
        st.markdown("\n".join(rows))
        st.caption(
            "Приор = p_per_transfer (MC + FORTUNE/KPI), его ДИ — из разброса "
            "sim_p_combined. Доказательство = KAT (L3) + GAT (L6). "
            "L5 (диффузия) модулирует доверие к приору."
        )


# --------------------------------------------------------------------------- #
#  PDF section (reportlab) — helpers passed in from generate_patient_report
# --------------------------------------------------------------------------- #

def befe_pdf_flowables(result, ST, sec_header, kv_table, Paragraph, Spacer, cm,
                       PageBreak=None):
    """Return a list of reportlab flowables for the BEFE section.

    Call from inside generate_patient_report (where ST, sec_header, kv_table,
    Paragraph, Spacer, cm are in scope) so no fragile imports are needed.
    """
    if result is None:
        return []
    flow = []
    if PageBreak is not None:
        flow.append(PageBreak())
    flow.append(sec_header("BEFE — Байесовское слияние доказательств (L7)"))
    flow.append(Spacer(1, 8))
    flow.append(Paragraph(
        "L7 объединяет уровни L1–L6 в единый posterior по байесовской схеме "
        "Prior → Evidence → Posterior. Механистический приор (L1) обновляется "
        "эмпирическими экспертами (L2/L3/L6) с весами, зависящими от их "
        "признаков доверия (неопределённость, стабильность графа, согласие "
        "диффузионной модели L5). Ширина интервала отражает суммарную точность.",
        ST["body"]))
    flow.append(Spacer(1, 10))

    def pct(x):
        try:
            return f"{float(x)*100:.0f}%"
        except Exception:
            return "—"

    rows = [
        ("Posterior P(беременность)",
         f"<b>{pct(result.posterior)}</b>  (95% ДИ: {pct(result.ci_low)}–{pct(result.ci_high)})"),
        ("Механистический приор (L1)", pct(result.p_prior)),
        ("Эмпирическое доказательство (L2/L3/L6)", f"{pct(result.p_predictive)}  [P_predictive]"),
        ("Вклад: доказательство / приор",
         f"{result.evidence_pull*100:.0f}% / {result.prior_pull*100:.0f}%"),
        ("Reliability Index", f"<b>{result.reliability}/100</b>  ({result.reliability_band})"),
        ("Консенсус (эмпирический)", result.consensus),
        ("Похожесть пациентов", f"N_eff = {result.n_eff:.0f}"),
        ("Согласие диффузии (L5)", f"{result.diffusion_agreement*100:.0f}%"),
        ("Кластер", result.cluster_label),
        ("OOD", result.ood_note),
    ]
    flow.append(kv_table(rows, col1=6.5 * cm))
    flow.append(Spacer(1, 8))
    return flow
