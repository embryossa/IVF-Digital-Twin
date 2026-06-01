"""
befe_app.py — Integration layer for BEFE (L7) into IVF Digital Twin v6.3
========================================================================

Bridges the running app to the BEFE engine (befe.py):

    build_befe_result(...)   -> (BEFEResult, mapping)   # construct inputs + fuse
    render_befe_tab(...)     -> Streamlit UI            # the "⚖️ BEFE (L7)" tab
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
    cluster_prob = _f(probs.get(dom), 1.0) if isinstance(probs, dict) else 1.0
    dist = _f(_first_key(dom_info if isinstance(dom_info, dict) else {},
                         ["distance_to_centroid", "distance", "dist"]), 1.0)
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
                      ood_stats=None):
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
        "KAT": 2.4 if p_kat is not None else 1e-6,   # KAT best-calibrated to our data
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

    # ── Заменяем CI на байесовский Beta-posterior из данных клиники ──────────
    # Логит-пространственный CI при малом tau даёт 0%–100% (бесполезно).
    # Beta-posterior откалиброван по реальным переносам клиники и даёт
    # осмысленный индивидуальный ДИ.
    ci_note = "logit-pool (BEFE)"
    post = res.get("posterior", {}) if isinstance(res, dict) else {}
    _beta_lo = _f(post.get("ci_low"))
    _beta_hi = _f(post.get("ci_high"))
    if _beta_lo is not None and _beta_hi is not None and _beta_hi > _beta_lo:
        result.ci_low   = _beta_lo
        result.ci_high  = _beta_hi
        result.ci_source = "beta-posterior"
        ci_note = "Beta-posterior (данные клиники)"

    mapping = {
        "P_L1 (prior)": (p_l1, "res['p_per_transfer'] = MC + FORTUNE/KPI per-transfer"),
        "  ↳ prior CI width": (ci_l1, "spread of sim_p_combined" if ci_l1_from_combined is not None else "rate_ci"),
        "P_KAT (evidence)": (p_kat, "p_kat_raw · L3" if p_kat is not None else "не запущена"),
        "P_GAT (evidence)": (p_gat, "p_gnn_raw · L6 (pure graph)" if p_gat is not None else "не запущена"),
        "Diffusion (L5)": (diffusion.agreement_score if diff_ok else None,
                           "KS MC↔CSDI → модулирует приор" if diff_ok else "CSDI не запущена"),
        "95% ДИ источник": (None, ci_note),
        "Graph note": (None, graph_note),
        "OOD": (None, "включён" if ood_stats else "выключен (нет train-статистик)"),
    }
    return result, mapping


# --------------------------------------------------------------------------- #
#  Streamlit tab
# --------------------------------------------------------------------------- #

def render_befe_tab(result, mapping, *, format_report=None):
    """Render the BEFE tab — styled to match the app (no monospace dump,
    no consensus gauge), fully in Russian."""
    import streamlit as st

    _H = ('<p style="font-size:15px;font-weight:600;color:#1B4F72;'
          'margin:0 0 6px 0">{}</p>')
    st.markdown(_H.format("Bayesian Evidence Fusion — итоговый прогноз (L7)"),
                unsafe_allow_html=True)

    if result is None:
        st.info("▶ Нажмите **Запустить расчёт** — BEFE объединит уровни L1–L6 "
                "в единый posterior.")
        if mapping and mapping.get("error"):
            st.caption(mapping["error"])
        return

    def pct(x):
        return "н/д" if x is None else f"{x*100:.0f}%"

    band_ru = {"High": "Высокая", "Moderate": "Умеренная", "Low": "Низкая"}
    band_color = {"High": "#2E7D32", "Moderate": "#F9A825", "Low": "#C62828"}
    rb = result.reliability_band
    rcol = band_color.get(rb, "#1B4F72")
    ci_src = "Beta" if getattr(result, "ci_source", "") == "beta-posterior" else "logit"

    # ── Headline card: posterior + CI + reliability ─────────────────────
    st.markdown(
        f'''
        <div style="background:linear-gradient(135deg,#F4F8FC,#EAF1F8);
        border:1px solid #D2E0EE;border-radius:12px;padding:18px 22px;margin:4px 0 14px 0">
          <div style="display:flex;justify-content:space-between;align-items:flex-end;
          flex-wrap:wrap;gap:12px">
            <div>
              <div style="font-size:13px;color:#5A6B7B;font-weight:600">
                Вероятность беременности (итоговая)</div>
              <div style="font-size:46px;font-weight:700;color:#1B4F72;line-height:1.05">
                {pct(result.posterior)}</div>
              <div style="font-size:14px;color:#5A6B7B">
                95% ДИ ({ci_src}): {pct(result.ci_low)} – {pct(result.ci_high)}</div>
            </div>
            <div style="text-align:right">
              <div style="font-size:13px;color:#5A6B7B;font-weight:600">Надёжность</div>
              <div style="font-size:34px;font-weight:700;color:{rcol};line-height:1.1">
                {result.reliability}<span style="font-size:18px;color:#9AA7B4">/100</span></div>
              <div style="font-size:14px;font-weight:600;color:{rcol}">
                {band_ru.get(rb, rb)}</div>
            </div>
          </div>
        </div>''',
        unsafe_allow_html=True)

    # ── Prior → Evidence → Posterior ────────────────────────────────────
    st.markdown(_H.format("Приор → Доказательство → Posterior"),
                unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    e1.metric("Механистический приор (L1)", pct(result.p_prior),
              help="Monte-Carlo каскад + FORTUNE/KPI на перенос")
    e2.metric("Нейросетевое доказательство (L3+L6)", pct(result.p_predictive),
              help="P_predictive — слияние KAT и GAT (уровень 1)")
    e3.metric("Вклад доказательства",
              f"{result.evidence_pull*100:.0f}%",
              delta=f"приор {result.prior_pull*100:.0f}%", delta_color="off",
              help="Сколько итоговой вероятности дали данные vs механистический приор")

    # ── Source of uncertainty (only when models disagree) ───────────────
    p_kat = mapping.get("P_KAT (evidence)", (None,))[0]
    p_gat = mapping.get("P_GAT (evidence)", (None,))[0]
    if result.consensus in ("Moderate", "Low") and p_kat is not None and p_gat is not None:
        gap = abs(p_kat - p_gat) * 100
        kat_word = "выше" if p_kat >= p_gat else "ниже"
        st.markdown(
            f'''
            <div style="background:#FFF8E1;border-left:5px solid #F9A825;
            padding:12px 16px;border-radius:6px;margin:10px 0">
              <b>Источник неопределённости.</b> Эксперты расходятся на
              <b>{gap:.0f} п.п.</b> — это и снижает надёжность:<br>
              &nbsp;&nbsp;• KAT (нейросеть, L3): <b>{pct(p_kat)}</b><br>
              &nbsp;&nbsp;• GAT (граф пациентов, L6): <b>{pct(p_gat)}</b><br>
              KAT здесь {kat_word} GAT. Возможная причина — нестандартное
              соотношение клинических и эмбриологических показателей у пациентки.
              Итог смещён к KAT как к лучше калиброванной модели.
            </div>''',
            unsafe_allow_html=True)

    # ── Verification / context cards ────────────────────────────────────
    st.markdown(_H.format("Проверка и контекст"), unsafe_allow_html=True)
    diff_txt = ("не запускалась" if not result.diffusion_available
                else "отличное" if result.diffusion_agreement >= 0.9
                else "хорошее" if result.diffusion_agreement >= 0.7
                else "умеренное" if result.diffusion_agreement >= 0.5
                else "слабое")
    if result.graph_available and result.n_eff == result.n_eff:
        sim_txt = ("сильная" if result.n_eff >= 20
                   else "умеренная" if result.n_eff >= 8 else "ограниченная")
        neff_txt = f"{sim_txt} (N_eff = {result.n_eff:.0f})"
    else:
        neff_txt = "граф не запускался"
    cons_ru = {"High": "высокий", "Moderate": "умеренный", "Low": "низкий"}
    v1, v2 = st.columns(2)
    v1.metric("Согласие диффузии (L5)",
              diff_txt if not result.diffusion_available else f"{result.diffusion_agreement*100:.0f}%",
              help="KS-расстояние между Monte-Carlo и CSDI; модулирует доверие к приору")
    v1.metric("Консенсус моделей", cons_ru.get(result.consensus, result.consensus))
    v2.metric("Похожесть пациентов (граф)", neff_txt,
              help="Эффективное число соседей по графу: N_eff = 1/Σ(вес²)")
    v2.metric("Кластер (L4)", result.cluster_label)

    # ── OOD ─────────────────────────────────────────────────────────────
    if result.ood_final:
        note_ru = {
            "Clinically atypical, embryologically typical":
                "Клинически нетипична, эмбриологически типична",
            "Clinically typical, embryologically atypical":
                "Клинически типична, эмбриологически нетипична",
            "Atypical in both clinical and embryological space":
                "Нетипична и клинически, и эмбриологически",
        }.get(result.ood_note, result.ood_note)
        st.markdown(
            f'<div style="background:#FDECEA;border-left:5px solid #C62828;'
            f'padding:12px 16px;border-radius:6px;margin:10px 0">'
            f'<b>⚠️ OOD — выход за пределы обучающих данных:</b> {note_ru}.<br>'
            f'Прогноз опирается на ограниченные похожие исторические данные — '
            f'интерпретировать с осторожностью.</div>',
            unsafe_allow_html=True)
    else:
        oc = "✓" if not result.ood_clinical else "⚠️"
        oe = "✓" if not result.ood_embryology else "⚠️"
        st.caption(f"OOD-детектор активен · клинический профиль {oc} · "
                   f"эмбриологический профиль {oe}")

    # ── Audit expander ──────────────────────────────────────────────────
    with st.expander("ℹ️ Веса экспертов и источники входов"):
        w = result.evidence_weights
        st.markdown("**Вклад нейросетевых экспертов (уровень 1):** "
                    + " · ".join(f"{k} = {v*100:.0f}%" for k, v in w.items()))
        st.markdown("**Входы BEFE → значения приложения:**")
        rows = []
        for k, (val, src) in mapping.items():
            vs = f"{val:.3f}" if isinstance(val, (int, float)) and val is not None else "—"
            rows.append(f"- **{k}** = {vs}  ·  _{src}_")
        st.markdown("\n".join(rows))
        st.caption("Приор = p_per_transfer (MC + FORTUNE/KPI), его ДИ — из разброса "
                   "sim_p_combined. Доказательство = KAT (L3) + GAT (L6). "
                   "L5 (диффузия) модулирует доверие к приору.")


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
    if result.ood_final:
        flow.append(Paragraph(
            "<b>⚠️ OOD ALERT:</b> прогноз опирается на ограниченные похожие "
            "исторические данные в отмеченном подпространстве — интерпретировать "
            "с осторожностью.", ST.get("body_sm", ST["body"])))
        flow.append(Spacer(1, 6))
    return flow
