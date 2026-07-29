# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""Clinician-first result page for the independent UI redesign."""

from __future__ import annotations

import math
import copy
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from i18n import tr, reliability_label


PALETTE = {
    "navy": "#163E59",
    "blue": "#6F93B7",
    "teal": "#78AAA5",
    "green": "#8DBA8D",
    "amber": "#DDBB72",
    "orange": "#D9A36A",
    "red": "#C98282",
    "purple": "#A792C6",
    "grey": "#71808C",
}


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _pct(value, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _num(value, digits: int = 0) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _plot_theme(fig: go.Figure, *, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12,
                  color="#314A5B"),
        margin=dict(l=54, r=26, t=42, b=50),
        legend=dict(
            orientation="h", x=0, xanchor="left", y=1.03, yanchor="bottom",
            bgcolor="rgba(255,255,255,0)", font=dict(size=11),
        ),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#D4E4F0"),
    )
    fig.update_xaxes(
        gridcolor="rgba(111,147,183,0.16)", zeroline=False,
        linecolor="rgba(111,147,183,0.22)",
    )
    fig.update_yaxes(
        gridcolor="rgba(111,147,183,0.16)", zeroline=False,
        linecolor="rgba(111,147,183,0.22)",
    )
    return fig


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .clinical-hero {
          background:linear-gradient(135deg,#FFFFFF 0%,#F4F8FB 100%);
          border:1px solid #C9DBE8;border-radius:18px;padding:22px 24px;
          box-shadow:0 8px 24px rgba(22,62,89,.06);margin:4px 0 16px;
        }
        .clinical-kicker {font-size:11px;text-transform:uppercase;letter-spacing:.09em;
          color:#6A8294;font-weight:700;margin-bottom:5px}
        .clinical-value {font-size:54px;line-height:1;font-weight:780;color:#163E59}
        .clinical-title {font-size:18px;font-weight:720;color:#244C65;margin:4px 0 12px}
        .clinical-row {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
        .clinical-tile {background:rgba(235,243,249,.78);border:1px solid #D7E5EF;
          border-radius:12px;padding:11px 13px}
        .clinical-label {font-size:12px;color:#607687;line-height:1.25}
        .clinical-number {font-size:24px;font-weight:750;color:#244C65;margin-top:3px}
        .clinical-corridor {margin-top:13px;padding:10px 12px;border-radius:10px;
          background:#F8FAFC;border:1px dashed #B7C9D7;color:#496273;font-size:12px}
        .clinical-context {display:flex;gap:7px;flex-wrap:wrap;margin:0 0 14px}
        .clinical-chip {background:#F3F7FA;border:1px solid #D9E5ED;border-radius:999px;
          padding:5px 10px;font-size:12px;color:#496273}
        .risk-card {background:#fff;border:1px solid #D8E4ED;border-radius:13px;
          padding:14px 15px;min-height:96px}
        .risk-card .value {font-size:25px;font-weight:750;color:#244C65}
        .risk-card .label {font-size:12px;color:#627888;margin-top:3px}
        .model-line {display:flex;justify-content:space-between;align-items:center;
          padding:9px 0;border-bottom:1px solid #E7EEF3;font-size:13px}
        .model-ok {color:#3F7558;font-weight:700}.model-off {color:#8C6A55;font-weight:700}
        @media(max-width:900px){.pipeline-row{grid-template-columns:repeat(3,minmax(0,1fr)) !important}}
        @media(max-width:800px){.clinical-row{grid-template-columns:1fr}.pipeline-row{grid-template-columns:repeat(2,minmax(0,1fr)) !important}.clinical-value{font-size:44px}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero(g: dict, language: str) -> None:
    res = g.get("res") or {}
    befe = g.get("_befe_res")
    score = int(getattr(befe, "reliability", 0) or 0)
    high = int(g.get("_rel_high_threshold", 60))
    moderate = int(g.get("_rel_moderate_threshold", 35))
    rel = reliability_label(language, score, high, moderate)
    posterior = getattr(befe, "posterior", None)
    if posterior is None:
        posterior = res.get("p_per_transfer")
    lo = getattr(befe, "ci_low", None)
    hi = getattr(befe, "ci_high", None)

    corridor = "—" if lo is None or hi is None else f"{lo*100:.1f}–{hi*100:.1f}%"
    st.markdown(
        f"""
        <div class="clinical-hero">
          <div class="clinical-kicker">IVF Digital Twin · Clinical synthesis</div>
          <div class="clinical-value">{_pct(posterior)}</div>
          <div class="clinical-title">{tr(language, 'main_outcome')}</div>
          <div class="clinical-row">
            <div class="clinical-tile"><div class="clinical-label">{tr(language,'per_transfer')}</div>
              <div class="clinical-number">{_pct(res.get('p_per_transfer'))}</div></div>
            <div class="clinical-tile"><div class="clinical-label">{tr(language,'viable_cycle')}</div>
              <div class="clinical-number">{_pct(res.get('p_cum_if_viable'))}</div></div>
            <div class="clinical-tile"><div class="clinical-label">{tr(language,'whole_cycle')}</div>
              <div class="clinical-number">{_pct(res.get('p_overall_cycle'))}</div></div>
          </div>
          <div class="clinical-corridor"><b>{tr(language,'clinical_corridor')}:</b> {corridor}
            &nbsp; · &nbsp; <b>{tr(language,'reliability')}:</b> {rel} ({score}/100)<br>
            <span>{tr(language,'corridor_note')}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _context(g: dict, language: str) -> None:
    bmi = g.get("bmi")
    chips = [
        f"{tr(language,'age')}: {g.get('age','—')}",
        f"{tr(language,'amh')}: {_num(g.get('amh'),2)}",
        f"{tr(language,'afc')}: {g.get('afc','—')}",
        f"BMI: {_num(bmi,1)}",
        f"{tr(language,'attempt')}: {g.get('attempt','—')}",
    ]
    st.markdown(
        '<div class="clinical-context">' +
        ''.join(f'<span class="clinical-chip">{x}</span>' for x in chips) +
        '</div>', unsafe_allow_html=True,
    )


def _risk_cards(g: dict, language: str) -> None:
    res = g.get("res") or {}
    p_cancel = res.get("p_cancel_risk")
    if p_cancel is None:
        arr = np.asarray(res.get("sim_okk", []))
        p_cancel = float(np.mean(arr == 0)) if arr.size else None
    p_no_blast = (res.get("empty") or {}).get("p_no_blast")
    ca = res.get("cluster_analysis") or {}
    probs = ca.get("cluster_probs") or {}
    top = max(probs, key=probs.get) if probs else None
    names_ru = {0: "Стандартный", 1: "Сниженный", 2: "Высокий"}
    names_en = {0: "Standard", 1: "Reduced", 2: "High"}
    response = (names_en if language == "English" else names_ru).get(top, "—")
    response_prob = probs.get(top) if top is not None else None

    cols = st.columns(3)
    cards = [
        (_pct(p_no_blast), tr(language, "no_blast")),
        (_pct(p_cancel), tr(language, "cancel")),
        (response, f"{tr(language,'response')} · {_pct(response_prob)}"),
    ]
    for col, (value, label) in zip(cols, cards):
        col.markdown(
            f'<div class="risk-card"><div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>', unsafe_allow_html=True,
        )


def _pipeline(g: dict, language: str) -> None:
    res = g.get("res") or {}
    labels = (["Oocytes", "MII", "2PN", "Blastocysts", "Good quality", "Euploid"]
              if language == "English" else
              ["Ооциты", "MII", "2PN", "Бластоцисты", "Хор. качества", "Эуплоидные"])
    medians = [res.get("okk_med"), res.get("mii_med"), res.get("pn2_med"),
               res.get("blasts_med"), res.get("good_med"), res.get("euploid_med")]
    tiles = [(label, _num(value)) for label, value in zip(labels, medians)]
    st.markdown(
        '<div class="clinical-row pipeline-row" style="grid-template-columns:repeat(6,minmax(0,1fr))">' +
        ''.join(
            f'<div class="clinical-tile"><div class="clinical-label">{label}</div>'
            f'<div class="clinical-number">{value}</div></div>' for label, value in tiles
        ) + '</div>', unsafe_allow_html=True,
    )

    arrays = [res.get("sim_okk"), res.get("sim_mii"), res.get("sim_pn2"),
              res.get("sim_blasts"), res.get("sim_good"), res.get("sim_euploid")]
    if all(a is not None for a in arrays):
        exp_label = "Stage distributions (violin)" if language == "English" else "Распределения по стадиям (violin)"
        with st.expander(exp_label, expanded=False):
            fig = go.Figure()
            colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["green"],
                      PALETTE["amber"], PALETTE["orange"], PALETTE["purple"]]
            for label, arr, color in zip(labels, arrays, colors):
                fig.add_trace(go.Violin(
                    y=np.asarray(arr), name=label, box_visible=True, meanline_visible=False,
                    points=False, fillcolor=_rgba(color, .24), opacity=.85,
                    line=dict(color=_rgba(color, .88), width=1.5),
                ))
            fig.update_layout(showlegend=False)
            fig.update_yaxes(title="Count" if language == "English" else "Количество")
            st.plotly_chart(_plot_theme(fig, height=390), use_container_width=True)


def _attempt_chart(g: dict, language: str) -> None:
    res = g.get("res") or {}
    curve = res.get("attempt_curve") or {}
    _attempts_raw = curve.get("attempts")
    attempts = list(_attempts_raw) if _attempts_raw is not None else []
    if not attempts:
        st.info("Attempt trajectory is unavailable." if language == "English" else
                "Траектория по попыткам недоступна.")
        return

    p_lo = list(curve.get("p_lo")) if curve.get("p_lo") is not None else []
    p_hi = list(curve.get("p_hi")) if curve.get("p_hi") is not None else []
    p_mean = list(curve.get("p_mean")) if curve.get("p_mean") is not None else []
    p_kat = list(curve.get("p_nn_raw")) if curve.get("p_nn_raw") is not None else []
    fig = go.Figure()
    if len(p_lo) == len(attempts) and len(p_hi) == len(attempts):
        fig.add_trace(go.Scatter(
            x=attempts + attempts[::-1],
            y=[v*100 for v in p_hi] + [v*100 for v in p_lo[::-1]],
            fill="toself", fillcolor=_rgba(PALETTE["blue"], .13),
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
        ))
    if len(p_kat) == len(attempts):
        fig.add_trace(go.Scatter(
            x=attempts, y=[v*100 for v in p_kat], mode="lines+markers",
            name="KAT", line=dict(color=PALETTE["purple"], width=2.4, dash="dot"),
            marker=dict(size=7, color=PALETTE["purple"]),
        ))
    if len(p_mean) == len(attempts):
        fig.add_trace(go.Scatter(
            x=attempts, y=[v*100 for v in p_mean], mode="lines+markers+text",
            name="KAT + clinical pipeline",
            line=dict(color=PALETTE["navy"], width=3),
            marker=dict(size=10, color=PALETTE["blue"], line=dict(color="white", width=1.5)),
            text=[f"{v*100:.0f}%" for v in p_mean], textposition="top center",
        ))
    fig.update_xaxes(title="Attempt" if language == "English" else "Номер попытки",
                     tickmode="array", tickvals=attempts)
    fig.update_yaxes(title="Pregnancy probability (%)" if language == "English"
                     else "Вероятность беременности (%)")
    st.plotly_chart(_plot_theme(fig, height=410), use_container_width=True)
    st.caption(tr(language, "attempt_curve_note"))


def _banking(g: dict, language: str, *, include_trp: bool = True) -> None:
    eb = g.get("_eb") or {}
    if not eb:
        st.info("Banking estimate is unavailable." if language == "English" else
                "Оценка банкинга недоступна.")
        return
    need70 = (eb.get("euploid_for_preg") or {}).get(0.70)
    mii_for_70 = None
    if need70 is not None:
        k = min(need70, max(eb.get("k_targets") or [1]))
        mii_for_70 = ((eb.get("mii_table") or {}).get(k) or {}).get(0.80)
    cols = st.columns(3)
    values = [
        (str(mii_for_70 or ">200"), tr(language, "mii_target")),
        (_pct(eb.get("p_per_mii")), tr(language, "euploid_per_mii")),
        (_num((eb.get("forward_at_median") or {}).get("mean"), 1),
         "Expected euploid blastocysts" if language == "English" else "Ожидаемо эуплоидных бластоцист"),
    ]
    for col, (value, label) in zip(cols, values):
        col.markdown(
            f'<div class="risk-card"><div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>', unsafe_allow_html=True,
        )

    if not include_trp:
        return

    st.markdown(f"#### {tr(language, 'trp_title')}")
    TRPInput = g.get("_TRPInput")
    PastCycle = g.get("_PastCycle")
    compute_trp = g.get("_compute_trp")
    if TRPInput is None or PastCycle is None or compute_trp is None:
        st.info("TRP module is unavailable." if language == "English" else
                "Модуль TRP недоступен.")
        return

    kat_base = g.get("_p_kat_raw")
    _metric_cards = st.columns(1)
    _metric_cards[0].markdown(
        f'<div class="risk-card"><div class="value">{_pct(kat_base)}</div>'
        f'<div class="label">{tr(language, "trp_kat_anchor")}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(tr(language, "trp_kat_note"))

    with st.expander(tr(language, "trp_settings"), expanded=False):
        c1, c2, c3 = st.columns(3)
        max_cycles = c1.slider(tr(language, "trp_cycles"), 1, 10, 6,
                               key="brief_trp_cycles")
        desired_children = c2.radio(tr(language, "trp_children"), [1, 2],
                                    horizontal=True, key="brief_trp_children")
        interval_months = c3.slider(tr(language, "trp_interval"), 2, 12, 3,
                                    key="brief_trp_interval")

        c4, c5, c6 = st.columns(3)
        amh_min = c4.number_input(tr(language, "trp_amh_min"), value=0.10,
                                  step=0.05, format="%.2f", key="brief_trp_amh_min")
        age_max = c5.number_input(tr(language, "trp_age_max"), value=45.0,
                                  step=0.5, format="%.1f", key="brief_trp_age_max")
        n_trajectories = c6.select_slider(
            tr(language, "trp_precision"), options=[1000, 2000, 5000, 10000],
            value=5000, key="brief_trp_trajectories",
        )

        st.markdown(f"**{tr(language, 'trp_past')}**")
        n_past = st.number_input(tr(language, "trp_past_count"), 0, 5, 0,
                                 key="brief_trp_past_count")
        past_cycles = []
        past_signature = []
        outcome_options = [tr(language, "trp_no_data"),
                           tr(language, "trp_success"),
                           tr(language, "trp_failure")]
        for i in range(int(n_past)):
            st.caption(f"{tr(language, 'trp_past_attempt')} #{i + 1}")
            pc1, pc2, pc3, pc4, pc5 = st.columns(5)
            default_age = max(20.0, float(g.get("age")) - (int(n_past) - i))
            age_pc = pc1.number_input(tr(language, "trp_past_age"), 20.0, 50.0,
                                      default_age, step=0.5,
                                      key=f"brief_trp_pc_age_{i}")
            okk_pc = pc2.number_input(tr(language, "trp_past_oocytes"), 0, 60, 8,
                                      key=f"brief_trp_pc_okk_{i}")
            blasts_pc = pc3.number_input(tr(language, "trp_past_blasts"), 0, 30, 2,
                                         key=f"brief_trp_pc_blasts_{i}")
            amh_pc = pc4.number_input(tr(language, "trp_past_amh"), 0.0, 20.0,
                                      float(g.get("amh")), step=0.1,
                                      key=f"brief_trp_pc_amh_{i}")
            outcome_text = pc5.selectbox(tr(language, "trp_past_outcome"),
                                         outcome_options,
                                         key=f"brief_trp_pc_outcome_{i}")
            outcome_value = (None if outcome_text == outcome_options[0]
                             else 1 if outcome_text == outcome_options[1] else 0)
            past_cycles.append(PastCycle(
                age_at_cycle=float(age_pc), okk_actual=int(okk_pc),
                blasts_actual=int(blasts_pc), outcome=outcome_value,
                amh_at_cycle=float(amh_pc), cycle_index=i + 1,
            ))
            past_signature.append((float(age_pc), int(okk_pc), int(blasts_pc),
                                   float(amh_pc), outcome_value))

    trp_key = (
        float(g.get("age", 0)), round(float(g.get("amh", 0)), 3),
        int(g.get("afc", 0)), round(float(g.get("bmi", 0)), 2),
        int(max_cycles), int(desired_children), int(interval_months),
        round(float(amh_min), 3), round(float(age_max), 2), int(n_trajectories),
        tuple(past_signature), round(float(kat_base), 6) if kat_base is not None else None,
    )
    has_result = "_brief_trp_result_v2" in st.session_state
    button_label = tr(language, "trp_update") if has_result else tr(language, "trp_calculate")
    run_trp = st.button(button_label, key="brief_trp_run", type="primary",
                        disabled=kat_base is None)
    if kat_base is None:
        st.warning("KAT probability is unavailable; TRP cannot be calculated."
                   if language == "English" else
                   "Вероятность KAT недоступна — расчёт TRP невозможен.")
    if run_trp:
        inp = TRPInput(
            age=float(g.get("age")), amh=float(g.get("amh")),
            afc=int(g.get("afc")), bmi=float(g.get("bmi")),
            past_cycles=past_cycles,
            max_future_cycles=int(max_cycles),
            desired_children=int(desired_children),
            cycle_interval_mo=float(interval_months),
            amh_min=float(amh_min), age_max=float(age_max),
            n_trajectories=int(n_trajectories),
            p_base_override=float(kat_base),
        )
        spinner = ("Calculating the reproductive horizon…" if language == "English"
                   else "Расчёт репродуктивного горизонта…")
        with st.spinner(spinner):
            st.session_state["_brief_trp_result_v2"] = compute_trp(inp)
        st.session_state["_brief_trp_key_v2"] = trp_key

    trp_res = st.session_state.get("_brief_trp_result_v2")
    if trp_res is None:
        return
    if st.session_state.get("_brief_trp_key_v2") != trp_key:
        st.info(tr(language, "trp_changed"))
    expected_cycles = getattr(trp_res, "expected_cycles_to_success", math.nan)
    expected_label = (tr(language, "not_calculable") if not math.isfinite(expected_cycles)
                      else f"{expected_cycles:.1f}")
    trp_values = [
        (_pct(getattr(trp_res, "p_success_total", None)), tr(language, "trp_total")),
        (f"{getattr(trp_res, 'window_years_p50', math.nan):.1f} {tr(language, 'years')}",
         tr(language, "trp_window")),
        (expected_label, tr(language, "trp_expected_cycles")),
        (_pct(getattr(trp_res, "p_window_closes_first", None)),
         tr(language, "trp_closes_first")),
    ]
    cols = st.columns(4)
    for col, (value, label) in zip(cols, trp_values):
        col.markdown(
            f'<div class="risk-card"><div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>', unsafe_allow_html=True,
        )


def _gat(g: dict, language: str) -> None:
    gnn_result = g.get("_gnn_result") or {}
    if not gnn_result.get("available"):
        st.info("GAT graph is unavailable for this calculation." if language == "English"
                else "График GAT недоступен для этого расчёта.")
        return

    raw = g.get("_p_gnn_raw")
    ensemble = g.get("_p_gnn_ens")
    kat = g.get("_p_kat_raw")
    cols = st.columns(3)
    for col, (value, label) in zip(cols, [
        (_pct(raw), "GAT"), (_pct(kat), "KAT"),
        (_pct(ensemble), "GAT + KAT"),
    ]):
        col.markdown(
            f'<div class="risk-card"><div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>', unsafe_allow_html=True,
        )

    fig = st.session_state.get("_pdf_fig_gnn")
    if fig is None:
        builder = g.get("_build_gnn_figure")
        if builder is not None:
            try:
                fig = builder(gnn_result, gnn_prob=raw, ensemble_prob=ensemble)
                styler = g.get("_apply_gnn_style")
                fig = styler(fig) if styler is not None else fig
                st.session_state["_pdf_fig_gnn"] = fig
            except Exception:
                fig = None
    if fig is not None:
        if language == "English":
            fig = copy.deepcopy(fig)
            replacements = {
                "Граф клинических соседей": "Clinical-neighbour graph",
                "Распределение GNN-вероятностей": "GNN probability distribution",
                "Пациентка": "Patient",
                "Соседи": "Neighbours",
                "Сосед": "Neighbour",
                "P(беременность)": "P(pregnancy)",
                "P(бер.)": "P(pregnancy)",
                "Вероятность беременности": "Pregnancy probability",
                "Медиана": "Median",
                "Ансамбль": "Ensemble",
            }

            def translate_text(value):
                if not isinstance(value, str):
                    return value
                for source, target in replacements.items():
                    value = value.replace(source, target)
                return value

            for trace in fig.data:
                if getattr(trace, "name", None):
                    trace.name = translate_text(trace.name)
                if getattr(trace, "hovertemplate", None):
                    trace.hovertemplate = translate_text(trace.hovertemplate)
                if getattr(trace, "text", None) is not None:
                    trace.text = (translate_text(trace.text) if isinstance(trace.text, str)
                                  else tuple(translate_text(v) for v in trace.text))
                if getattr(trace, "hovertext", None) is not None:
                    trace.hovertext = (translate_text(trace.hovertext)
                                       if isinstance(trace.hovertext, str)
                                       else tuple(translate_text(v) for v in trace.hovertext))
            for annotation in (fig.layout.annotations or []):
                annotation.text = translate_text(annotation.text)
            if fig.layout.title and fig.layout.title.text:
                fig.layout.title.text = translate_text(fig.layout.title.text)
            for axis in list(fig.select_xaxes()) + list(fig.select_yaxes()):
                if axis.title and axis.title.text:
                    axis.title.text = translate_text(axis.title.text)
            for trace in fig.data:
                marker = getattr(trace, "marker", None)
                colorbar = getattr(marker, "colorbar", None) if marker is not None else None
                if colorbar is not None and colorbar.title and colorbar.title.text:
                    colorbar.title.text = translate_text(colorbar.title.text)
        st.plotly_chart(fig, use_container_width=True)
    st.caption(tr(language, "gat_note"))


def _evidence(g: dict, language: str) -> None:
    res = g.get("res") or {}
    befe = g.get("_befe_res")
    p_kat = g.get("_p_kat_raw")
    p_gat = g.get("_p_gnn_ens")
    csdi = st.session_state.get("csdi_result") or {}
    rows = [
        (tr(language, "kat"), _pct(p_kat), p_kat is not None),
        (tr(language, "gat"), _pct(p_gat), p_gat is not None),
        (tr(language, "csdi"), _pct(csdi.get("P_pregnancy")), bool(csdi)),
        ("BEFE", _pct(getattr(befe, "posterior", None)), befe is not None),
    ]
    for label, value, ok in rows:
        status = tr(language, "available") if ok else tr(language, "unavailable")
        cls = "model-ok" if ok else "model-off"
        st.markdown(
            f'<div class="model-line"><span>{label}</span><span><b>{value}</b> · '
            f'<span class="{cls}">{status}</span></span></div>', unsafe_allow_html=True,
        )
    st.info(tr(language, "corridor_note"))
    with st.expander(tr(language, "technical_evidence"), expanded=False):
        st.write({
            "KAT": "available" if g.get("nn_model") is not None else "unavailable",
            "CSDI": "available" if g.get("csdi_model") is not None else "unavailable",
            "GAT": "available" if (g.get("_gnn_bundle") or {}).get("available") else "unavailable",
            "BEFE reliability score": getattr(befe, "reliability", None),
            "BEFE corridor source": getattr(befe, "ci_source", None),
            "MC iterations": len(np.asarray(res.get("sim_okk", []))),
        })


def render(g: dict) -> None:
    language = g.get("_LANG", "Русский")
    _inject_css()

    nav_col, _ = st.columns([2, 8])
    with nav_col:
        if st.button(tr(language, "detailed_report"), key="_to_full_report",
                     use_container_width=True):
            st.session_state["_view_mode"] = "Detailed report"
            st.rerun()

    _context(g, language)
    _hero(g, language)

    tabs = st.tabs([
        tr(language, "section_overview"),
        tr(language, "section_embryology"),
        tr(language, "section_pregnancy"),
        tr(language, "section_banking"),
        tr(language, "section_gat"),
    ])
    with tabs[0]:
        st.subheader(tr(language, "risks"))
        _risk_cards(g, language)
    with tabs[1]:
        st.subheader(tr(language, "cycle_path"))
        _pipeline(g, language)
    with tabs[2]:
        st.subheader(tr(language, "attempt_curve"))
        _attempt_chart(g, language)
    with tabs[3]:
        st.subheader(tr(language, "banking"))
        _banking(g, language)
    with tabs[4]:
        st.subheader(tr(language, "gat_title"))
        _gat(g, language)

    st.caption(tr(language, "support_only"))
