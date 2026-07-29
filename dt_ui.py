# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
dt_ui.py — IVF Digital Twin v7.0 · Дизайн-система для Streamlit

Единая библиотека компонентов:
  inject_css()          — глобальный CSS (вызвать один раз после st.set_page_config)
  tab_header()          — заголовок вкладки с иконкой слоя и описанием
  section_header()      — секционный заголовок с акцентной полосой
  card_start() / end()  — HTML-обёртка карточки
  metric_row()          — строка метрик (2–4 плитки)
  key_number()          — большая цифра + подпись + опциональный бейдж
  ci_bar()              — горизонтальная линейка с Prior / CI / Posterior
  badge()               — цветной бейдж: success / warning / danger / info / neutral
  ood_strip()           — строка из трёх OOD-статусов
  reliability_bar()     — полоса надёжности с цветовой индикацией
  result_box()          — синий/зелёный/жёлтый информационный блок
  verification_grid()   — сетка 2×2 для verification-сигналов
  layer_tag()           — тег слоя (L1, L2, … L7) в шапке вкладки
"""

from __future__ import annotations
import streamlit as st


# ────────────────────────────────────────────────────────────────────────────
# Палитра (совпадает с Plotly-цветами в app.py)
# ────────────────────────────────────────────────────────────────────────────
NAVY   = "#0F4C75"
BLUE   = "#1B4F72"
BLUE2  = "#2471A3"
TEAL   = "#0D7377"
GREEN  = "#1E8449"
GREEN_BG  = "#EAF3DE"
GREEN_BD  = "#A9D18E"
AMBER  = "#B7770D"
AMBER_BG  = "#FEF3C7"
AMBER_BD  = "#F9C74F"
RED    = "#C0392B"
RED_BG = "#FDECEA"
RED_BD = "#E57373"
GREY   = "#5F6B7A"
GREY_BG = "#F4F6F8"
CARD_BG = "#FFFFFF"
CARD_BD = "#D8E4ED"
SECONDARY_BG = "#F0F5FA"


# ────────────────────────────────────────────────────────────────────────────
# 1.  ГЛОБАЛЬНЫЙ CSS
# ────────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
/* ── Основа ── */
.main { background-color: #F4F7FA; }
.block-container { padding-top: 1.2rem !important; }

/* ── Заголовки ── */
h1 { color: #0F4C75 !important; font-weight: 700 !important; letter-spacing: -0.4px; }
h2 { color: #1B4F72 !important; font-weight: 600 !important;
     border-bottom: 1px solid #D0E4F0; padding-bottom: 4px; }
h3 { color: #154360 !important; font-weight: 600 !important; }

/* ── Вкладки Streamlit ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 3px;
  background: #EEF3F6;
  border: 1px solid #DDE7EE;
  border-radius: 9px;
  padding: 4px;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 6px !important;
  padding: 7px 13px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  color: #596977 !important;
  background: transparent !important;
  border: none !important;
}
.stTabs [aria-selected="true"] {
  background: #FFFFFF !important;
  color: #163E59 !important;
  box-shadow: 0 1px 2px rgba(25,50,70,0.10) !important;
}

/* ── Метрики ── */
[data-testid="metric-container"] {
  background: #FFFFFF;
  border: 0.5px solid #D4E4F0;
  border-radius: 10px;
  padding: 12px 16px !important;
  border-left: 3px solid #1B4F72 !important;
}
[data-testid="metric-container"] label {
  font-size: 12px !important;
  color: #5A6B7B !important;
  font-weight: 500 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-size: 22px !important;
  font-weight: 700 !important;
  color: #0F4C75 !important;
}

/* ── Кнопки ── */
.stButton > button {
  border-radius: 8px !important;
  font-weight: 500 !important;
}
.stButton > button[kind="primary"] {
  background: #1B4F72 !important;
  border-color: #1B4F72 !important;
}

/* ── Сайдбар ── */
[data-testid="stSidebar"] { background: #EEF4F9 !important; }
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
  border-bottom: 1px solid #C9DAEA !important;
}

/* ── Таблицы ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Expander ── */
.streamlit-expanderHeader {
  background: #EBF2F8 !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
  color: #1B4F72 !important;
}

/* ── Именованные классы ── */
.dt-disclaimer,
.disclaimer {
  background: #FFF8E1;
  border-left: 4px solid #F9A825;
  padding: 10px 14px;
  border-radius: 6px;
  font-size: 0.85em;
  color: #5C4A00;
  margin-bottom: 1rem;
}
.result-box {
  background: #EAF4FB;
  border-left: 5px solid #1B4F72;
  padding: 14px 18px;
  border-radius: 8px;
  margin: 10px 0;
}
.diff-box {
  background: #E8F5E9;
  border-left: 5px solid #2E7D32;
  padding: 14px 18px;
  border-radius: 8px;
  margin: 10px 0;
}
.diff-warn {
  background: #FFF8E1;
  border-left: 5px solid #F9A825;
  padding: 14px 18px;
  border-radius: 8px;
  margin: 10px 0;
}
.cluster-c0 { background: #E3F2FD; border-left: 4px solid #1976D2;
              padding: 10px; border-radius: 6px; }
.cluster-c1 { background: #FFEBEE; border-left: 4px solid #C62828;
              padding: 10px; border-radius: 6px; }
.cluster-c2 { background: #E8F5E9; border-left: 4px solid #2E7D32;
              padding: 10px; border-radius: 6px; }
.ks-pass { color: #2E7D32; font-weight: bold; }
.ks-fail { color: #C62828; font-weight: bold; }

/* ── Карточки dt_ui ── */
.dt-card {
  background: #FFFFFF;
  border: 0.5px solid #D4E4F0;
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 14px;
}
.dt-card-header {
  padding: 12px 18px 10px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.dt-card-body {
  padding: 14px 18px 16px;
}
.dt-metric-tile {
  background: #F0F5FA;
  border-radius: 9px;
  padding: 10px 13px;
  min-width: 0;
}
.dt-metric-tile.accent {
  background: #E6F0F8;
  border: 0.5px solid #B8D0E8;
}
.dt-metric-tile.highlight {
  background: #EAF3DE;
  border: 0.5px solid #A9D18E;
}
.dt-metric-label {
  font-size: 11px;
  color: #5A6B7B;
  font-weight: 500;
  margin-bottom: 3px;
}
.dt-metric-value {
  font-size: 22px;
  font-weight: 700;
  color: #0F4C75;
  line-height: 1.1;
}
.dt-section-label {
  font-size: 13px;
  font-weight: 600;
  color: #1B4F72;
  border-left: 3px solid #1B4F72;
  padding-left: 9px;
  margin: 12px 0 8px 0;
  line-height: 1.3;
}
.dt-badge {
  display: inline-block;
  font-size: 11px;
  font-weight: 600;
  padding: 2px 10px;
  border-radius: 20px;
}
.dt-badge-success  { background: #EAF3DE; color: #1E6B2E; }
.dt-badge-warning  { background: #FEF3C7; color: #8A5C00; }
.dt-badge-danger   { background: #FDECEA; color: #9B1C1C; }
.dt-badge-info     { background: #E6F0F8; color: #0F4C75; }
.dt-badge-neutral  { background: #F0F5FA; color: #4A6070; }
.dt-ood-chip {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 6px 10px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 500;
}
.dt-ood-ok   { background: #EAF3DE; color: #1E6B2E; }
.dt-ood-warn { background: #FEF3C7; color: #8A5C00; }
.dt-verif-cell {
  background: #F0F5FA;
  border-radius: 9px;
  padding: 8px 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
}
.dt-verif-label { color: #5A6B7B; }
.dt-verif-value { font-weight: 600; }
.dt-tab-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 0 14px 0;
  border-bottom: 1px solid #D0E4F0;
  margin-bottom: 14px;
}
.dt-tab-icon {
  min-width: 34px;
  height: 34px;
  border: 1px solid #C7D8E4;
  border-radius: 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #F8FBFD;
  color: #1B4F72;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0;
  line-height: 1;
}
.dt-tab-title {
  font-size: 17px;
  font-weight: 700;
  color: #0F4C75;
  margin: 0;
  line-height: 1.2;
}
.dt-tab-desc {
  font-size: 12px;
  color: #5A6B7B;
  margin: 0;
}
.dt-layer-tag {
  margin-left: auto;
  background: #E6F0F8;
  color: #1B4F72;
  border: 1px solid #B8D0E8;
  font-size: 11px;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 20px;
  letter-spacing: 0.03em;
}
.dt-result-hero {
  background: #FFFFFF;
  border: 1px solid #B8D0E8;
  border-radius: 12px;
  overflow: hidden;
  margin: 4px 0 14px 0;
  box-shadow: 0 1px 2px rgba(15,76,117,0.05);
}
.dt-result-hero-top {
  background: #E8F1F8;
  border-bottom: 1px solid #B8D0E8;
  padding: 13px 18px 11px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.dt-result-kicker {
  font-size: 10px;
  letter-spacing: 0.08em;
  color: #6B8499;
  text-transform: uppercase;
  margin-bottom: 1px;
  font-weight: 700;
}
.dt-result-title {
  font-size: 15px;
  font-weight: 800;
  color: #1B4F72;
}
.dt-result-body {
  padding: 16px 18px 18px;
}
.dt-result-main {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 13px;
}
.dt-result-value {
  font-size: 48px;
  font-weight: 800;
  color: #1B4F72;
  line-height: 0.95;
}
.dt-result-sub {
  font-size: 13px;
  color: #5A6B7B;
  margin-top: 5px;
}
.dt-result-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}
@media (max-width: 900px) {
  .dt-result-main { display: block; }
  .dt-result-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 560px) {
  .dt-result-grid { grid-template-columns: 1fr; }
}
</style>
"""


def inject_css() -> None:
    """Вызвать один раз сразу после st.set_page_config(). Заменяет старый блок CSS."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# 2.  КОМПОНЕНТЫ
# ────────────────────────────────────────────────────────────────────────────

def tab_header(icon: str, title: str, desc: str = "", layer_tag: str = "") -> None:
    """Заголовок вкладки с иконкой, описанием и необязательным тегом слоя.

    Пример:
        tab_header("L1", "Pipeline", "Стохастический каскад S1–S6b", "L1")
    """
    tag_html = (
        f'<span class="dt-layer-tag">{layer_tag}</span>' if layer_tag else ""
    )
    icon_html = f'<span class="dt-tab-icon">{icon}</span>' if icon else ""
    st.markdown(
        f"""
        <div class="dt-tab-header">
          {icon_html}
          <div>
            <p class="dt-tab-title">{title}</p>
            {"" if not desc else f'<p class="dt-tab-desc">{desc}</p>'}
          </div>
          {tag_html}
        </div>""",
        unsafe_allow_html=True,
    )


def section_header(text: str) -> None:
    """Секционный подзаголовок с синей акцентной полосой слева."""
    st.markdown(
        f'<p class="dt-section-label">{text}</p>',
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "info") -> str:
    """Возвращает HTML-строку с бейджем. kind: success | warning | danger | info | neutral.

    Используется внутри st.markdown(..., unsafe_allow_html=True).
    """
    return f'<span class="dt-badge dt-badge-{kind}">{text}</span>'


def metric_tile(label: str, value: str, style: str = "") -> str:
    """HTML одной плитки метрики. style: '' | 'accent' | 'highlight'."""
    cls = f"dt-metric-tile {style}".strip()
    return (
        f'<div class="{cls}">'
        f'  <div class="dt-metric-label">{label}</div>'
        f'  <div class="dt-metric-value">{value}</div>'
        f"</div>"
    )


def metric_row(items: list[tuple[str, str, str]], gap: int = 10) -> None:
    """Строка плиток метрик.

    items: [(label, value, style), ...]   style: '' | 'accent' | 'highlight'
    Пример:
        metric_row([
            ("Приор (L1)", "52%", ""),
            ("Доказательство (L3/L6)", "61%", "accent"),
            ("Posterior", "57%", "highlight"),
        ])
    """
    tiles = "".join(metric_tile(lbl, val, sty) for lbl, val, sty in items)
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat({len(items)},1fr);'
        f'gap:{gap}px;margin-bottom:12px">{tiles}</div>',
        unsafe_allow_html=True,
    )


def key_number(
    label: str,
    value: str,
    sub: str = "",
    badge_text: str = "",
    badge_kind: str = "info",
    color: str = BLUE,
) -> None:
    """Большая цифра в карточке — используется для главного результата."""
    badge_html = (
        f' &nbsp;{badge(badge_text, badge_kind)}' if badge_text else ""
    )
    sub_html = (
        f'<div style="font-size:13px;color:{GREY};margin-top:3px">{sub}</div>'
        if sub
        else ""
    )
    st.markdown(
        f"""
        <div style="background:{SECONDARY_BG};border:0.5px solid {CARD_BD};
        border-radius:12px;padding:16px 20px;margin-bottom:12px">
          <div style="font-size:12px;color:{GREY};font-weight:600;margin-bottom:4px">
            {label}{badge_html}
          </div>
          <div style="font-size:40px;font-weight:700;color:{color};line-height:1.05">
            {value}
          </div>
          {sub_html}
        </div>""",
        unsafe_allow_html=True,
    )


def result_summary_card(
    *,
    title: str,
    value: str,
    subtitle: str = "",
    badge_text: str = "L7",
    badge_kind: str = "info",
    secondary: list[tuple[str, str, str]] | None = None,
) -> None:
    """Главная карточка результатов: итоговая вероятность + компактные опорные метрики."""
    secondary = secondary or []
    tiles = "".join(metric_tile(lbl, val, sty) for lbl, val, sty in secondary)
    st.markdown(
        f"""
        <div class="dt-result-hero">
          <div class="dt-result-hero-top">
            <div>
              <div class="dt-result-kicker">Digital Twin · Итоговый posterior</div>
              <div class="dt-result-title">{title}</div>
            </div>
            {badge(badge_text, badge_kind)}
          </div>
          <div class="dt-result-body">
            <div class="dt-result-main">
              <div>
                <div class="dt-result-value">{value}</div>
                {"" if not subtitle else f'<div class="dt-result-sub">{subtitle}</div>'}
              </div>
            </div>
            {"" if not tiles else f'<div class="dt-result-grid">{tiles}</div>'}
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


def ci_bar(
    prior_pct: float,
    posterior_pct: float,
    ci_low_pct: float,
    ci_high_pct: float,
) -> None:
    """Горизонтальная шкала 0–100 с отметками Prior и Posterior + CI-полосой.

    Все аргументы в процентах (0–100).
    """
    w_ci  = max(ci_high_pct - ci_low_pct, 1)
    left_ci = ci_low_pct
    # Prior marker width clamp
    pr_left = max(0, min(prior_pct, 98))
    po_left = max(0, min(posterior_pct, 98))

    st.markdown(
        f"""
        <div style="margin:10px 0 4px">
          <div style="position:relative;height:8px;background:#E8EFF5;border-radius:4px">
            <div style="position:absolute;left:{left_ci:.1f}%;top:0;height:100%;
              width:{w_ci:.1f}%;background:#B5D4F4;border-radius:4px;opacity:0.7"></div>
            <div style="position:absolute;left:{po_left:.1f}%;top:-4px;width:3px;
              height:16px;background:{NAVY};border-radius:2px"></div>
            <div style="position:absolute;left:{pr_left:.1f}%;top:-2px;width:2px;
              height:12px;background:#888;border-radius:2px;opacity:0.6"></div>
          </div>
          <div style="display:flex;justify-content:space-between;
            font-size:10px;color:{GREY};margin-top:5px">
            <span>0%</span>
            <span style="color:#888">Приор {prior_pct:.0f}%</span>
            <span style="color:{NAVY};font-weight:600">Posterior {posterior_pct:.0f}%</span>
            <span>100%</span>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


def reliability_bar(score: int, band: str) -> None:
    """Полоса надёжности с цветовой индикацией.

    band: 'High' | 'Moderate' | 'Low'
    """
    band_ru   = {"High": "Высокая", "Moderate": "Умеренная", "Low": "Ограниченная"}
    bar_color = {"High": "#1E8449", "Moderate": "#D97706", "Low": "#C0392B"}
    col  = bar_color.get(band, BLUE)
    ru   = band_ru.get(band, band)
    pct  = min(score, 100)
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin:8px 0">
          <span style="font-size:12px;color:{GREY};font-weight:500;min-width:80px">
            Надёжность</span>
          <div style="flex:1;height:6px;background:#E8EFF5;border-radius:4px;overflow:hidden">
            <div style="height:100%;width:{pct}%;background:{col};border-radius:4px"></div>
          </div>
          <span style="font-size:14px;font-weight:700;color:{col};min-width:52px">
            {score}/100</span>
          <span style="font-size:11px;font-weight:600;color:{col};background:{col}22;
            padding:2px 9px;border-radius:20px">{ru}</span>
        </div>""",
        unsafe_allow_html=True,
    )


def ood_strip(clinical: bool, embryology: bool, final: bool) -> None:
    """Три чипа OOD в одну строку."""
    def chip(label: str, ok: bool) -> str:
        kind = "dt-ood-ok" if not ok else "dt-ood-warn"
        icon = "OK" if not ok else "OOD"
        return (
            f'<div class="dt-ood-chip {kind}">'
            f'  <span>{icon}</span>'
            f'  <span>{label}</span>'
            f"</div>"
        )

    st.markdown(
        f"""
        <div style="display:flex;gap:6px;margin:8px 0">
          {chip("OOD клинический", clinical)}
          {chip("OOD эмбриология", embryology)}
          {chip("OOD итоговый", final)}
        </div>""",
        unsafe_allow_html=True,
    )


def verification_grid(items: list[tuple[str, str, str]]) -> None:
    """Сетка 2×N ячеек верификации.

    items: [(label, value, value_color), ...]
    value_color: 'green' | 'amber' | 'red' | ''
    """
    color_map = {
        "green": "#1E8449",
        "amber": "#B7770D",
        "red":   "#C0392B",
        "":      "#1B4F72",
    }
    cells_html = ""
    for label, value, vc in items:
        col = color_map.get(vc, BLUE)
        cells_html += (
            f'<div class="dt-verif-cell">'
            f'  <span class="dt-verif-label">{label}</span>'
            f'  <span class="dt-verif-value" style="color:{col}">{value}</span>'
            f"</div>"
        )
    n = len(items)
    cols = 2 if n > 1 else 1
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat({cols},1fr);'
        f'gap:6px;margin:8px 0">{cells_html}</div>',
        unsafe_allow_html=True,
    )


def result_box(html_content: str, kind: str = "info") -> None:
    """Информационный блок с цветной полосой.

    kind: 'info' (синий) | 'success' (зелёный) | 'warning' (жёлтый) | 'danger' (красный)
    """
    colors = {
        "info":    ("rgba(232,241,248,0.48)", "#8FB1CA", "#405565"),
        "success": ("rgba(232,245,233,0.40)", "#9DBF9D", "#3F6244"),
        "warning": ("rgba(254,243,199,0.30)", "#D8BE74", "#6A5A2E"),
        "danger":  ("rgba(253,236,234,0.30)", "#D1A1A1", "#694747"),
    }
    bg, border, text_col = colors.get(kind, colors["info"])
    st.markdown(
        f'<div style="background:{bg};border:1px solid rgba(184,208,232,0.45);'
        f'border-left:3px solid {border};color:{text_col};font-size:12px;'
        f'line-height:1.45;padding:9px 13px;border-radius:7px;margin:8px 0">'
        f'{html_content}</div>',
        unsafe_allow_html=True,
    )


def befe_card_header(posterior_pct: float, ci_low_pct: float, ci_high_pct: float) -> None:
    """Шапка карточки BEFE — светлая версия без тёмного фона."""
    st.markdown(
        f"""
        <div style="background:#E8F1F8;border:1.5px solid #B8D0E8;
        border-radius:12px 12px 0 0;padding:13px 18px 11px;
        display:flex;align-items:center;justify-content:space-between;margin-bottom:0">
          <div>
            <div style="font-size:10px;letter-spacing:0.08em;color:#6B8499;
              text-transform:uppercase;margin-bottom:1px;font-weight:600">
              Digital Twin · Layer 7</div>
            <div style="font-size:15px;font-weight:700;color:#1B4F72">
              Bayesian Evidence Fusion Engine</div>
          </div>
          <span style="background:#1B4F72;color:#fff;
            font-size:11px;font-weight:600;padding:3px 12px;border-radius:20px">BEFE</span>
        </div>
        <div style="background:#FFFFFF;border:1.5px solid #B8D0E8;border-top:none;
          border-radius:0 0 12px 12px;padding:12px 18px;
          display:flex;align-items:baseline;gap:16px;margin-bottom:12px">
          <span style="font-size:42px;font-weight:700;color:#1B4F72;line-height:1">
            {posterior_pct:.0f}%</span>
          <div>
            <div style="font-size:12px;color:#5A6B7B;font-weight:500">
              Итоговая P(беременность)</div>
            <div style="font-size:13px;color:#2471A3;font-weight:600">
              Исторический коридор клиники: {ci_low_pct:.0f}% – {ci_high_pct:.0f}%</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# 3.  SHORTCUT-ЗАГОЛОВКИ ДЛЯ КАЖДОЙ ВКЛАДКИ
# ────────────────────────────────────────────────────────────────────────────

TAB_META = {
    "pipeline": ("L1", "Pipeline", "Стохастический каскад S1–S6b · Monte Carlo N=5000", "L1"),
    "pregnancy": ("L2", "Беременность", "FORTUNE · KPI · ансамбль на перенос", "L2"),
    "cluster":   ("L4", "Кластер", "Ближайший центроид · 18D z-пространство", "L4"),
    "bayes":     ("L3", "Байес + попытки", "Beta-Binomial posterior · коэффициент убывания", "L3"),
    "risks":     ("", "Риски", "ССЯГ · пустой цикл · ZINB-распределение", "L1"),
    "banking":   ("", "Банкинг", "Модуль Esteves · планирование накопления эуплоидов", ""),
    "diffusion": ("L5", "Diffusion", "CSDI Hybrid v3 · ~15 000 циклов · конформные PI", "L5"),
    "gat":       ("L6", "GAT Graph", "Graph Attention Transformer · граф клинических соседей", "L6"),
    "befe":      ("L7", "BEFE", "Bayesian Evidence Fusion · итоговый posterior", "L7"),
}


def tab_header_by_key(key: str) -> None:
    """Вставить заголовок вкладки по ключу из TAB_META."""
    meta = TAB_META.get(key)
    if meta:
        tab_header(*meta)
