# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
IVF Digital Twin v7.1 — Streamlit Clinical Application
Запуск: streamlit run app.py

Офлайн-лицензирование (RSA + AES-256):
  Лицензия проверяется локально без обращения к серверу.
  Выдача ключей — через generate_license.py (только у разработчика).
"""

import sys, os, warnings, math, csv, uuid as _uuid, json
from datetime import datetime, date
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

# Combined application: redesigned concise clinical summary plus the complete
# original report. All runtime assets live in this project directory.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_DIR = _APP_DIR
for _path in (_APP_DIR, _SOURCE_DIR):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, _APP_DIR)
sys.path.insert(1, _SOURCE_DIR)

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import beta as beta_dist, norm, ks_2samp
from i18n import tr as _translate, reliability_label as _reliability_label

# ── BEFE (L7) — Bayesian Evidence Fusion Engine ──────────────
try:
    from befe import BEFE as _BEFE
    from befe_app import build_befe_result, render_befe_tab
    _BEFE_OK, _BEFE_ERR = True, ""
except Exception as _befe_e:
    _BEFE_OK, _BEFE_ERR = False, str(_befe_e)

# ── Дизайн-система dt_ui ─────────────────────────────────────
try:
    import dt_ui as UI
    _UI_OK = True
except ImportError:
    _UI_OK = False

# ══════════════════════════════════════════════════════════════
#  ЕДИНЫЙ СТИЛЬ ГРАФИКОВ (применяется во всех вкладках и PDF)
# ══════════════════════════════════════════════════════════════

def hex_rgba(h: str, a: float = 1.0) -> str:
    """#RRGGBB → 'rgba(r,g,b,a)' — Plotly не принимает 8-значный HEX."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a:.2f})"


def _apply_gnn_style(fig):
    """
    Применяет единый стиль к фигуре из gnn_predictor.build_gnn_neighborhood_figure.
    Вызывается после построения — т.к. фигура строится во внешнем модуле.
    Фигура — make_subplots(1,2): левая панель = граф, правая = гистограмма.
    """
    if fig is None:
        return fig
    try:
        import copy as _copy
        fig = _copy.deepcopy(fig)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, Arial, sans-serif", size=12, color="#1C2833"),
            margin=dict(l=55, r=30, t=60, b=55),
            legend=dict(
                orientation="h",
                x=0.5, xanchor="center",
                y=1.04, yanchor="bottom",
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#dddddd",
                borderwidth=1,
                font=dict(size=11),
            ),
        )
        # Применяем стиль сетки к обеим осям subplot
        fig.update_xaxes(
            gridcolor="rgba(200,210,220,0.35)",
            zeroline=False,
            tickfont=dict(size=11),
        )
        # ── 2× зум + центрирование пациентки в области точек графа ──────
        # Сдвигаем все scatter-трейсы левой панели так, чтобы звезда пациентки
        # оказалась в (0,0), затем берём фиксированное окно вокруг центра.
        # Часть точек уходит за пределы — это допустимо.
        try:
            import numpy as _np
            def _xax(tr):
                return getattr(tr, "xaxis", None) or "x"
            # 1) находим координаты пациентки (звезда / name='Пациентка')
            pat_x = pat_y = None
            for tr in fig.data:
                if getattr(tr, "type", "") != "scatter" or _xax(tr) != "x":
                    continue
                nm = (getattr(tr, "name", "") or "")
                sym = getattr(getattr(tr, "marker", None), "symbol", None)
                if "Пациент" in nm or sym == "star":
                    xs, ys = tr.x, tr.y
                    if xs is not None and ys is not None and len(xs) and len(ys):
                        pat_x, pat_y = float(xs[0]), float(ys[0])
                        break
            # 2) сдвигаем все scatter левой панели на (-pat_x, -pat_y)
            if pat_x is not None:
                for tr in fig.data:
                    if getattr(tr, "type", "") == "scatter" and _xax(tr) == "x":
                        if tr.x is not None:
                            tr.x = tuple(float(v) - pat_x for v in tr.x)
                        if tr.y is not None:
                            tr.y = tuple(float(v) - pat_y for v in tr.y)
            # 3) фиксированное окно вокруг центра (зум ~1.5×)
            fig.update_xaxes(range=[-0.87, 0.87], row=1, col=1)
            fig.update_yaxes(range=[-0.87, 0.87], row=1, col=1,
                             scaleanchor="x", scaleratio=1)
        except Exception:
            try:
                fig.update_xaxes(range=[-0.87, 0.87], row=1, col=1)
                fig.update_yaxes(range=[-0.87, 0.87], row=1, col=1,
                                 scaleanchor="x", scaleratio=1)
            except Exception:
                pass
        fig.update_yaxes(
            gridcolor="rgba(200,210,220,0.35)",
            zeroline=False,
            tickfont=dict(size=11),
        )
        # Правая панель (гистограмма соседей) — заменяем цвета маркеров на палитру
        # GNN-граф использует цветовую шкалу по вероятности (RdYlGn) — оставляем,
        # но текущую пациентку (звезда) подчёркиваем нашим красным
        for trace in fig.data:
            # Ребра графа — тонкие серые линии
            if hasattr(trace, 'mode') and trace.mode == 'lines' and \
               hasattr(trace, 'line') and trace.line.color is not None:
                if 'rgba' in str(trace.line.color) and trace.line.width and \
                   trace.line.width < 3:
                    trace.line.color = hex_rgba(C["grey"], 0.25)
            # Бары правой панели — стиль histogram
            if hasattr(trace, 'type') and trace.type == 'bar':
                if trace.marker.color is not None and \
                   not isinstance(trace.marker.color, (list, tuple)):
                    trace.marker.color = hex_rgba(C["blue"], 0.70)
                    trace.marker.line = dict(
                        color=hex_rgba(C["blue"], 0.90), width=1.0
                    )
    except Exception:
        pass
    return fig

# ── Цветовые палитры ──────────────────────────────────────────
C = {
    "blue":   "#6F93B7",
    "teal":   "#78AAA5",
    "green":  "#8DBA8D",
    "orange": "#D9A36A",
    "red":    "#C98282",
    "purple": "#A792C6",
    "amber":  "#DDBB72",
    "grey":   "#71808C",
}

# Цвета для стадий воронки / violin (7 стадий)
STAGE_COLORS = ["#6F93B7", "#7EA9C7", "#78AAA5", "#8DBA8D",
                "#C6B27E", "#D9A36A", "#A792C6"]

# Возрастные группы
AGE_COLORS = {
    "<30":   "#6F93B7",
    "30–35": "#8DBA8D",
    "35–38": "#DDBB72",
    "38–41": "#D9A36A",
    ">41":   "#C98282",
}

# Кластеры
CLUSTER_HEX = {0: "#6F93B7", 1: "#C98282", 2: "#8DBA8D"}
CLUSTER_NAMES = {0: "C0 Standard", 1: "C1 Poor", 2: "C2 High"}

# ── Базовый layout (применять через **LAYOUT) ─────────────────
_FONT = dict(family="Inter, Arial, sans-serif", size=12, color="#243746")
_GRID = "rgba(115,132,145,0.16)"
_AXIS = "rgba(115,132,145,0.42)"
_PLOT_BG = "rgba(0,0,0,0)"
_HATCHES = ["/", "\\", "x", "-", "|", "+", "."]
LAYOUT = dict(
    font=_FONT,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=_PLOT_BG,
    hovermode="closest",
    legend=dict(
        orientation="h",
        x=0.5, xanchor="center",
        y=1.04, yanchor="bottom",
        bgcolor="rgba(255,255,255,0.72)",
        bordercolor="rgba(196,211,222,0.80)",
        borderwidth=1,
        font=dict(size=11),
    ),
)


def _apply_plot_theme(fig):
    """Единый тихий стиль графиков: пастель, мягкая сетка, штриховка столбцов."""
    if fig is None:
        return fig
    try:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=_PLOT_BG,
            font=_FONT,
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="rgba(196,211,222,0.95)",
                font=dict(family="Inter, Arial, sans-serif", size=12, color="#243746"),
            ),
        )
        fig.update_xaxes(
            showline=True, linecolor=_AXIS, linewidth=1,
            gridcolor=_GRID, zeroline=False,
            tickfont=dict(size=11, color="#526473"),
            title_font=dict(size=12, color="#405565"),
        )
        fig.update_yaxes(
            showline=True, linecolor=_AXIS, linewidth=1,
            gridcolor=_GRID, zeroline=False,
            tickfont=dict(size=11, color="#526473"),
            title_font=dict(size=12, color="#405565"),
        )
        for i, trace in enumerate(fig.data):
            t = getattr(trace, "type", "")
            if t in ("bar", "histogram"):
                try:
                    _pattern = getattr(trace.marker, "pattern", None)
                    _shape = getattr(_pattern, "shape", None) if _pattern is not None else None
                    trace.marker.pattern = dict(
                        shape=_shape or _HATCHES[i % len(_HATCHES)],
                        solidity=0.12,
                        fgcolor="rgba(75,92,105,0.22)",
                        bgcolor="rgba(255,255,255,0)",
                    )
                    trace.marker.line.width = max(getattr(trace.marker.line, "width", 0) or 0, 0.9)
                    if not getattr(trace.marker.line, "color", None):
                        trace.marker.line.color = "rgba(75,92,105,0.28)"
                except Exception:
                    pass
            elif t in ("scatter", "violin", "box"):
                try:
                    if getattr(trace, "opacity", None) is None:
                        trace.opacity = 0.92
                except Exception:
                    pass
    except Exception:
        pass
    return fig


def _plot_chart(fig, **kwargs):
    """Streamlit wrapper so every on-page Plotly chart receives the same theme."""
    return st.plotly_chart(_apply_plot_theme(fig), **kwargs)

# ── Настройка страницы ────────────────────────────────────────
st.set_page_config(
    page_title="IVF Digital Twin · Clinical UI",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

_LANG = st.sidebar.radio(
    "Язык / Language",
    ["Русский", "English"],
    horizontal=True,
    key="_clinical_language",
)
_t = lambda key: _translate(_LANG, key)

# -- Podklyuchaem kriptograficheskiy dvizhok
# -- PDF generator
try:
    from src.pdf_report import generate_patient_report
    _PDF_OK = True
except ImportError as _pe:
    _PDF_OK = False
    _PDF_ERR = str(_pe)
_BASE_DIR = _SOURCE_DIR
sys.path.insert(0, os.path.join(_BASE_DIR, "src"))
try:
    from crypt_engine import verify_license, get_model_key
    _CRYPT_ENGINE_OK = True
except ImportError as _ce:
    _CRYPT_ENGINE_OK = False
    _CRYPT_ENGINE_ERR = str(_ce)
    _CRYPT_ENGINE_PYTHON = sys.executable
    if "crypt_engine" in _CRYPT_ENGINE_ERR:
        _CRYPT_ENGINE_HINT = (
            "Не найден модуль `src/crypt_engine.py` рядом с `app.py`. "
            "Запускайте приложение из корневой папки проекта, где есть папка `src`."
        )
    elif "cryptography" in _CRYPT_ENGINE_ERR:
        _CRYPT_ENGINE_HINT = (
            f"Установите пакет в тот же Python, которым запущен Streamlit: "
            f"`\"{_CRYPT_ENGINE_PYTHON}\" -m pip install cryptography`"
        )
    else:
        _CRYPT_ENGINE_HINT = (
            "Проверьте зависимости лицензирования и запуск из корневой папки проекта."
        )

# ── Файл лицензии ─────────────────────────────────────────────
_LICENSE_FILE = os.path.join(_BASE_DIR, "license.lic")
_SESSION_KEY  = "ivf_license_valid"

def _render_license_gate():
    """Показывает красивый экран ввода лицензии."""
    # Центрированный блок
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        logo_path = os.path.join(_BASE_DIR, "logo22.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=90)

        st.markdown("## IVF Digital Twin v7.1")
        st.markdown("### Активация лицензии")
        st.markdown("---")

        # Проверяем, есть ли сохранённый ключ
        saved_key = ""
        if os.path.exists(_LICENSE_FILE):
            try:
                saved_key = open(_LICENSE_FILE, encoding="utf-8").read().strip()
            except Exception:
                pass

        if saved_key:
            st.info("Лицензионный файл найден. Проверка...")
        else:
            st.markdown(
                "Введите ваш лицензионный ключ, полученный от поставщика. "
                "Ключ начинается с **IVF-** и вводится один раз — "
                "затем сохраняется в файле `license.lic`."
            )

        key_input = st.text_area(
            "Лицензионный ключ",
            value=saved_key,
            height=120,
            placeholder="IVF-eyJjbGluaWMi...",
            help="Скопируйте ключ, полученный от разработчика, и вставьте сюда целиком"
        )

        activate_btn = st.button("🔓 Активировать", use_container_width=True, type="primary")

        if activate_btn or (saved_key and saved_key == key_input.strip()):
            _try_activate(key_input.strip())

        st.markdown("---")
        st.caption(
            "IVF Digital Twin · from in vitro to in silico · embryossa@gmail.com · Research prototype"
        )


def _try_activate(key_str: str):
    """Проверяет и активирует лицензию."""
    if not key_str:
        st.warning("Введите лицензионный ключ")
        return

    if not _CRYPT_ENGINE_OK:
        st.error(f"Модуль лицензирования недоступен: {_CRYPT_ENGINE_ERR}")
        st.info(_CRYPT_ENGINE_HINT)
        st.caption(f"Python: `{_CRYPT_ENGINE_PYTHON}`")
        return

    with st.spinner("Проверка лицензии..."):
        valid, clinic_name, expires_date, reason = verify_license(key_str)

    if valid:
        # Сохраняем ключ в файл
        try:
            with open(_LICENSE_FILE, "w", encoding="utf-8") as f:
                f.write(key_str)
        except Exception:
            pass  # Не критично

        st.session_state[_SESSION_KEY] = True
        st.session_state["ivf_clinic_name"] = clinic_name
        st.session_state["ivf_expires"] = expires_date
        st.success(f"Лицензия активирована. Добро пожаловать, {clinic_name}")
        st.rerun()
    else:
        st.error(reason)
        # Если файл есть но ключ невалиден — удаляем
        if os.path.exists(_LICENSE_FILE):
            try:
                os.remove(_LICENSE_FILE)
            except Exception:
                pass


def check_license():
    """Проверяет офлайн-лицензию перед показом интерфейса.

    Исследовательская сборка поставляется без `src/crypt_engine.py`.
    В этом случае гейт пропускается и приложение работает в Research Mode:
    все слои доступны, но сборка не лицензирована для клинического
    применения. Коммерческая сборка содержит crypt_engine, и гейт
    работает как обычно.
    """
    # 0. Research Mode — модуль лицензирования отсутствует
    if not _CRYPT_ENGINE_OK:
        st.session_state[_SESSION_KEY] = True
        st.session_state["ivf_research_mode"] = True
        return

    # 1. Session state (уже проверена в этой сессии)
    if st.session_state.get(_SESSION_KEY):
        return

    # 2. Из файла license.lic
    if os.path.exists(_LICENSE_FILE):
        try:
            saved_key = open(_LICENSE_FILE, encoding="utf-8").read().strip()
            if saved_key and _CRYPT_ENGINE_OK:
                valid, clinic_name, expires_date, reason = verify_license(saved_key)
                if valid:
                    st.session_state[_SESSION_KEY] = True
                    st.session_state["ivf_clinic_name"] = clinic_name
                    st.session_state["ivf_expires"] = expires_date
                    return
        except Exception:
            pass

    # 3. Показываем экран активации
    _render_license_gate()
    st.stop()


check_license()

if st.session_state.get("ivf_research_mode"):
    st.warning(
        "**Research Mode** — исследовательская сборка без модуля лицензирования. "
        "Все слои доступны, но эта сборка **не лицензирована для клинического "
        "применения**: результаты предназначены для изучения методов и "
        "воспроизведения опубликованных экспериментов. См. DISCLAIMER.md.",
        icon="🔬",
    )

# ── Баннер лицензии в sidebar ─────────────────────────────────
_clinic = st.session_state.get("ivf_clinic_name", "")
_expires = st.session_state.get("ivf_expires")
if _clinic and _expires:
    _days_left = (_expires - date.today()).days
    if _days_left <= 14:
        st.sidebar.warning(
            (f"Licence expires in **{_days_left} days** ({_expires}).\n"
             "Contact the supplier for renewal."
             if _LANG == "English" else
             f"Лицензия истекает через **{_days_left} дн.** ({_expires})\n"
             "Обратитесь к поставщику для продления.")
        )
    else:
        st.sidebar.success(
            f"Клиника: **{_clinic}**\n"
            f"Лицензия до: {_expires} ({_days_left} дн.)"
        )


# ── подключаем основной pipeline (.py или скомпилированный .pyd) ──
_src_dir = os.path.join(_SOURCE_DIR, "src")
sys.path.insert(0, _src_dir)

# 7.1: обычный импорт (.py или .pyd). Тот же объект модуля использует ivf_core,
# поэтому PatientInput/KnownValues интерфейса и ядра — одни и те же классы.
try:
    import ivf_digital_twin as _ivf_mod
except Exception as _pipeline_exc:
    st.error(f"Критическая ошибка: ivf_digital_twin не загружен ({_pipeline_exc})")
    st.stop()
globals().update({k: getattr(_ivf_mod, k)
                  for k in dir(_ivf_mod) if not k.startswith("__")})
import dt_bridge as _dt_bridge

# ── подключаем CSDI Hybrid v3 (L5) ───────────────────────────
CSDI_AVAILABLE   = False
CSDI_LOAD_ERROR  = ""
_CSDI_CLASS_READY = False

# ── Подключаем GNN Predictor (Graph Transformer) ──────────────
_GNN_IMPORT_OK  = False
_GNN_LOAD_ERROR = ""
try:
    from gnn_predictor import load_gnn_model as _load_gnn_model
    from gnn_predictor import predict_gnn    as _predict_gnn
    from gnn_predictor import build_patient_features as _build_gnn_features
    from gnn_predictor import build_gnn_neighborhood_figure as _build_gnn_figure
    _GNN_IMPORT_OK = True
except ImportError as _gnn_ie:
    _GNN_LOAD_ERROR = str(_gnn_ie)

# ── TRP Engine (Total Reproductive Potential) ─────────────────
_TRP_OK    = False
_TRP_ERROR = ""
try:
    from trp_engine import (
        compute_trp    as _compute_trp,
        build_trp_tab  as _build_trp_tab,
        build_trp_inputs as _build_trp_inputs,
        TRPInput       as _TRPInput,
        PastCycle      as _PastCycle,
    )
    _TRP_OK = True
except ImportError as _trp_ie:
    _TRP_ERROR = str(_trp_ie)

# CSDI импортируется как отдельный модуль: выполнение его кода в globals()
# перезаписывало константы и функции ядра (исправлено в 7.1, см. ivf_core).
# Загрузка модели и запуск CSDI — в ivf_core (L5 с проверкой применимости).
try:
    from ivf_core import _CSDI_CLASS_READY, CSDI_LOAD_ERROR
    CSDI_AVAILABLE = _CSDI_CLASS_READY
except Exception as _e:
    CSDI_LOAD_ERROR = str(_e)

# ── CSS ───────────────────────────────────────────────────────
if _UI_OK:
    UI.inject_css()
else:
    st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stMetric { background: white; border-radius: 10px;
                padding: 12px; border-left: 4px solid #1B4F72; }
    .block-container { padding-top: 1.5rem; }
    h1 { color: #1B4F72; }
    h2 { color: #1B4F72; border-bottom: 1px solid #d0e4f0; padding-bottom: 4px; }
    h3 { color: #154360; }
    .disclaimer { background: #FFF3CD; border-left: 4px solid #FFC107;
                  padding: 10px 14px; border-radius: 6px;
                  font-size: 0.85em; color: #555; margin-bottom: 1rem; }
    .result-box { background: #EAF4FB; border-left: 5px solid #1B4F72;
                  padding: 14px 18px; border-radius: 8px; margin: 10px 0; }
    .diff-box   { background: #E8F5E9; border-left: 5px solid #2E7D32;
                  padding: 14px 18px; border-radius: 8px; margin: 10px 0; }
    .diff-warn  { background: #FFF8E1; border-left: 5px solid #F9A825;
                  padding: 14px 18px; border-radius: 8px; margin: 10px 0; }
    .cluster-c0 { background: #E3F2FD; border-left: 4px solid #1976D2;
                  padding: 10px; border-radius: 6px; }
    .cluster-c1 { background: #FFEBEE; border-left: 4px solid #C62828;
                  padding: 10px; border-radius: 6px; }
    .cluster-c2 { background: #E8F5E9; border-left: 4px solid #2E7D32;
                  padding: 10px; border-radius: 6px; }
    .ks-pass { color: #2E7D32; font-weight: bold; }
    .ks-fail { color: #C62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  DT ANALYTICS COLLECTOR
#  Автоматически сохраняет каждый расчёт в dt_analytics_data/dt_predictions.csv
# ══════════════════════════════════════════════════════════════

_ANALYTICS_DIR = _Path(_APP_DIR) / "dt_analytics_data"
_ANALYTICS_CSV = _ANALYTICS_DIR / "dt_predictions.csv"

def _save_analytics(result, clinic_name, patient_name="", patient_id=""):
    """
    Записывает одну строку расчёта в master CSV dt_predictions.csv.
    Вызывается один раз при успешной генерации PDF-отчёта.

    7.1: та же схема и тот же writer, что у пакетных скриптов
    (ivf_core.save_analytics_record): p_kat_raw в определении 7.0 плюс
    kat_transfer_* — KAT в том виде, в каком она входит в L7. Файл со старой
    схемой архивируется, а не дописывается со сдвигом колонок.
    Параметры пациентки берутся из рассчитанного случая, а не из текущих
    полей боковой панели. Возвращает record_id (str) или None при ошибке.
    """
    try:
        from ivf_core import save_analytics_record
        p = result["patient"]
        record_id = save_analytics_record(
            result=result, age=p["age"], amh=p["amh"], afc=p["afc"], bmi=p["bmi"],
            attempt=p["attempt"], sperm_source=p["sperm_source"],
            follicles=p["follicles"], clinic_name=clinic_name or "",
            patient_name=patient_name or "", patient_id=patient_id or "",
            analytics_csv=str(_ANALYTICS_CSV),
        )
        if record_id is None:
            raise RuntimeError("save_analytics_record failed")
        return record_id
    except Exception as _analytics_exc:
        # Аналитика никогда не блокирует основную работу приложения
        try:
            _err_path = _ANALYTICS_DIR / "analytics_errors.log"
            _ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
            with open(_err_path, "a", encoding="utf-8") as _ef:
                _ef.write(f"{datetime.now().isoformat()} | {_analytics_exc}\n")
        except Exception:
            pass
        return None


# ── SIDEBAR — Ввод данных ─────────────────────────────────────

# ── Кешированная загрузка GNN модели ─────────────────────────
@st.cache_resource(show_spinner=False)
def _get_gnn_bundle():
    """Загружает GNN один раз на сессию."""
    if not _GNN_IMPORT_OK:
        return {'available': False, 'error': _GNN_LOAD_ERROR}
    # Тот же экземпляр, что использует расчёт ivf_core.
    from ivf_core import load_gnn_bundle
    return load_gnn_bundle()

_gnn_bundle = _get_gnn_bundle()
st.sidebar.image(os.path.join(_SOURCE_DIR, "logo22.png"), width=80)
st.sidebar.title("IVF Digital Twin")
st.sidebar.caption("Clinical UI Redesign · from in vitro to in silico")
st.sidebar.markdown("---")

# ── Краткий отчёт для пациента (модуль patient_brief.py, опционально) ──────
_brief_source_path = os.path.join(_APP_DIR, "patient_brief.py")
try:
    import importlib.util as _brief_importlib_util
    _brief_module_spec = _brief_importlib_util.find_spec("patient_brief")
except Exception:
    _brief_module_spec = None
_BRIEF_AVAILABLE = os.path.exists(_brief_source_path) or _brief_module_spec is not None
if _BRIEF_AVAILABLE:
    # Краткий режим — по умолчанию. Расширенный включается кнопкой в отчёте.
    if "_view_mode" not in st.session_state:
        st.session_state["_view_mode"] = "Clinical summary"

st.sidebar.header(_t("patient_parameters"))
age  = st.sidebar.number_input(_t("age"), 18, 50, 35, 1)
amh  = st.sidebar.number_input(_t("amh"), 0.01, 15.0, 2.50, 0.10,
                                 format="%.2f")
afc  = st.sidebar.number_input(_t("afc"), 1, 60, 15, 1)
bmi_col1, bmi_col2 = st.sidebar.columns(2)
with bmi_col1:
    _height_cm = st.number_input(_t("height"), 140, 200, 165, 1)
with bmi_col2:
    _weight_kg = st.number_input(_t("weight"), 40, 150, 65, 1)
bmi = round(_weight_kg / (_height_cm / 100) ** 2, 1)
_bmi_band = (
    ("↑ high" if _LANG == "English" else "↑ избыток") if bmi >= 30 else
    (("↓ low" if _LANG == "English" else "↓ дефицит") if bmi < 18.5 else
     ("✓ normal" if _LANG == "English" else "✓ норма"))
)
st.sidebar.caption(f"BMI: **{bmi} kg/m²** · {_bmi_band}" if _LANG == "English" else
                   f"ИМТ: **{bmi} кг/м²** · {_bmi_band}")

st.sidebar.markdown("---")
st.sidebar.header(_t("cycle_parameters"))
attempt = st.sidebar.number_input(_t("attempt"), 1, 10, 1, 1)
follicles = st.sidebar.number_input(_t("follicles"),
                                     0, 60, 0, 1)
follicles = None if follicles == 0 else int(follicles)

sperm_label = st.sidebar.selectbox(
    _t("sperm"),
    (["Ejaculate", "Testicular (NOA)", "Testicular (OA)", "Epididymal"]
     if _LANG == "English" else
     ["Эякулят", "Тестикулярная (НОА)", "Тестикулярная (ОА)", "Эпидидимальная"]),
    index=0,
    help="Модуль банкинга учитывает источник спермы при оценке выхода эуплоидных",
)
_sperm_map = {
    "Эякулят":               "ejaculate",
    "Тестикулярная (НОА)":   "testicular_NOA",
    "Тестикулярная (ОА)":    "testicular_OA",
    "Эпидидимальная":        "epididymal",
    "Ejaculate":             "ejaculate",
    "Testicular (NOA)":      "testicular_NOA",
    "Testicular (OA)":       "testicular_OA",
    "Epididymal":            "epididymal",
}
sperm_source = _sperm_map[sperm_label]

st.sidebar.markdown("---")
st.sidebar.header(_t("mid_cycle"))
st.sidebar.caption("Enter observed values; leave 0 when unavailable." if _LANG == "English"
                   else "Введите наблюдённые значения. Оставьте 0 = не наблюдалось.")

def optional_int(val): return int(val) if val > 0 else None

okk_obs    = st.sidebar.number_input("Oocytes retrieved (OCC)" if _LANG == "English" else "Получено ооцитов (ОКК)", 0, 60, 0)
mii_obs    = st.sidebar.number_input("MII oocytes" if _LANG == "English" else "MII ооцитов", 0, 60, 0)
pn2_obs    = st.sidebar.number_input("2PN zygotes" if _LANG == "English" else "2PN зигот", 0, 50, 0)
blasts_obs = st.sidebar.number_input("Total blastocysts" if _LANG == "English" else "Бластоцист всего", 0, 40, 0)
good_obs   = st.sidebar.number_input("Good-quality blastocysts" if _LANG == "English" else "Бластоцист хор. кач.", 0, 40, 0)
euploid_obs= st.sidebar.number_input("Euploid (PGT-A)" if _LANG == "English" else "Эуплоидных (ПГТ-А)", 0, 30, 0)

known = KnownValues(
    okk     = optional_int(okk_obs),
    mii     = optional_int(mii_obs),
    pn2     = optional_int(pn2_obs),
    blasts  = optional_int(blasts_obs),
    good    = optional_int(good_obs),
    euploid = optional_int(euploid_obs),
)

st.sidebar.markdown("---")
st.sidebar.header("Clinic data (prior)" if _LANG == "English" else "Данные клиники (prior)")

# Конфиг клиники: success/transfer батчи берутся из clinic_config.json, который
# редактируется под каждую клинику. Файл — единственный источник данных, чтобы
# исключить ручную правку врачами в интерфейсе.
_clinic_cfg_path = os.path.join(_BASE_DIR, "clinic_config.json")
_clinic_cfg = None
if os.path.exists(_clinic_cfg_path):
    try:
        with open(_clinic_cfg_path, "r", encoding="utf-8") as _cf:
            _clinic_cfg = json.load(_cf)
    except Exception as _cfg_e:
        st.sidebar.error(f"Ошибка чтения clinic_config.json: {_cfg_e}")

clinic_s, clinic_t = None, None
if _clinic_cfg is not None:
    _cfg_clinic_name = (_clinic_cfg.get("clinic_name") or "").strip()
    if _cfg_clinic_name:
        st.session_state["ivf_clinic_name"] = _cfg_clinic_name
    _cfg_default_on = bool(_clinic_cfg.get("use_clinic_data", True))
    use_clinic = st.sidebar.checkbox(
        "Use clinic data" if _LANG == "English" else "Использовать данные клиники",
        value=_cfg_default_on,
        help=("Source: clinic_config.json. Edit the file to change the batches."
              if _LANG == "English" else
              "Источник — clinic_config.json. Отредактируйте файл, чтобы изменить."))
    if use_clinic:
        try:
            # Пересчитываем КАЖДЫЙ раз из конфига — частота не «залипает».
            _batches = _clinic_cfg.get("batches", []) or []
            clinic_s = [int(b[0]) for b in _batches]
            clinic_t = [int(b[1]) for b in _batches]
            if clinic_s and clinic_t and sum(clinic_t) > 0:
                obs_rate = sum(clinic_s) / sum(clinic_t)
                _name_txt = f" · {_cfg_clinic_name}" if _cfg_clinic_name else ""
                st.sidebar.success(
                    (f"✓ Clinic data active{_name_txt}: {len(clinic_s)} batches, "
                     f"observed rate {obs_rate*100:.1f}% ({sum(clinic_s)}/{sum(clinic_t)})"
                     if _LANG == "English" else
                     f"✓ Данные клиники активны{_name_txt}: "
                     f"{len(clinic_s)} батчей, факт. частота "
                     f"{obs_rate*100:.1f}% ({sum(clinic_s)}/{sum(clinic_t)})"))
            else:
                clinic_s, clinic_t = None, None
                st.sidebar.warning("clinic_config.json: пустой список batches")
        except Exception as _b_e:
            clinic_s, clinic_t = None, None
            st.sidebar.error(f"clinic_config.json: неверный формат batches ({_b_e})")
    with st.sidebar.expander("Show clinic batches" if _LANG == "English" else
                             "Показать батчи клиники"):
        st.caption("Edited only in clinic_config.json" if _LANG == "English" else
                   "Редактируется только в файле clinic_config.json")
        if clinic_s and clinic_t:
            st.dataframe(
                {"Успехи": clinic_s, "Переносы": clinic_t},
                use_container_width=True, hide_index=True)
        else:
            st.caption("—")
else:
    st.sidebar.info("clinic_config.json не найден — данные клиники не используются "
                    "(prior строится без них). Добавьте файл рядом с app.py.")
    use_clinic = False

n_sim = st.sidebar.select_slider(
    "MC iterations" if _LANG == "English" else "Итераций MC",
    options=[500, 1000, 2000, 5000], value=2000)

with st.sidebar.expander(_t("reliability_settings"), expanded=False):
    st.caption(
        "These thresholds change only the displayed reliability category, not the probability calculation."
        if _LANG == "English" else
        "Пороги меняют только словесную категорию надёжности и не влияют на расчёт вероятности."
    )
    _rel_high_threshold = st.slider(_t("high_from"), 45, 80, 60, 1,
                                    key="_rel_high_threshold")
    _rel_moderate_threshold = st.slider(_t("moderate_from"), 20, 60, 35, 1,
                                        key="_rel_moderate_threshold")
    if _rel_moderate_threshold >= _rel_high_threshold:
        _rel_moderate_threshold = max(20, _rel_high_threshold - 5)

# ── Загрузка нейросети (L3) ───────────────────────────────────
st.sidebar.markdown("---")
_model_status_box = st.sidebar.expander(_t("model_status"), expanded=False)
_model_status_box.markdown("**KAT · KAN + FT-Transformer**")

@st.cache_resource(show_spinner="Загрузка нейросетевых моделей...")
def get_nn_model():
    # Тот же экземпляр, что использует расчёт ivf_core.
    from ivf_core import load_nn_model
    return load_nn_model()

nn_model = get_nn_model()

if nn_model is not None:
    _model_status_box.success("KAT (KAN + FT-Transformer) loaded" if _LANG == "English" else
                              "KAT (KAN + FT-Transformer) загружена")
else:
    if not NN_LIBS_AVAILABLE:
        if "dll_error" in NN_LIBS_ERROR:
            _model_status_box.error(
                "**Ошибка DLL (fbgemm.dll)**\n\n"
                "Запустите `fix_torch_dll.bat`\n\n"
                "Работает FORTUNE+KPI без нейросети."
            )
        else:
            _model_status_box.warning(
                "torch не установлен\n\n"
                "Запустите `fix_torch_dll.bat`\n\n"
                "Работает FORTUNE+KPI без нейросети."
            )
    else:
        _model_status_box.info(
            "Файлы моделей не найдены.\n\n"
            "Поместите в `src/` или `models/`:\n"
            "- `Prediction_KAN.pth`\n"
            "- `FTTransformer.joblib`\n"
            "- `KAT_calibrated_model.pkl`\n\n"
            "Работает FORTUNE+KPI без нейросети."
        )

# ── Загрузка CSDI Hybrid v3 (L5) ─────────────────────────────
_model_status_box.markdown("**CSDI Hybrid v3**")
CSDI_MODEL_LOAD_ERROR = ""

@st.cache_resource(show_spinner="Загрузка CSDI Hybrid v3...")
def get_csdi_model():
    if not _CSDI_CLASS_READY:
        return None
    # Тот же экземпляр, что использует расчёт ivf_core (CSDI запускается там).
    from ivf_core import load_csdi_model
    return load_csdi_model()

csdi_model = get_csdi_model()


if csdi_model is not None:
    _model_status_box.success(
        (f"CSDI Hybrid v3 loaded (threshold: {csdi_model.best_threshold:.2f})"
         if _LANG == "English" else
         f"CSDI Hybrid v3 загружена (порог: {csdi_model.best_threshold:.2f})")
    )
elif not _CSDI_CLASS_READY:
    _model_status_box.info(
        "CSDI Hybrid v3 не загружен.\n\n"
        f"Причина: `{CSDI_LOAD_ERROR or 'src/embryo_csdi_v3.py не найден'}`\n\n"
        "Проверьте файл `src/embryo_csdi_v3.py` и зависимости L5."
    )
elif CSDI_MODEL_LOAD_ERROR:
    _model_status_box.info(
        "Модель CSDI найдена, но не загрузилась.\n\n"
        f"Причина: `{CSDI_MODEL_LOAD_ERROR}`\n\n"
        "Проверьте файлы в `models/embryo_v3_model/`."
    )
else:
    _model_status_box.info(
        "Модель не найдена.\n\n"
        "Поместите папку `embryo_v3_model/` в `models/`.\n\n"
        "Обучение: `python src/embryo_csdi_v3.py`"
    )

# ── Статус GNN модели (Graph Transformer) ─────────────────────
_model_status_box.markdown("**GAT · Graph Transformer**")

if _gnn_bundle.get('available'):
    _model_status_box.success("GNN (Graph Transformer) loaded" if _LANG == "English" else
                              "GNN (Graph Transformer) загружена")
elif not _GNN_IMPORT_OK:
    _model_status_box.warning(
        "torch-geometric не установлен\n\n"
        "Запустите `INSTALL.bat` (шаг 8) или:\n"
        "```\npip install torch-scatter torch-sparse \\\n"
        "  torch-cluster torch-spline-conv \\\n"
        "  -f https://data.pyg.org/whl/torch-2.5.1+cpu.html\n"
        "pip install torch-geometric\n```\n\n"
        "GAT Ансамбль будет недоступен."
    )
else:
    _gnn_err_short = _gnn_bundle.get('error', 'Файл не найден')[:80]
    _model_status_box.info(
        "GNN модель не загружена.\n\n"
        f"Причина: `{_gnn_err_short}`\n\n"
        "Поместите `gnn_ivf_model.pt` в `models/`.\n\n"
        "Обучение: `python gnn_ivf_562.py clinical_protocols.xlsx`"
    )

run_btn = st.sidebar.button(_t("run"), use_container_width=True,
                             type="primary")

# ── ГЛАВНАЯ СТРАНИЦА ──────────────────────────────────────────
st.title("IVF Digital Twin · Clinical UI")
_disclaimer_text = (
    "<b>Clinical decision support only.</b> Predictions are probabilistic estimates. "
    "Final management is determined by the fertility specialist."
    if _LANG == "English" else
    "<b>Только для поддержки клинического решения.</b> Все прогнозы являются "
    "вероятностными оценками. Окончательную тактику определяет врач-репродуктолог."
)
st.markdown(f"""
<div class="disclaimer">
{_disclaimer_text}
<br>IVF Digital Twin v7.1 · <i>from in vitro to in silico</i>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    # If we have cached results, skip the welcome screen and proceed to show results + PDF
    if st.session_state.get("_pdf_res") is None:
        col1, col2, col3 = st.columns(3)
        if _LANG == "English":
            col1.info("← Enter patient data in the sidebar")
            col2.info("Select **Calculate prognosis**")
            col3.info("Review the concise clinical report")
        else:
            col1.info("← Введите данные пациентки в панели слева")
            col2.info("Нажмите **Рассчитать прогноз**")
            col3.info("Получите лаконичное клиническое резюме")

        with st.expander(_t("about_system")):
            if _LANG == "English":
                st.markdown("""
        **IVF Digital Twin v7.1** is an integrated IVF outcome-prediction
        system combining seven independent assessment layers.

        *from in vitro to in silico.*

        | Layer | Method |
        |---|---|
        | L1 Stochastic pipeline | NB (log link) + beta-binomial filters (S1–S6b) |
        | L2 Transfer ensemble | FORTUNE + KPIScore (logit weighting) |
        | L3 Neural network | KAN + FT-Transformer + Venn–Abers (KAT) |
        | L4 Response cluster | Nearest 18D centroid |
        | L5 Laboratory forecast | CSDI-Transformer + LightGBM + Conformal PI |
        | L6 Patient graph | Graph Attention Transformer (GAT) + KAT ensemble |
        | **L7 BEFE** | **Bayesian Evidence Fusion: Prior → Evidence → Posterior** |

        **L7 (BEFE)** combines the layers into a single posterior estimate.
        The mechanistic L1 prior is updated with neural evidence from KAT and
        GAT, verified by the CSDI laboratory model and accompanied by clinical
        and embryology OOD checks. The clinician-facing report presents the
        final probability together with an explicit reliability assessment.
        """)
            else:
                st.markdown("""
        **IVF Digital Twin v7.1** — интегрированная система прогнозирования
        исходов ЭКО, объединяющая 7 независимых слоёв оценки.

        *from in vitro to in silico.*

        | Слой | Метод |
        |---|---|
        | L1 Стохастический pipeline | NB (log-связь) + бета-биномиальные фильтры (S1–S6b) |
        | L2 Ансамбль на перенос | FORTUNE + KPIScore (логит-взвешивание) |
        | L3 Нейросеть | KAN + FT-Transformer + Venn-Abers (KAT) |
        | L4 Кластер | Ближайший центроид 18D |
        | L5 Лабораторный прогноз | CSDI-Transformer + LightGBM + Conformal PI |
        | L6 Граф пациентов | Graph Attention Transformer (GAT) + ансамбль с KAT |
        | **L7 BEFE** | **Bayesian Evidence Fusion: Prior → Evidence → Posterior** |

        **L7 (BEFE)** объединяет все уровни в единый posterior: механистический
        приор L1 обновляется доказательствами KAT и GAT, верифицируется CSDI и
        сопровождается клиническим и эмбриологическим OOD-контролем. Врач видит
        итоговую вероятность вместе с явной оценкой надёжности.
        """)
        st.stop()
    else:
        # Restore cached results so the rest of the page renders normally
        res         = st.session_state["_pdf_res"]
        _eb         = st.session_state.get("_pdf_eb")
        known       = st.session_state.get("_pdf_known", {})
        sperm_source = st.session_state.get("_pdf_sperm", "")

# ── РАСЧЁТ ───────────────────────────────────────────────────
# 7.1: один поток данных для всех выводов (экран, PDF, аналитика):
# ivf_core (L1–L6) → BEFE L7 → клиническая сводка → вероятность цикла.
_reliability_limits = {"high": int(_rel_high_threshold),
                       "moderate": int(_rel_moderate_threshold)}
if run_btn:
    _mc_spinner = (f"Running {n_sim} Monte Carlo iterations…" if _LANG == "English"
                   else f"Выполняется {n_sim} итераций Monte Carlo…")
    with st.spinner(_mc_spinner):
        _dt71 = _dt_bridge.compute(
            {
                "age": float(age), "amh": float(amh), "afc": int(afc),
                "bmi": float(bmi), "attempt": int(attempt),
                "follicles": follicles, "sperm_source": sperm_source,
                "known_okk": known.okk, "known_mii": known.mii,
                "known_pn2": known.pn2, "known_blasts": known.blasts,
                "known_good": known.good, "known_euploid": known.euploid,
                "n_sim": int(n_sim), "seed": 42,
            },
            clinic_successes=clinic_s, clinic_trials=clinic_t,
            reliability=_reliability_limits,
        )
        res = _dt71["res"]
        known = _dt71["known"]
        st.session_state["_dt71"]        = _dt71
        st.session_state["_pdf_res"]     = res
        st.session_state["_pdf_known"]   = known
        st.session_state["_pdf_age"]     = float(age)
        st.session_state["_pdf_amh"]     = float(amh)
        st.session_state["_pdf_afc"]     = int(afc)
        st.session_state["_pdf_bmi"]     = float(bmi)
        st.session_state["_pdf_attempt"] = int(attempt)
        st.session_state["_pdf_sperm"]   = sperm_source
    # L3 KAT на сценариях с переносом (res_transfer); None без весов KAT.
    _nn_s = _dt71["res_transfer"].get('nn_prediction', {}) if _dt71["nn_available"] else {}
    _nv_s = res.get('nn_nvsa', {})
    st.session_state["_pdf_p_kat_raw"] = _dt71["p_kat_raw"]
    st.session_state["_pdf_p_nvsa"]    = _nv_s.get('adjusted_mean')
    st.session_state["_pdf_ci_kat"]    = _nn_s.get('base_prob_ci',  (None, None))
    st.session_state["_pdf_ci_nvsa"]   = _nv_s.get('adjusted_ci',   (None, None))
else:
    # Use cached values (user interacting with PDF form after calculation)
    res          = st.session_state["_pdf_res"]
    known        = st.session_state.get("_pdf_known", {})
    sperm_source = st.session_state.get("_pdf_sperm", "")

_dt71 = st.session_state["_dt71"]
# Порог надёжности из боковой панели меняет только словесную категорию.
if _dt71["fusion"] is not None:
    _dt71["fusion"].reliability_band = _dt_bridge.reliability_band(
        _dt71["fusion"].reliability, _reliability_limits["high"],
        _reliability_limits["moderate"])

# ── BANKING MODULE (Esteves model, 7.1: esteves_banking_analysis ядра) ─────
_eb = _dt71["eb"]
st.session_state["_pdf_eb"] = _eb

# ── БЛОК РЕЗУЛЬТАТОВ ─────────────────────────────────────────
st.markdown("---")
st.header(_t("results"))

ca   = res['cluster_analysis']
post = res['posterior']
dom  = ca['dominant_cluster']

# ── Ключевые метрики считаются ниже, после BEFE/GAT, и выводятся единой карточкой.
# L3 KAT — из ядра 7.1: среднее по сценариям с переносом; None без весов KAT
# (тогда nn_prediction содержит FORTUNE+KPI, т.е. сам приор L1).
_nn_pred = _dt71["res_transfer"].get('nn_prediction', {}) if _dt71["nn_available"] else {}
_nn_nvsa = res.get('nn_nvsa', {})
_p_kat_raw  = _dt71["p_kat_raw"]
_p_nvsa     = _nn_nvsa.get('adjusted_mean',  None)
_ci_kat     = _nn_pred.get('base_prob_ci',   (None, None))
_ci_nvsa    = _nn_nvsa.get('adjusted_ci',    (None, None))

# ── GNN / GAT Ансамбль (L6 посчитан ядром на профиле переноса) ─────────────
_gnn_result = _dt71["gnn_result"]
st.session_state['_gnn_result'] = _gnn_result
if run_btn and _gnn_result.get('gnn_prob') is not None:
    # Строим фигуру для PDF сразу после инференса
    try:
        _gnn_fig = _build_gnn_figure(
            _gnn_result,
            gnn_prob      = _gnn_result.get('gnn_prob'),
            ensemble_prob = _gnn_result.get('ensemble_prob'),
        )
        _gnn_fig = _apply_gnn_style(_gnn_fig)
        st.session_state['_pdf_fig_gnn'] = _gnn_fig
    except Exception:
        st.session_state['_pdf_fig_gnn'] = None

_p_gnn_ens  = _dt71["p_gnn_ens"]   # итоговый скор для отображения
_p_gnn_raw  = _dt71["p_gnn_raw"]
_w_gnn      = _gnn_result.get('w_gnn', 0.35)

# Сохраняем в session_state для PDF
st.session_state['_pdf_p_gnn_ens'] = _p_gnn_ens
st.session_state['_pdf_p_gnn_raw'] = _p_gnn_raw
st.session_state['_pdf_w_gnn']     = _w_gnn

# ── L5 CSDI — посчитан ядром (медианы сценариев с переносом, KPIScore по
# формуле обучения, проверка области применимости). None — CSDI не участвует.
_csdi_applicability = _dt71.get("csdi_applicability") or {}
st.session_state["csdi_result"] = _dt71["csdi_result"]

# ── L7 BEFE — главная цифра Результатов (compute_l7_posterior ядра) ────────
# 7.1: исследовательская температурная адаптация (calibrate_for_clinic) больше
# не входит в клинический расчёт; OOD-статистики загружает befe_batch_utils.
_befe_res = _dt71["fusion"] if _BEFE_OK else None
_befe_map = _dt71["fusion_mapping"]
st.session_state['_pdf_befe'] = _befe_res

# ── Клиническая сводка 7.1: единая главная цифра для экрана, PDF и истории ──
_clinical_summary   = _dt71["clinical_summary"]
_clinical_probability = _clinical_summary["probability"]
_cycle_probability  = _dt71["cycle_probability"]
_per_transfer_if_transfer = _dt71["per_transfer"]
# Якорь TRP (кэшируется в _dt71; для закрытого цикла — проспективный пересчёт).
def _trp_anchor_fn():
    return _dt_bridge.trp_anchor(_dt71)

# ── Clinical summary: concise clinician-first result page ──────────────────
if st.session_state.get("_view_mode", "").startswith("Clinical"):
    _brief_ok = False
    try:
        import importlib as _importlib
        import importlib.util as _importlib_util
        _brief_path = os.path.join(_APP_DIR, "patient_brief.py")
        if os.path.exists(_brief_path):
            _brief_spec = _importlib_util.spec_from_file_location(
                "clinical_ui_patient_brief", _brief_path
            )
            patient_brief = _importlib_util.module_from_spec(_brief_spec)
            _brief_spec.loader.exec_module(patient_brief)
        else:
            patient_brief = _importlib.import_module("patient_brief")
        patient_brief.render(globals())
        _brief_ok = True
    except Exception as _brief_exc:
        st.warning(
            (f"Clinical summary is unavailable; detailed view shown: {_brief_exc}"
             if _LANG == "English" else
             f"Клиническое резюме недоступно, показан подробный интерфейс: {_brief_exc}")
        )
    if _brief_ok:
        st.stop()

# ── Complete report: original RU body, translated equivalent for English ──
if (st.session_state.get("_view_mode") == "Detailed report"
        and _LANG == "English"):
    _col_back, _ = st.columns([2, 8])
    with _col_back:
        if st.button(_t("back_summary"), key="_full_to_brief_report",
                     help=_t("clinical_summary"), use_container_width=True):
            st.session_state["_view_mode"] = "Clinical summary"
            st.rerun()
    try:
        from legacy_full_report import render as _render_legacy_full_report
        _render_legacy_full_report(globals(), "English")
    except Exception as _full_exc:
        st.error(f"The complete report could not be rendered: {_full_exc}")
    st.stop()

# ── Кнопка возврата к краткому отчёту (расширенный режим) ───────────────────
if _BRIEF_AVAILABLE and not st.session_state.get("_view_mode", "").startswith("Clinical"):
    _col_back, _ = st.columns([2, 8])
    with _col_back:
        if st.button(_t("back_summary"), key="_to_brief_report",
                     help=_t("clinical_summary")):
            st.session_state["_view_mode"] = "Clinical summary"
            st.rerun()

# ── Визуальный summary результатов: L7 posterior как главная цифра ─────────
_kat_display = f"{_p_kat_raw*100:.1f}%" if _p_kat_raw is not None else "—"
if _p_gnn_ens is not None:
    _gat_display = f"{_p_gnn_ens*100:.1f}%"
elif _gnn_bundle.get('available'):
    _gat_display = "—"
else:
    _gat_display = "н/д"

_cycle_display = (f"{_cycle_probability*100:.1f}%" if _cycle_probability is not None
                  else f"{res['p_overall_cycle']*100:.1f}%")
if _clinical_summary["no_transfer_confirmed"]:
    # Введённые результаты исключают перенос в текущем цикле.
    _main_label = _t("main_outcome")
    _main_value = "0.0%"
    _main_sub = ("No transfer is possible in the current cycle given the entered results"
                 if _LANG == "English" else
                 "По введённым результатам перенос в текущем цикле невозможен")
elif _befe_res is not None and _clinical_probability is not None:
    _main_label = _t("main_outcome")
    _main_value = f"{_clinical_probability*100:.1f}%"
    _display_reliability = _reliability_label(
        _LANG, int(_befe_res.reliability),
        int(_rel_high_threshold), int(_rel_moderate_threshold),
    )
    _main_sub = (
        f"{_t('clinical_corridor')}: "
        f"{_befe_res.ci_low*100:.1f}–{_befe_res.ci_high*100:.1f}% · "
        f"{_t('reliability')}: {_display_reliability} ({_befe_res.reliability}/100)"
    )
else:
    _main_label = _t("per_transfer")
    _main_value = f"{res['p_per_transfer']*100:.1f}%"
    _main_sub = "Fallback L1–L2: BEFE недоступен для этого расчёта"

if _UI_OK:
    UI.result_summary_card(
        title=_main_label,
        value=_main_value,
        subtitle=_main_sub,
        badge_text="L7" if _befe_res is not None else "L2",
        badge_kind="success" if _befe_res is not None else "warning",
        secondary=[
            ("Если цикл viable", f"{res['p_cum_if_viable']*100:.1f}%", ""),
            ("Успех цикла", _cycle_display, "accent"),
            ("KAT ensemble", _kat_display, ""),
            ("GAT ensemble", _gat_display, "highlight"),
        ],
    )
else:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(_main_label, _main_value, help=_main_sub)
    c2.metric("Если цикл viable", f"{res['p_cum_if_viable']*100:.1f}%",
              help="Кумулятивная при ≥1 эмбрионе для переноса")
    c3.metric("Успех цикла", _cycle_display,
              help="От начала стимуляции, включая риск пустого цикла; "
                   "согласована с итоговой вероятностью L7")
    c4.metric("KAT (ансамбль NN)", _kat_display,
              help="Чистый выход нейросетевого ансамбля KAN+FT-Transformer")
    c5.metric("GAT Ансамбль", _gat_display,
              help="Graph Attention Transformer + KAT ансамбль")

p_cancel = np.mean(res['sim_okk'] == 0)
if p_cancel > 0.05:
    st.warning(f"Риск отмены цикла (сценарии без ооцитов): "
               f"**{p_cancel*100:.1f}%**")
if _csdi_applicability.get("reason") not in (None, "in_support"):
    from presentation import csdi_note as _csdi_note
    st.caption(_csdi_note(_csdi_applicability, "en" if _LANG == "English" else "ru"))

# ── Вкладки ───────────────────────────────────────────────────
tab_pipeline, tab_preg, tab_risk, tab_bank, tab_trp, tab_cluster, tab_diff, tab_gat, tab_befe, tab_llm = st.tabs(
    (["L1 Pipeline", "L2 Pregnancy", "Risks", "Banking", "TRP",
      "L4 Response", "L5 Diffusion", "L6 GAT Graph", "L7 BEFE", "LLM"]
     if _LANG == "English" else
     ["L1 Pipeline", "L2 Беременность", "Риски", "Банкинг",
      "TRP", "L4 Кластер", "L5 Diffusion", "L6 GAT Graph", "L7 BEFE", "LLM"]))

# ── TAB: Pipeline ─────────────────────────────────────────────
with tab_pipeline:
    if _UI_OK:
        UI.tab_header_by_key("pipeline")
        UI.metric_row([
            ("ОКК (медиана)",           f"{int(res['okk_med'])}",         ""),
            ("Бластоцист (медиана)",    f"{int(res['blasts_med'])}",      ""),
            ("Хор.кач. (медиана)",      f"{int(res['good_med'])}",        "accent"),
            ("Эупл. прогноз (медиана)", f"{int(res['euploid_med'])}",     "highlight"),
        ])
    col_f, col_v = st.columns([1, 2])

    with col_f:
        (UI.section_header if _UI_OK else lambda _t: st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">{_t}</p>', unsafe_allow_html=True))("Воронка (медианы)")
        stages = ["ОКК", "MII", "2PN", "Бласт.", "Хор.кач.", "Эупл.", "Разм."]
        meds   = [int(res['okk_med']), int(res['mii_med']), int(res['pn2_med']),
                  int(res['blasts_med']), int(res['good_med']),
                  int(res['euploid_med']), int(res['warmed_med'])]
        funnel = go.Figure(go.Funnel(
            y=stages, x=meds,
            textinfo="value+percent initial",
            textfont=dict(family="Inter, Arial, sans-serif", size=12),
            marker=dict(
                color=[hex_rgba(c, 0.82) for c in STAGE_COLORS],
                line=dict(color=[hex_rgba(c, 1.0) for c in STAGE_COLORS], width=1.5),
            ),
            opacity=0.90,
            connector=dict(line=dict(color="rgba(150,150,150,0.4)", width=1.5)),
        ))
        funnel.update_layout(
            **LAYOUT,
            height=400,
            margin=dict(l=120, r=30, t=40, b=40),
        )
        _plot_chart(funnel, use_container_width=True)
        st.session_state["_pdf_fig_funnel"] = funnel

    with col_v:
        (UI.section_header if _UI_OK else lambda _t: st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">{_t}</p>', unsafe_allow_html=True))("Распределения по стадиям")
        arrays = [res['sim_okk'], res['sim_mii'], res['sim_pn2'],
                  res['sim_blasts'], res['sim_good'],
                  res['sim_euploid'], res['sim_warmed']]
        colors = ["#1B4F72","#1A5276","#2471A3","#2E86C1",
                  "#3498DB","#85C1E9","#AED6F1"]

        vfig = go.Figure()
        for name, arr, col in zip(stages, arrays, STAGE_COLORS):
            vfig.add_trace(go.Violin(
                y=arr, name=name,
                box_visible=True,
                box=dict(fillcolor=hex_rgba(col, 0.55), line_color=hex_rgba(col, 0.9)),
                meanline_visible=True,
                meanline=dict(color=hex_rgba(col, 1.0), width=2),
                fillcolor=hex_rgba(col, 0.22),
                line=dict(color=hex_rgba(col, 0.85), width=1.5),
                opacity=0.90,
                points=False,
            ))
        vfig.update_layout(
            **LAYOUT,
            showlegend=True,
            height=400,
            margin=dict(l=65, r=30, t=65, b=60),
            yaxis=dict(
                title="Количество",
                gridcolor="rgba(200,210,220,0.35)",
                zeroline=False,
            ),
        )
        vfig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        vfig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        _plot_chart(vfig, use_container_width=True)
        st.session_state["_pdf_fig_violin"] = vfig

    with st.expander("95% доверительные интервалы по стадиям"):
        pct = lambda arr, q: int(np.percentile(arr, q))
        table_data = {
            "Стадия": stages,
            "P2.5": [pct(a, 2.5) for a in arrays],
            "Медиана": meds,
            "P97.5": [pct(a, 97.5) for a in arrays],
        }
        st.dataframe(table_data, use_container_width=True, hide_index=True)

# ── TAB: Беременность ─────────────────────────────────────────
with tab_preg:
    if _UI_OK:
        UI.tab_header("L2", "Беременность", "FORTUNE · KPIScore · трёхуровневая декомпозиция", "L2")
        UI.metric_row([
            ("На один перенос",          f"{res['p_per_transfer']*100:.1f}%",   ""),
            ("Cumul. если viable",       f"{res['p_cum_if_viable']*100:.1f}%",  "accent"),
            ("Успех цикла (от стим.)",   f"{res['p_overall_cycle']*100:.1f}%",  "highlight"),
            ("P(viable цикл)",           f"{res['p_viable']*100:.0f}%",         ""),
        ])
    col_a, col_b = st.columns(2)

    with col_a:
        (UI.section_header if _UI_OK else lambda _t: st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">{_t}</p>', unsafe_allow_html=True))("Шансы ≥k беременностей в цикле")
        p_at_least = pregnancy_count_distribution(res, max_k=12)
        _k_labels = [f"≥{k}" for k in range(1, 13)]
        _bar_colors = [hex_rgba(C["blue"], 0.75 - k * 0.04) for k in range(12)]
        bar_fig = go.Figure(go.Bar(
            x=_k_labels,
            y=[v * 100 for v in p_at_least],
            marker=dict(
                color=_bar_colors,
                line=dict(color=[hex_rgba(C["blue"], 0.90)] * 12, width=1.2),
            ),
            text=[f"{v*100:.1f}%" for v in p_at_least],
            textposition="outside",
            textfont=dict(family="Inter, Arial, sans-serif", size=11),
        ))
        bar_fig.add_hline(
            y=res["p_per_transfer"] * 100,
            line_dash="dash",
            line_color=hex_rgba(C["red"], 0.75),
            line_width=1.8,
            annotation_text=f"На перенос: {res['p_per_transfer']*100:.1f}%",
            annotation_font=dict(family="Inter, Arial, sans-serif", size=11, color=C["red"]),
            annotation_bgcolor="rgba(255,255,255,0.85)",
        )
        _y_max = max(v * 100 for v in p_at_least) * 1.25 + 5
        bar_fig.update_layout(
            **LAYOUT,
            height=400,
            margin=dict(l=65, r=30, t=60, b=60),
            yaxis=dict(
                range=[0, max(_y_max, 110)],
                title="Вероятность (%)",
                gridcolor="rgba(200,210,220,0.35)",
                zeroline=False,
            ),
            xaxis=dict(title="Число беременностей"),
            bargap=0.25,
        )
        bar_fig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        bar_fig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        _plot_chart(bar_fig, use_container_width=True)
        st.session_state["_pdf_fig_bar"] = bar_fig

    with col_b:
        (UI.section_header if _UI_OK else lambda _t: st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">{_t}</p>', unsafe_allow_html=True))("Вероятность по попыткам ЭКО")
        curve = res['attempt_curve']
        afig = go.Figure()
        afig.add_traces([go.Scatter(
            x=curve["attempts"] + curve["attempts"][::-1],
            y=[p * 100 for p in curve["p_hi"]] +
              [p * 100 for p in curve["p_lo"][::-1]],
            fill="toself", fillcolor=hex_rgba(C["blue"], 0.10),
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
        )])
        afig.add_trace(go.Scatter(
            x=curve["attempts"], y=[p * 100 for p in curve["p_sel_decay"]],
            mode="lines+markers", name="Аналит. снижение",
            line=dict(color=hex_rgba(C["orange"], 0.80), dash="dot", width=2),
            marker=dict(size=7, color=hex_rgba(C["orange"], 0.80)),
        ))
        afig.add_trace(go.Scatter(
            x=curve["attempts"], y=[p * 100 for p in curve["p_nn_raw"]],
            mode="lines+markers", name="NN (raw)",
            line=dict(color=hex_rgba(C["grey"], 0.70), dash="dash", width=2),
            marker=dict(size=7, color=hex_rgba(C["grey"], 0.70)),
        ))
        afig.add_trace(go.Scatter(
            x=curve["attempts"], y=[p * 100 for p in curve["p_mean"]],
            mode="lines+markers+text", name="Совмещённый",
            line=dict(color=hex_rgba(C["blue"], 1.0), width=3),
            marker=dict(size=11, color=hex_rgba(C["blue"], 0.90),
                        line=dict(width=1.5, color="white")),
            text=[f"{p*100:.1f}%" for p in curve["p_mean"]],
            textposition="top center",
            textfont=dict(family="Inter, Arial, sans-serif", size=11),
        ))
        afig.update_layout(
            **LAYOUT, height=400, margin=dict(l=65, r=30, t=65, b=60),
            xaxis=dict(title="Номер попытки", tickmode="array", tickvals=curve["attempts"]),
            yaxis=dict(title="Вероятность беременности (%)", gridcolor="rgba(200,210,220,0.35)"),
        )
        afig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        afig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        _plot_chart(afig, use_container_width=True)
        st.session_state["_pdf_fig_attempts"] = afig
        st.caption(f"Снижение per-attempt: \u03b1={curve['decay_alpha']:.2f}  (Malizia et al. NEJM 2009)")

    if _UI_OK:
        UI.result_box(
            f"<b>Трёхуровневая декомпозиция вероятности беременности:</b><br><br>"
            f"<b>[1] На один перенос:</b> {res['p_per_transfer']*100:.1f}%"
            f" &nbsp;&nbsp;<span style='color:#5A6B7B'>(если перенос состоится)</span><br>"
            f"<b>[2] Если цикл viable (≥1 перенос):</b> {res['p_cum_if_viable']*100:.1f}%"
            f" &nbsp;&nbsp;(95% CI: {res['rate_ci'][0]*100:.1f}–{res['rate_ci'][1]*100:.1f}%)<br>"
            f"<b>[3] Успех цикла (от стимуляции):</b> {res['p_overall_cycle']*100:.1f}%"
            f" &nbsp;&nbsp;= P(viable {res['p_viable']*100:.0f}%) × [2]",
            kind="info",
        )
    else:
        st.markdown(f"""
    <div class="result-box">
    <b>Трёхуровневая декомпозиция вероятности беременности:</b><br><br>
    <b>[1] На один перенос:</b> {res['p_per_transfer']*100:.1f}%
    &nbsp;&nbsp;(если перенос состоится)<br>
    <b>[2] Если цикл viable (≥1 перенос):</b> {res['p_cum_if_viable']*100:.1f}%
    &nbsp;&nbsp;(95% CI: {res['rate_ci'][0]*100:.1f}–{res['rate_ci'][1]*100:.1f}%)<br>
    <b>[3] Успех цикла (от стимуляции):</b> {res['p_overall_cycle']*100:.1f}%
    &nbsp;&nbsp;= P(viable {res['p_viable']*100:.0f}%) × [2]
    </div>
    """, unsafe_allow_html=True)

# ── TAB: Кластер ──────────────────────────────────────────────
with tab_cluster:
    if _UI_OK:
        UI.tab_header("L4", "Кластер", "Ближайший центроид · 18D z-пространство · k-means k=3", "L4")
    col_pca, col_info = st.columns([3, 2])

    with col_pca:
        (UI.section_header if _UI_OK else lambda _t: st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">{_t}</p>', unsafe_allow_html=True))("PCA(2) — кластерная принадлежность")
        n_syn = ca['n_synthetic']
        emb   = ca['pca_embedded']
        syn_2d    = emb[:n_syn]
        pat_2d    = emb[n_syn:]
        syn_labels= ca['synthetic_labels']
        expl = ca['pca_explained']

        pca_fig = go.Figure()
        # Облако синтетических точек (очень прозрачное)
        for c in (0, 1, 2):
            mask = syn_labels == c
            pca_fig.add_trace(go.Scatter(
                x=syn_2d[mask, 0], y=syn_2d[mask, 1],
                mode="markers",
                name=f"{CLUSTER_NAMES[c]} (фон)",
                marker=dict(
                    size=5,
                    color=hex_rgba(CLUSTER_HEX[c], 0.18),
                    line=dict(width=0),
                ),
                hoverinfo="skip",
            ))
        # Пациент по кластерам (насыщенные)
        for c in (0, 1, 2):
            mask = ca["assignments"] == c
            if mask.sum() == 0:
                continue
            pca_fig.add_trace(go.Scatter(
                x=pat_2d[mask, 0], y=pat_2d[mask, 1],
                mode="markers",
                name=f"Пациентка → {CLUSTER_NAMES[c]}",
                marker=dict(
                    size=8,
                    color=hex_rgba(CLUSTER_HEX[c], 0.85),
                    line=dict(width=1.2, color="white"),
                ),
                hoverinfo="skip",
            ))
        # Медиана пациентки — звезда
        mx, my = np.median(pat_2d[:, 0]), np.median(pat_2d[:, 1])
        pca_fig.add_trace(go.Scatter(
            x=[mx], y=[my], mode="markers+text",
            marker=dict(
                size=20, symbol="star",
                color=hex_rgba(C["red"], 0.95),
                line=dict(width=2, color="white"),
            ),
            text=["Пациентка"], textposition="top center",
            textfont=dict(family="Inter, Arial, sans-serif", size=12, color=C["red"]),
            name="Медиана пациентки",
        ))
        pca_fig.update_layout(
            **LAYOUT,
            height=440,
            margin=dict(l=65, r=30, t=65, b=60),
            xaxis=dict(title=f"PC1 ({expl[0]*100:.1f}%)", gridcolor="rgba(200,210,220,0.35)"),
            yaxis=dict(title=f"PC2 ({expl[1]*100:.1f}%)", gridcolor="rgba(200,210,220,0.35)"),
        )
        pca_fig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        pca_fig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        _plot_chart(pca_fig, use_container_width=True)
        st.session_state["_pdf_fig_pca"] = pca_fig

    with col_info:
        if _UI_OK:
            UI.section_header("Распределение по кластерам")
        else:
            st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">Распределение по кластерам</p>', unsafe_allow_html=True)
        probs = ca['cluster_probs']
        for c in (0,1,2):
            info = CLUSTER_INTERPRETATIONS[c]
            mark = " ← доминирует" if c == dom else ""
            if _UI_OK:
                kind_map = {0: "info", 1: "danger", 2: "success"}
                UI.result_box(
                    f"<b>C{c} — {info['name']}{mark}</b><br>"
                    f"Прогноз беременности: {info['preg_rate']*100:.0f}%<br>"
                    f"<b>Вероятность: {probs[c]*100:.1f}%</b>",
                    kind=kind_map[c],
                )
            else:
                css = ["cluster-c0","cluster-c1","cluster-c2"][c]
                st.markdown(f"""
            <div class="{css}">
            <b>C{c} — {info['name']}{mark}</b><br>
            Прогноз беременности: {info['preg_rate']*100:.0f}%<br>
            <b>Вероятность: {probs[c]*100:.1f}%</b>
            </div><br>
            """, unsafe_allow_html=True)

        st.markdown("---")
        dom_info = CLUSTER_INTERPRETATIONS[dom]
        st.markdown(f"**Клинические рекомендации — {dom_info['name']}:**")
        st.info(dom_info['clinical_notes'])

# ── TAB: Риски ────────────────────────────────────────────────
with tab_risk:
    if _UI_OK:
        UI.tab_header("", "Риски", "ССЯГ · пустой цикл · NB-распределение ооцитов", "L1")
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        (UI.section_header if _UI_OK else lambda _t: st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">{_t}</p>', unsafe_allow_html=True))("Профиль рисков")
        ohss  = res['ohss']
        empty = res['empty']
        rl = ["ССЯГ умеренный\n(15–19 ооц.)",
              "ССЯГ тяжёлый\n(≥20 ооц.)",
              "Любой ССЯГ",
              "Пустой цикл\n(нет бластоцист)",
              "Нет хор.кач.\nбластоцист"]
        rv = [ohss['p_moderate_ohss']*100, ohss['p_severe_ohss']*100,
              ohss['p_any_ohss']*100, empty['p_no_blast']*100,
              empty['p_no_good_blast']*100]
        rc = ["rgba(255,200,60,0.85)","rgba(220,60,60,0.85)",
              "rgba(255,140,40,0.85)","rgba(160,90,200,0.85)",
              "rgba(130,80,190,0.85)"]
        _risk_cols = [C["amber"], C["red"], C["orange"], C["purple"], C["purple"]]
        rfig = go.Figure(go.Bar(
            x=rl, y=rv,
            marker=dict(
                color=[hex_rgba(c, 0.28) for c in _risk_cols],   # полупрозрачная заливка
                line=dict(color=[hex_rgba(c, 0.65) for c in _risk_cols], width=1.2),
                pattern=dict(  # штриховка вместо ярких цветов
                    shape="/",
                    fgcolor=[hex_rgba(c, 0.55) for c in _risk_cols],
                    bgcolor="rgba(0,0,0,0)",
                    size=7, solidity=0.30,
                ),
            ),
            text=[f"{v:.1f}%" for v in rv],
            textposition="outside",
            textfont=dict(family="Inter, Arial, sans-serif", size=12, color="#5A6B7B"),
        ))
        _rv_max = max(rv) * 1.40 + 3
        rfig.update_layout(
            **LAYOUT,
            height=400,
            margin=dict(l=65, r=35, t=60, b=110),
            yaxis=dict(
                range=[0, max(_rv_max, 15)],
                title="Вероятность (%)",
                gridcolor="rgba(200,210,220,0.35)",
            ),
            xaxis=dict(tickfont=dict(size=10)),
            bargap=0.30,
        )
        rfig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        rfig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        _plot_chart(rfig, use_container_width=True)
        st.session_state["_pdf_fig_risks"] = rfig

    with col_r2:
        (UI.section_header if _UI_OK else lambda _t: st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">{_t}</p>', unsafe_allow_html=True))("Распределение Ооцитов (NB)")
        okk_arr = res['sim_okk']
        p_zero  = np.mean(okk_arr == 0)
        _pos    = okk_arr[okk_arr > 0]
        ofig = go.Figure()
        ofig.add_trace(go.Histogram(
            x=_pos, nbinsx=30, histnorm="probability density",
            opacity=0.85, name="Гистограмма", showlegend=False,
            marker=dict(
                color=hex_rgba(C["blue"], 0.28),
                line=dict(color=hex_rgba(C["blue"], 0.65), width=1.2),
                pattern=dict(shape="/", fgcolor=hex_rgba(C["blue"], 0.55),
                             bgcolor="rgba(0,0,0,0)", size=7, solidity=0.30),
            ),
        ))
        # Линия распределения (сглаженная плотность поверх гистограммы)
        try:
            from scipy.stats import gaussian_kde as _kde
            if _pos.size > 1 and np.ptp(_pos) > 0:
                _kx = np.linspace(_pos.min(), _pos.max(), 200)
                _ky = _kde(_pos)(_kx)
                ofig.add_trace(go.Scatter(
                    x=_kx, y=_ky, mode="lines", name="Плотность", showlegend=False,
                    line=dict(color=hex_rgba(C["red"], 0.9), width=2.5),
                ))
        except Exception:
            pass
        ofig.update_layout(
            **LAYOUT,
            height=400, showlegend=False,
            margin=dict(l=65, r=30, t=55, b=60),
            xaxis=dict(title="Число ооцитов"),
            yaxis=dict(title="Плотность", gridcolor="rgba(200,210,220,0.35)"),
        )
        ofig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        ofig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        _plot_chart(ofig, use_container_width=True)

        st.markdown(f"""
        | Показатель | Значение |
        |---|---|
        | Риск отмены цикла | **{p_zero*100:.1f}%** |
        | Медиана ООЦ | **{int(np.median(okk_arr))}** |
        | 5-й – 95-й перцентиль | {int(np.percentile(okk_arr,5))} – {int(np.percentile(okk_arr,95))} |
        """)

        if p_zero > 0.05:
            st.error(f"Высокий риск отмены цикла: {p_zero*100:.1f}%")
        elif p_zero > 0.02:
            st.warning(f"Умеренный риск отмены: {p_zero*100:.1f}%")
        else:
            st.success(f"Риск отмены низкий: {p_zero*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# ── TAB 6: Банкинг ────────────────────────────────────────────
with tab_bank:
    if _UI_OK:
        UI.tab_header_by_key("banking")
    eb = _eb
    if not eb:
        st.info("Модуль банкинга недоступен для этого расчёта.")
    else:
        (UI.section_header if _UI_OK else lambda _t: st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">{_t}</p>', unsafe_allow_html=True))("Эуплоидность и банкинг ооцитов")
        st.caption("Независимый модуль планирования банкинга; не влияет на S1–S6b")

        c1, c2, c3 = st.columns(3)
        c1.metric("P(эуплоид/MII)", f"{eb['p_per_mii']*100:.1f}%",
                  help=f"Возраст {eb['age']:.0f}, {sperm_label}")
        c2.metric("MII медиана пациентки",
                  f"{eb['patient_mii_median']}",
                  help="Из основного pipeline")
        if eb['forward_at_median']:
            c3.metric("Ожид. эуплоидных бласт.",
                      f"{eb['forward_at_median']['mean']:.1f}",
                      help="При текущей MII медиане")

        col_a_b, col_b_b = st.columns(2)

        with col_a_b:
            st.markdown("**Эуплоидные бластоцисты в зависимости от MII**")
            from scipy.stats import binom as _binom_b
            mii_range = list(range(1, 41))
            pm = eb['p_per_mii']
            exp_e = [M * pm for M in mii_range]
            lo_b  = [_binom_b.ppf(0.05, M, pm) for M in mii_range]
            hi_b  = [_binom_b.ppf(0.95, M, pm) for M in mii_range]
            f1 = go.Figure()
            # CI-полоса
            f1.add_trace(go.Scatter(
                x=mii_range + mii_range[::-1], y=hi_b + lo_b[::-1],
                fill="toself", fillcolor=hex_rgba(C["purple"], 0.10),
                line=dict(color="rgba(0,0,0,0)"), showlegend=False,
            ))
            # Линия ожидаемого
            f1.add_trace(go.Scatter(
                x=mii_range, y=exp_e, mode="lines",
                line=dict(color=hex_rgba(C["purple"], 0.90), width=2.5),
                name="Ожидаемое",
            ))
            # Горизонтали целевых эуплоидных
            for t, k in eb["euploid_for_preg"].items():
                if k:
                    f1.add_hline(
                        y=k, line_dash="dot",
                        line_color=hex_rgba(C["red"], 0.65),
                        line_width=1.5,
                        annotation_text=f"{int(t*100)}%→{k}",
                        annotation_font=dict(family="Inter, Arial, sans-serif",
                                             size=10, color=C["red"]),
                        annotation_bgcolor="rgba(255,255,255,0.85)",
                    )
            # Вертикаль пациентки
            if eb["patient_mii_median"]:
                f1.add_vline(
                    x=eb["patient_mii_median"],
                    line_dash="dash",
                    line_color=hex_rgba(C["grey"], 0.75),
                    line_width=1.8,
                    annotation_text=f"MII={eb['patient_mii_median']}",
                    annotation_font=dict(family="Inter, Arial, sans-serif",
                                         size=11, color=C["grey"]),
                    annotation_bgcolor="rgba(255,255,255,0.85)",
                )
            f1.update_layout(
                **LAYOUT,
                height=380,
                margin=dict(l=65, r=30, t=60, b=60),
                xaxis=dict(title="MII ооцитов"),
                yaxis=dict(title="Эуплоидные бластоцисты",
                           gridcolor="rgba(200,210,220,0.35)"),
            )
            f1.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            f1.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            _plot_chart(f1, use_container_width=True)

        with col_b_b:
            st.markdown("**Сколько MII набанковать (обратная задача)**")
            f2 = go.Figure()
            _bank_conf_colors = [C["teal"], C["blue"], C["purple"]]
            for ci_idx, cf in enumerate(eb["confidences"]):
                ys = [eb["mii_table"][k][cf] or 0 for k in eb["k_targets"]]
                _col = _bank_conf_colors[ci_idx % 3]
                f2.add_trace(go.Bar(
                    x=[f"{k} эупл." for k in eb["k_targets"]],
                    y=ys,
                    name=f"{int(cf*100)}% увер.",
                    marker=dict(
                        color=hex_rgba(_col, 0.65 + ci_idx * 0.08),
                        line=dict(color=hex_rgba(_col, 0.92), width=1.5),
                    ),
                    text=[str(v) if v else ">200" for v in ys],
                    textposition="outside",
                    textfont=dict(family="Inter, Arial, sans-serif", size=11),
                ))
            f2.update_layout(
                **LAYOUT,
                height=380,
                margin=dict(l=65, r=30, t=60, b=60),
                barmode="group",
                bargap=0.22, bargroupgap=0.06,
                xaxis=dict(title="Цель: эуплоидных бластоцист"),
                yaxis=dict(title="Нужно MII ооцитов",
                           gridcolor="rgba(200,210,220,0.35)"),
            )
            f2.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            f2.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            _plot_chart(f2, use_container_width=True)

        st.markdown("**Клиническая сводка для банкинга**")
        need50 = eb['euploid_for_preg'][0.50]
        need70 = eb['euploid_for_preg'][0.70]
        need90 = eb['euploid_for_preg'][0.90]
        def _mii_str(k, cf=0.80):
            if k is None: return "—"
            k2 = min(k, max(eb['k_targets']))
            v = eb['mii_table'].get(k2, {}).get(cf)
            return str(v) if v else ">200"

        rows_md = (
            f"| Цель | Эуплоидных нужно | MII набанковать (80% увер.) |\n"
            f"|---|---|---|\n"
            f"| 50% беременности | {need50} | {_mii_str(need50)} |\n"
            f"| 70% беременности | {need70} | {_mii_str(need70)} |\n"
            f"| 90% беременности | {need90} | {_mii_str(need90)} |\n"
        )
        st.markdown(rows_md)

        if eb['patient_mii_median'] and need50:
            fwd_med = eb['forward_at_median']['median']
            if fwd_med >= need50:
                st.success(
                    f"При MII медиане {eb['patient_mii_median']} ожидается "
                    f"~{eb['forward_at_median']['mean']:.1f} эуплоидных — "
                    f"достаточно для цели 50% ({need50})")
            else:
                st.warning(
                    f"При MII медиане {eb['patient_mii_median']} ожидается "
                    f"~{eb['forward_at_median']['mean']:.1f} эуплоидных — "
                    f"для 50% нужно {need50}. Рекомендуется банкинг "
                    f"дополнительных циклов.")

        with st.expander("Параметры модели банкинга"):
            st.markdown(f"""
            | Параметр | Значение |
            |---|---|
            | Источник спермы | {sperm_label} (страта Esteves: `{eb['sperm_source']}`) |
            | Возраст | {eb['age']:.0f} |
            | P(эуплоид/MII) | {eb['p_per_mii']*100:.1f}% |
            | P(беременность) на перенос, использованная для целей | {'—' if eb.get('p_transfer_used') is None else f"{eb['p_transfer_used']*100:.1f}%"} |

            *Esteves et al. — логистическая модель эуплоидной бластоцисты на MII (возраст × источник спермы)*
            """)
        st.caption("p — вероятность эуплоидной бластоцисты на один MII ооцит. Число переносов "
                   "для цели учитывает общий эффект цикла (переносы коррелированы). "
                   "Независимая модель планирования, не заменяет основной pipeline.")

# ── TAB 7: Лабораторный прогноз (CSDI Hybrid v3 — L5) ─────────
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
#  TRP TAB — Совокупный репродуктивный потенциал
# ══════════════════════════════════════════════════════════
with tab_trp:
    if not _TRP_OK:
        st.warning(f"TRP Engine недоступен: {_TRP_ERROR}. "
                   "Убедитесь, что trp_engine.py находится рядом с app.py.")
    else:
        st.markdown(
            "### Совокупный репродуктивный потенциал (TRP)",
        )
        st.caption(
            "Сколько биологически возможных попыток остаётся, "
            "как меняется шанс с каждым годом, и какова суммарная "
            "вероятность хотя бы одной клинической беременности до закрытия окна."
        )

        # ── Входные данные TRP ──────────────────────────────
        # 7.1: якорь — вероятность цикла по всему стеку L1–L7 (та же, что на
        # карточке результата). Если перенос в текущем цикле невозможен,
        # якорь пересчитывается для нового цикла без текущих наблюдений.
        trp_inp = _build_trp_inputs(
            current_age  = float(age),
            current_amh  = float(amh),
            current_afc  = int(afc),
            current_bmi  = float(bmi),
            p_base       = None,
        )
        trp_inp.sperm_source = sperm_source
        if trp_inp.desired_children != 1:
            st.warning("TRP 7.1 оценивает время до первой беременности; "
                       "расчёт выполняется для одного ребёнка.")
            trp_inp.desired_children = 1

        run_trp_btn = st.button(
            "Рассчитать TRP",
            key="run_trp",
            type="primary",
            use_container_width=False,
        )

        if run_trp_btn:
            with st.spinner("MC-симуляция траекторий... (~2 сек)"):
                try:
                    _trp_anchor, _trp_anchor_source = _dt_bridge.trp_anchor(_dt71)
                    trp_inp.p_base_override = _trp_anchor
                    st.caption(f"Якорь TRP: {_trp_anchor*100:.1f}% ({_trp_anchor_source})")
                    _trp_res = _compute_trp(trp_inp)
                    st.session_state["_trp_result"] = _trp_res
                except Exception as _trp_exc:
                    st.error(f"Ошибка TRP: {_trp_exc}")
                    import traceback
                    st.code(traceback.format_exc())

        if "_trp_result" in st.session_state:
            try:
                _build_trp_tab(st.session_state["_trp_result"], theme_fn=_apply_plot_theme)
            except Exception as _trp_render_exc:
                st.error(f"Ошибка отображения TRP: {_trp_render_exc}")
        else:
            st.info(
                "Заполните параметры горизонта выше и нажмите "
                "**Рассчитать TRP** для запуска симуляции."
            )

with tab_diff:
    if _UI_OK:
        UI.tab_header("L5", "Diffusion", "CSDI Hybrid v3 · ~15 000 циклов · конформные предиктивные интервалы", "L5")
    (UI.section_header if _UI_OK else lambda _t: st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">{_t}</p>', unsafe_allow_html=True))("L5 · Лабораторный прогноз (CSDI Hybrid v3)")
    st.markdown("""
    Гибридная генеративная модель обучена на лабораторном этапе ЭКО
    и генерирует синтетические пары **(Число Bl, Число Bl хор.кач.)** без
    параметрических допущений MC-пайплайна. Частоты вычисляются аналитически
    из сгенерированных пар — это устраняет независимый дрейф числителя и
    знаменателя. Исход беременности предсказывает откалиброванный LightGBM + Platt.
    """)

    if csdi_model is None:
        # ── Модель не загружена ───────────────────────────────
        if _UI_OK:
            UI.result_box(
                "<b>CSDI Hybrid v3 не загружен</b><br><br>"
                "Для активации этой вкладки необходимо:<br>"
                "1. Убедиться, что <code>src/embryo_csdi_v3.py</code> присутствует<br>"
                "2. Обучить модель: <code>python src/embryo_csdi_v3.py</code><br>"
                "3. Скопировать папку <code>embryo_v3_model/</code> в <code>models/</code><br><br>"
                "MC-результаты (вкладки 1–6) работают независимо.",
                kind="warning",
            )
        else:
            st.markdown("""
        <div class="diff-warn">
        <b>CSDI Hybrid v3 не загружен</b><br><br>
        Для активации этой вкладки необходимо:<br>
        1. Убедиться, что <code>src/embryo_csdi_v3.py</code> присутствует<br>
        2. Обучить модель: <code>python src/embryo_csdi_v3.py</code><br>
        3. Скопировать папку <code>embryo_v3_model/</code> в <code>models/</code><br><br>
        MC-результаты (вкладки 1–6) работают независимо.
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.get("csdi_result") is None:
        # ── CSDI не запускалась или не завершилась для этого случая ──
        from presentation import csdi_note as _csdi_note
        st.info(_csdi_note(_csdi_applicability, "en" if _LANG == "English" else "ru"))
    else:
        # ── Вход CSDI сформирован ядром 7.1 (профиль переноса, KPIScore
        # по формуле обучения, число фолликулов на пункции) ─────────
        _csdi_res = st.session_state["csdi_result"]
        _patient_csdi = _csdi_res.get("conditioning", {})
        _fert_rate  = float(_patient_csdi.get("Частота оплодотворения", 0.0))
        if not _csdi_applicability.get("used_in_fusion"):
            from presentation import csdi_note as _csdi_note
            st.warning(_csdi_note(_csdi_applicability, "en" if _LANG == "English" else "ru"))

        _csdi_df   = _csdi_res['samples']
        _p_csdi    = _csdi_res['P_pregnancy']
        _p_mc      = _dt71["res_transfer"]['p_per_transfer']
        _opt_thr   = csdi_model.best_threshold
        _ci        = _csdi_res['CI_95']

        # ── Верхний ряд: ключевые метрики ────────────────────
        st.markdown("#### Ключевые показатели")
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("MC P(беременность)",
                   f"{_p_mc*100:.1f}%", help="L1+L2 Monte Carlo")
        dc2.metric("CSDI P(беременность)",
                   f"{_p_csdi*100:.1f}%",
                   delta=f"{(_p_csdi - _p_mc)*100:+.1f} пп vs MC",
                   delta_color="normal",
                   help=f"LightGBM + Platt | 95% CI: {_ci[0]*100:.1f}–{_ci[1]*100:.1f}%")
        dc3.metric("Blast rate (CSDI медиана)",
                   f"{_csdi_res['blast_rate_median']*100:.1f}%",
                   help="Частота формирования бластоцист — CSDI")
        dc4.metric("TGBDR (CSDI медиана)",
                   f"{_csdi_res['good_rate_median']*100:.1f}%",
                   help="Частота бластоцист хор.кач. — CSDI")

        # Прогноз по порогу
        _pred_ok = _p_csdi >= _opt_thr
        _pred_label = "Благоприятный" if _pred_ok else "Осторожный"
        _pred_color = "#2E7D32" if _pred_ok else "#E65100"
        st.markdown(
            f'<div class="{"diff-box" if _pred_ok else "diff-warn"}">'
            f'<b>Прогноз CSDI (порог {_opt_thr:.2f}):</b> '
            f'<span style="color:{_pred_color}">{_pred_label}</span>'
            f'&nbsp;&nbsp;P = {_p_csdi*100:.1f}% &nbsp;'
            f'95% CI: {_ci[0]*100:.1f}–{_ci[1]*100:.1f}%'
            f'</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Основные графики: два столбца ────────────────────
        st.markdown("#### Сравнение распределений целевых переменных")
        col_d1, col_d2 = st.columns(2)

        # ─ График 1: Бластоцисты всего ───────────────────────
        with col_d1:
            st.markdown("**Бластоцисты всего — MC vs CSDI**")
            _mc_bl   = res['sim_blasts']
            _csdi_bl = _csdi_df["Число Bl"].values
            _ks_bl, _p_bl = ks_2samp(_mc_bl, _csdi_bl)

            blfig = go.Figure()
            blfig.add_trace(go.Histogram(
                x=_mc_bl, name="MC pipeline", opacity=0.58,
                histnorm="probability density",
                marker=dict(
                    color=hex_rgba(C["blue"], 0.56),
                    line=dict(color=hex_rgba(C["blue"], 0.95), width=1.4),
                    pattern=dict(shape="/", solidity=0.16),
                ),
                xbins=dict(size=1),
            ))
            blfig.add_trace(go.Histogram(
                x=_csdi_bl, name="CSDI", opacity=0.58,
                histnorm="probability density",
                marker=dict(
                    color=hex_rgba(C["green"], 0.56),
                    line=dict(color=hex_rgba(C["green"], 0.95), width=1.4),
                    pattern=dict(shape="\\", solidity=0.16),
                ),
                xbins=dict(size=1),
            ))
            blfig.update_layout(
                **LAYOUT,
                barmode="overlay",
                height=340,
                margin=dict(l=60, r=30, t=50, b=55),
                xaxis=dict(title="Число бластоцист"),
                yaxis=dict(title="Плотность"),
            )
            blfig.add_annotation(
                text=f"KS={_ks_bl:.3f}  p={_p_bl:.3f}  {'✓ схожи' if _p_bl > 0.05 else '≠ различны'}",
                xref="paper", yref="paper", x=0.99, y=0.99,
                showarrow=False,
                font=dict(size=10, color=C["green"] if _p_bl > 0.05 else C["red"]),
                bgcolor="rgba(244,247,250,0.90)",
                bordercolor="rgba(115,132,145,0.28)", borderwidth=1, borderpad=4,
            )
            blfig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            blfig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            _plot_chart(blfig, use_container_width=True)

        # ─ График 2: TGBDR ───────────────────────────────────
        with col_d2:
            st.markdown("**TGBDR — MC vs CSDI**")
            _mc_tgbdr   = res['sim_good'] / np.maximum(res['sim_pn2'], 1)
            _mc_tgbdr   = np.clip(_mc_tgbdr, 0, 1)
            _csdi_tgbdr = _csdi_df[
                "Частота формирования бластоцист хорошего качества"].values
            _ks_tgbdr, _p_tgbdr = ks_2samp(_mc_tgbdr, _csdi_tgbdr)

            tfig = go.Figure()
            tfig.add_trace(go.Histogram(
                x=_mc_tgbdr * 100, name="MC pipeline", opacity=0.58,
                histnorm="probability density",
                marker=dict(
                    color=hex_rgba(C["blue"], 0.56),
                    line=dict(color=hex_rgba(C["blue"], 0.95), width=1.4),
                    pattern=dict(shape="/", solidity=0.16),
                ),
                xbins=dict(size=2),
            ))
            tfig.add_trace(go.Histogram(
                x=_csdi_tgbdr * 100, name="CSDI", opacity=0.58,
                histnorm="probability density",
                marker=dict(
                    color=hex_rgba(C["green"], 0.56),
                    line=dict(color=hex_rgba(C["green"], 0.95), width=1.4),
                    pattern=dict(shape="\\", solidity=0.16),
                ),
                xbins=dict(size=2),
            ))
            tfig.update_layout(
                **LAYOUT,
                barmode="overlay",
                height=340,
                margin=dict(l=60, r=30, t=50, b=55),
                xaxis=dict(title="TGBDR (%)"),
                yaxis=dict(title="Плотность"),
            )
            tfig.add_annotation(
                text=f"KS={_ks_tgbdr:.3f}  p={_p_tgbdr:.3f}  {'✓ схожи' if _p_tgbdr > 0.05 else '≠ различны'}",
                xref="paper", yref="paper", x=0.99, y=0.99,
                showarrow=False,
                font=dict(size=10, color=C["green"] if _p_tgbdr > 0.05 else C["red"]),
                bgcolor="rgba(244,247,250,0.90)",
                bordercolor="rgba(115,132,145,0.28)", borderwidth=1, borderpad=4,
            )
            tfig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            tfig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            _plot_chart(tfig, use_container_width=True)

        # ── Второй ряд: промежуточные переменные ─────────────
        st.markdown("#### Промежуточные стадии эмбриогенеза")
        col_d3, col_d4 = st.columns(2)

        with col_d3:
            st.markdown("**Бластоцисты хор. кач. — MC vs CSDI**")
            _mc_gb   = res['sim_good']
            _csdi_gb = _csdi_df["Число Bl хор.кач-ва"].values
            _ks_gb, _p_gb = ks_2samp(_mc_gb, _csdi_gb)

            gbfig = go.Figure()
            gbfig.add_trace(go.Histogram(
                x=_mc_gb, name="MC", opacity=0.58,
                histnorm="probability density",
                marker=dict(
                    color=hex_rgba(C["blue"], 0.56),
                    line=dict(color=hex_rgba(C["blue"], 0.95), width=1.4),
                    pattern=dict(shape="/", solidity=0.16),
                ),
                xbins=dict(size=1),
            ))
            gbfig.add_trace(go.Histogram(
                x=_csdi_gb, name="CSDI", opacity=0.58,
                histnorm="probability density",
                marker=dict(
                    color=hex_rgba(C["green"], 0.56),
                    line=dict(color=hex_rgba(C["green"], 0.95), width=1.4),
                    pattern=dict(shape="\\", solidity=0.16),
                ),
                xbins=dict(size=1),
            ))
            gbfig.update_layout(
                **LAYOUT,
                barmode="overlay",
                height=280,
                xaxis=dict(title="Число бластоцист хор. кач."),
                yaxis=dict(title="Плотность"),
            )
            gbfig.add_annotation(
                text=f"KS={_ks_gb:.3f}  p={_p_gb:.3f}",
                xref="paper", yref="paper", x=0.99, y=0.99,
                showarrow=False, font=dict(size=10),
                bgcolor="rgba(244,247,250,0.90)",
                bordercolor="rgba(115,132,145,0.28)", borderwidth=1, borderpad=4,
            )
            gbfig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            gbfig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            _plot_chart(gbfig, use_container_width=True)

        with col_d4:
            st.markdown("**Blast rate — MC vs CSDI**")
            _mc_br   = res['sim_blasts'] / np.maximum(res['sim_pn2'], 1)
            _mc_br   = np.clip(_mc_br, 0, 1)
            _csdi_br = _csdi_df["Частота формирования бластоцист"].values
            _ks_br, _p_br = ks_2samp(_mc_br, _csdi_br)

            brfig = go.Figure()
            brfig.add_trace(go.Histogram(
                x=_mc_br * 100, name="MC", opacity=0.58,
                histnorm="probability density",
                marker=dict(
                    color=hex_rgba(C["blue"], 0.56),
                    line=dict(color=hex_rgba(C["blue"], 0.95), width=1.4),
                    pattern=dict(shape="/", solidity=0.16),
                ),
                xbins=dict(size=2),
            ))
            brfig.add_trace(go.Histogram(
                x=_csdi_br * 100, name="CSDI", opacity=0.58,
                histnorm="probability density",
                marker=dict(
                    color=hex_rgba(C["green"], 0.56),
                    line=dict(color=hex_rgba(C["green"], 0.95), width=1.4),
                    pattern=dict(shape="\\", solidity=0.16),
                ),
                xbins=dict(size=2),
            ))
            brfig.update_layout(
                **LAYOUT,
                barmode="overlay",
                height=280,
                xaxis=dict(title="Blast rate (%)"),
                yaxis=dict(title="Плотность"),
            )
            brfig.add_annotation(
                text=f"KS={_ks_br:.3f}  p={_p_br:.3f}",
                xref="paper", yref="paper", x=0.99, y=0.99,
                showarrow=False, font=dict(size=10),
                bgcolor="rgba(244,247,250,0.90)",
                bordercolor="rgba(115,132,145,0.28)", borderwidth=1, borderpad=4,
            )
            brfig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            brfig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            _plot_chart(brfig, use_container_width=True)

        # ── KS-сводная таблица ────────────────────────────────
        st.markdown("#### KS-тест: итоговая верификация")
        _ks_table = {
            "Переменная": [
                "Бластоцисты всего", "TGBDR (хор. бластоцисты)",
                "Бластоцисты хор. кач.", "Blast rate"
            ],
            "KS-статистика": [
                f"{_ks_bl:.3f}", f"{_ks_tgbdr:.3f}",
                f"{_ks_gb:.3f}", f"{_ks_br:.3f}"
            ],
            "p-value": [
                f"{_p_bl:.4f}", f"{_p_tgbdr:.4f}",
                f"{_p_gb:.4f}", f"{_p_br:.4f}"
            ],
            "Вывод": [
                "схожи" if _p_bl    > 0.05 else "различны",
                "схожи" if _p_tgbdr > 0.05 else "различны",
                "схожи" if _p_gb    > 0.05 else "различны",
                "схожи" if _p_br    > 0.05 else "различны",
            ],
        }
        st.dataframe(_ks_table, use_container_width=True, hide_index=True)

        # ── Интерпретация ─────────────────────────────────────
        _n_pass = sum([
            _p_bl    > 0.05, _p_tgbdr > 0.05,
            _p_gb    > 0.05, _p_br    > 0.05
        ])

        if _n_pass >= 2:
            _box_class = "diff-box"
            _icon = ""
            _verdict = "подтверждает"
            _color_word = "сходство"
        else:
            _box_class = "diff-warn"
            _icon = ""
            _verdict = "не подтверждает"
            _color_word = "расхождение"

        _ks_html = (
            f"<b>{_icon} Интерпретация верификации</b><br>"
            f"CSDI Hybrid v3 (1000 траекторий, DDIM 50 шагов) генерирует "
            f"эмбриологический каскад без параметрических допущений MC-пайплайна.<br>"
            f"KS-тест <b>{_verdict}</b> статистическое {_color_word} "
            f"финальных распределений: {_n_pass}/4 переменных прошли порог p&nbsp;>&nbsp;0.05.<br>"
            f"<b>Примечание:</b> CSDI предсказывает P(беременность) = <b>{_p_csdi*100:.1f}%</b> "
            f"через калиброванный LightGBM (ECE ≈ 0.03), порог {_opt_thr:.2f}. "
            f"Различия промежуточных переменных при совпадении целевых "
            f"соответствуют принципу <i>equifinality</i> и подтверждают "
            f"архитектуру MC-пайплайна."
        )
        if _UI_OK:
            UI.result_box(_ks_html, kind="success" if _n_pass >= 2 else "warning")
        else:
            st.markdown(f"""
        <div class="{_box_class}">
        <b>{_icon} Интерпретация верификации</b><br>
        CSDI Hybrid v3 (1000 траекторий, DDIM 50 шагов) генерирует
        эмбриологический каскад без параметрических допущений MC-пайплайна.<br>
        KS-тест <b>{_verdict}</b> статистическое {_color_word}
        финальных распределений: {_n_pass}/4 переменных прошли порог p&nbsp;>&nbsp;0.05.<br>
        <b>Примечание:</b> CSDI предсказывает P(беременность) = <b>{_p_csdi*100:.1f}%</b>
        через калиброванный LightGBM (ECE ≈ 0.03), порог {_opt_thr:.2f}.
        Различия промежуточных переменных при совпадении целевых
        соответствуют принципу <i>equifinality</i> и подтверждают
        архитектуру MC-пайплайна.
        </div>
        """, unsafe_allow_html=True)

        # ── 90% предиктивные интервалы ────────────────────────
        _pi90 = _csdi_res.get('PI_90_counts', {})
        _pi50 = _csdi_res.get('PI_50_counts', {})

        if _pi90:
            st.markdown("#### Конформальные предиктивные интервалы (COUNT)")
            _pi_table = {
                "Признак": list(_pi90.keys()),
                "50% PI":  [f"[{v[0]:.0f}, {v[1]:.0f}]"
                             for v in _pi50.values()] if _pi50 else ["—"]*len(_pi90),
                "90% PI":  [f"[{v[0]:.0f}, {v[1]:.0f}]"
                             for v in _pi90.values()],
                "Медиана": [
                    f"{_csdi_res['blast_total_median']:.0f}",
                    f"{_csdi_res['good_blast_median']:.0f}",
                ],
            }
            st.dataframe(_pi_table, use_container_width=True, hide_index=True)

        # ── Параметры условия ─────────────────────────────────
        with st.expander("Параметры CSDI-условия (conditioning)"):
            st.markdown(f"""
            CSDI-модель обусловлена на upstream-результатах MC
            (медианы сценариев с переносом):

            | Параметр | Значение (из MC) |
            |---|---|
            | Фолликулов на пункции | {_patient_csdi.get('Количество фолликулов', 0):.0f} |
            | ОКК (MC медиана) | {_patient_csdi.get('Число ОКК', 0):.0f} |
            | MII как инсеминированные | {_patient_csdi.get('Число инсеминированных', 0):.0f} |
            | 2PN (MC медиана) | {_patient_csdi.get('2 pN', 0):.0f} |
            | Частота получения ОКК | {_patient_csdi.get('Частота получения ОКК', 0):.2f} |
            | Частота оплодотворения | {_fert_rate:.2f} |
            | KPIScore (формула обучения) | {_patient_csdi.get('KPIScore', 0):.1f} |

            *Генерация: 1000 траекторий, DDIM 50 шагов*
            """)

# ── TAB 8: GAT Graph ──────────────────────────────────────────
with tab_gat:
    if _UI_OK:
        UI.tab_header_by_key("gat")
    (UI.section_header if _UI_OK else lambda _t: st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">{_t}</p>', unsafe_allow_html=True))("Graph Attention Transformer — граф клинических соседей")

    if not _gnn_bundle.get('available'):
        _err = _gnn_bundle.get('error', 'Модель не загружена')
        st.info(f"GNN модель недоступна: {_err}\n\n"
                f"Поместите `gnn_ivf_model.pt` в папку `models/` и перезапустите приложение.")

    elif _p_gnn_raw is None:
        st.info("Нажмите **Запустить расчёт** чтобы получить предсказание Graph Transformer.")

    else:
        # ── Метрики ───────────────────────────────────────────
        g1, g2, g3 = st.columns(3)
        g1.metric(
            "Graph Transformer",
            f"{_p_gnn_raw*100:.1f}%",
            help="Чистый выход Graph Attention Transformer (без ансамблирования)",
        )
        g2.metric(
            "GAT Ансамбль",
            f"{_p_gnn_ens*100:.1f}%" if _p_gnn_ens is not None else "—",
            help=f"Взвешенный ансамбль: {_w_gnn:.2f}×GNN + {round(1-_w_gnn,2):.2f}×KAT",
        )
        if _p_kat_raw is not None and _p_gnn_ens is not None:
            _delta = (_p_gnn_ens - _p_kat_raw) * 100
            g3.metric(
                "ΔGAT vs KAT",
                f"{_delta:+.1f} п.п.",
                delta=f"{_delta:+.1f}%",
                help="Разница между ансамблем GAT и чистым KAT-скором",
            )
        else:
            g3.metric("KAT (для сравнения)",
                      f"{_p_kat_raw*100:.1f}%" if _p_kat_raw else "—")

        st.markdown("---")

        # ── Основной график ───────────────────────────────────
        _fig_gnn_tab = st.session_state.get('_pdf_fig_gnn')
        if _fig_gnn_tab is not None:
            _plot_chart(_fig_gnn_tab, use_container_width=True)
        else:
            # Строим на лету если не был сохранён (например после reload)
            try:
                _fig_gnn_tab = _build_gnn_figure(
                    _gnn_result,
                    gnn_prob      = _p_gnn_raw,
                    ensemble_prob = _p_gnn_ens,
                )
                _fig_gnn_tab = _apply_gnn_style(_fig_gnn_tab)
                if _fig_gnn_tab is not None:
                    st.session_state['_pdf_fig_gnn'] = _fig_gnn_tab
                    _plot_chart(_fig_gnn_tab, use_container_width=True)
                else:
                    st.warning("Не удалось построить граф: данные соседей недоступны.")
            except Exception as _gnn_fig_err:
                st.error(f"Ошибка построения графа: {_gnn_fig_err}")

        # ── Пояснение ─────────────────────────────────────────
        with st.expander("Как читать этот график"):
            st.markdown(f"""
**Левая панель — сетевой граф:**
- **Звезда** в центре — текущая пациентка
- **Круги** — 10 клинически наиболее похожих пациентов из обучающей когорты
- **Цвет** узла: зелёный = высокая GNN-вероятность, красный = низкая
- **Размер** узла и **толщина** ребра ∝ косинусное сходство профилей
- Сходство вычисляется по клиническим показателям: возраст, ОКК, бластоцисты,
  частоты оплодотворения / бластуляции и др. *(без учёта KAT-скора)*

**Правая панель — распределение GNN-вероятностей соседей:**
- Каждый бар = один сосед, отсортированы по вероятности
- Пунктирная линия = вероятность **текущей пациентки** по Graph Transformer
- Серая линия = медиана вероятностей среди соседей

**Интерпретация:**
Если пациентка попадает в область высоких вероятностей среди похожих случаев —
это дополнительный аргумент в пользу оптимистичного прогноза.
Если её позиция ниже медианы соседей — Graph Transformer выявляет
дополнительные неблагоприятные паттерны относительно похожих пациентов.

> Ансамбль: **{_w_gnn:.0%}×GNN + {1-_w_gnn:.0%}×KAT**
            """)

        # ── Технические детали (для исследователя) ───────────
        with st.expander("Технические параметры модели"):
            _cfg = _gnn_bundle.get('cfg', {})
            _n_train = (len(_gnn_bundle['train_X_scaled'])
                        if _gnn_bundle.get('train_X_scaled') is not None else '—')
            st.markdown(f"""
| Параметр | Значение |
|---|---|
| Обучающая выборка | {_n_train} протоколов |
| Архитектура | TransformerConv × {_cfg.get('n_layers', 3)} слоя |
| hidden_dim / heads | {_cfg.get('hidden_dim', 48)} / {_cfg.get('n_heads', 4)} |
| k соседей (топология) | {_cfg.get('k_neighbors', 10)} |
| Признаков (топология) | 18 клинических (без KAT-скоров) |
| Признаков (embedding) | {len(_gnn_bundle.get('features', []))} |
| AUC (5-fold CV) | ~0.63 |
| Вес в ансамбле | w_GNN = {_w_gnn:.2f} |
            """)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.caption("IVF Digital Twin v7.1  ·  from in vitro to in silico  ·  "
           "embryossa@gmail.com  ·  "
           "Research prototype — not for standalone clinical use")


# ══════════════════════════════════════════════════════════
#  PDF ОТЧЁТ
# ══════════════════════════════════════════════════════════
st.markdown("---")
# ── TAB 9: BEFE (L7) — Bayesian Evidence Fusion ───────────────
with tab_befe:
    if not _BEFE_OK:
        st.warning(f"BEFE недоступен: {_BEFE_ERR}")
    else:
        try:
            # 7.1: тот же результат BEFE, что на карточке и в PDF
            # (compute_l7_posterior ядра, CSDI с проверкой применимости).
            if _befe_res is None:
                st.info("BEFE недоступен для этого расчёта.")
            else:
                # Таблица входов показывает пары (значение, источник); служебные
                # поля 7.1 (csdi_applicability, ood_*) выводятся отдельно.
                render_befe_tab(_befe_res, {k: v for k, v in _befe_map.items()
                                            if isinstance(v, tuple) and len(v) == 2})
                if _befe_map.get("ood_available"):
                    _ood_unassessed = ", ".join(_befe_map.get("ood_unassessed") or []) or "—"
                    st.caption(f"OOD: эталон — {_befe_map.get('ood_reference')}; "
                               f"не оцениваются: {_ood_unassessed}")
                else:
                    st.caption("OOD-контроль выключен: нет обучающей статистики "
                               "(models/befe_ood_stats.npz).")
        except Exception as _befe_exc:
            st.error(f"Ошибка отображения BEFE: {_befe_exc}")


# ── TAB: LLM Консультант ──────────────────────────────────────────────────
with tab_llm:
    if _UI_OK:
        UI.tab_header("LLM", "Консультант",
                      "MedGemma · клинический нарратив · диалог", "")

    # ── Загрузка модуля и проверка Ollama ─────────────────────────────────
    _lc_ok   = False
    _lc_err  = ""
    _lc_live = False
    try:
        import llm_consultant as _LC
        _lc_ok   = True
        _lc_live = _LC.health_check()
    except Exception as _lc_import_err:
        _lc_err = str(_lc_import_err)

    if not _lc_ok:
        st.error(f"llm_consultant.py недоступен: {_lc_err}")
        st.caption("Убедитесь, что файл находится рядом с app.py.")
    else:
        # Статус Ollama
        if _lc_live:
            st.success(f"Ollama доступна · {_LC.OLLAMA_HOST} · модель: {_LC.MEDGEMMA}")
        else:
            st.warning(
                f"Ollama не отвечает на {_LC.OLLAMA_HOST}. "
                "Запустите `ollama serve` и убедитесь, что модель загружена "
                f"(`ollama pull {_LC.MEDGEMMA}`)."
            )

        st.divider()

        # ── Настройки запроса ─────────────────────────────────────────────
        _llm_col1, _llm_col2 = st.columns([3, 2])
        with _llm_col1:
            _llm_mode = st.radio(
                "Режим анализа",
                ["Клиническое резюме (Tier 0)",
                 "Анализ ансамбля (Tier 1)"],
                horizontal=True,
                key="_llm_mode_radio",
                help=(
                    "Tier 0 — клинический нарратив: объяснение прогноза, "
                    "сценарии цикла, банкинг, неопределённость.\n"
                    "Tier 1 — матрица согласованности моделей, CSDI-якорь, "
                    "GAT-соседи, ранг неопределённости."
                ),
            )
        with _llm_col2:
            _llm_tier = 0 if "Tier 0" in _llm_mode else 1
            if _llm_tier == 0:
                _llm_style = st.radio(
                    "Стиль",
                    ["narrative", "concise"],
                    horizontal=True,
                    key="_llm_style_radio",
                    help="narrative — полный нарратив (~400 слов, медленнее); "
                         "concise — краткий (~150 слов, быстро).",
                )
            else:
                _llm_style = "concise"  # Tier 1 не использует style

        # ── Выбор модели ──────────────────────────────────────────────────
        _available_models = _LC.list_models()
        if _available_models:
            _default_model = (
                _LC.MEDGEMMA if _LC.MEDGEMMA in _available_models
                else _available_models[0]
            )
            _llm_model = st.selectbox(
                "Модель Ollama",
                _available_models,
                index=_available_models.index(_default_model),
                key="_llm_model_sel",
                help=(
                    "Выберите любую загруженную модель.\n"
                    "medgemma1.5 — медицинская специализация.\n"
                    "gemma4 / gemma3 — более свободный язык, лучший нарратив.\n"
                    "Добавить модель: ollama pull <имя>"
                ),
            )
        else:
            _llm_model = _LC.MEDGEMMA
            st.caption(f"Модель по умолчанию: `{_llm_model}`")

        # ── Кнопки управления ─────────────────────────────────────────────
        _btn_col1, _btn_col2, _btn_col3 = st.columns([2, 1, 1])
        with _btn_col1:
            _llm_start_btn = st.button(
                "Запустить анализ",
                key="_llm_start",
                type="primary",
                use_container_width=True,
                disabled=not _lc_live,
            )
        with _btn_col2:
            _llm_clear_btn = st.button(
                "Очистить диалог",
                key="_llm_clear",
                use_container_width=True,
            )
        with _btn_col3:
            _llm_np = _LC._STYLE_NUM_PREDICT.get(_llm_style, 1200)
            _llm_to = _LC._TIMEOUT_READ
            st.caption(
                f"max_tokens: {_llm_np}\n"
                f"timeout: {_llm_to} с"
            )

        # ── Инициализация состояния диалога ──────────────────────────────
        if "_llm_chat" not in st.session_state:
            st.session_state["_llm_chat"] = []   # [{"role","content"}, ...]
        if "_llm_tier_used" not in st.session_state:
            st.session_state["_llm_tier_used"] = 0

        if _llm_clear_btn:
            st.session_state["_llm_chat"] = []
            st.rerun()

        # ── Первый запуск: начальный анализ ───────────────────────────────
        _just_streamed = False                       # [FIX] анти-дубль вывода LLM
        if _llm_start_btn:
            st.session_state["_llm_chat"] = []   # сбрасываем при новом запуске
            st.session_state["_llm_tier_used"] = _llm_tier
            with st.chat_message("assistant", avatar="🤖"):
                if _llm_tier == 0:
                    _response = st.write_stream(
                        _LC.consult_stream(
                            globals(),
                            style=_llm_style,
                            model=_llm_model,
                        )
                    )
                else:
                    _response = st.write_stream(
                        _LC.analyse_ensemble_stream(globals(), model=_llm_model)
                    )
            st.session_state["_llm_chat"].append(
                {"role": "assistant", "content": _response}
            )
            _just_streamed = True                    # [FIX] ответ уже показан вживую

        # ── Отображение истории диалога ───────────────────────────────────
        # [FIX] На старте ответ уже отрисован через write_stream выше — не
        # дублируем его в истории на ЭТОМ же прогоне. На последующих перерисовках
        # _just_streamed = False, и история показывает ответ один раз.
        _chat_to_show = (st.session_state["_llm_chat"][:-1]
                         if _just_streamed else st.session_state["_llm_chat"])
        for _msg in _chat_to_show:
            _avatar = "🤖" if _msg["role"] == "assistant" else "👨‍⚕️"
            with st.chat_message(_msg["role"], avatar=_avatar):
                st.markdown(_msg["content"])

        # ── Follow-up вопрос ─────────────────────────────────────────────
        if st.session_state["_llm_chat"]:
            _follow = st.chat_input(
                "Уточняющий вопрос по результатам...",
                key="_llm_follow",
            )
            if _follow and _lc_live:
                # Показываем вопрос
                with st.chat_message("user", avatar="👨‍⚕️"):
                    st.markdown(_follow)
                st.session_state["_llm_chat"].append(
                    {"role": "user", "content": _follow}
                )
                # История без последнего user (он уже в messages в chat_stream)
                _hist_for_llm = st.session_state["_llm_chat"][:-1]
                # Стримим ответ
                with st.chat_message("assistant", avatar="🤖"):
                    _follow_resp = st.write_stream(
                        _LC.chat_stream(
                            globals(),
                            history=_hist_for_llm,
                            question=_follow,
                            tier=st.session_state.get("_llm_tier_used", 0),
                            style=_llm_style,
                            model=_llm_model,
                        )
                    )
                st.session_state["_llm_chat"].append(
                    {"role": "assistant", "content": _follow_resp}
                )


st.header("Экспорт отчёта")

with st.expander("Сформировать PDF-отчёт для пациентки", expanded=False):
    col_pdf1, col_pdf2 = st.columns([2, 1])
    with col_pdf1:
        pdf_patient_name = st.text_input(
            "ФИО пациентки (для отчёта)",
            placeholder="Иванова Мария Петровна",
            help="Введите имя для отображения на титульном листе PDF"
        )
        pdf_patient_id = st.text_input(
            "Номер карты / ID",
            placeholder="ИВФ-2026-001",
        )
    with col_pdf2:
        st.markdown("<br>", unsafe_allow_html=True)
        gen_pdf_btn = st.button(
            "Сформировать PDF",
            use_container_width=True,
            type="primary",
            disabled=not _PDF_OK,
        )

    if not _PDF_OK:
        st.error(f"PDF генератор недоступен. Установите: pip install reportlab")

    if gen_pdf_btn and _PDF_OK:
        _ss = st.session_state
        def _kv_get(obj, attr):
            if obj is None: return None
            return getattr(obj, attr, None) if hasattr(obj, attr) else (obj.get(attr) if isinstance(obj, dict) else None)
        _known_clean = {k: v for k, v in {
            "OKK":        _kv_get(known, "okk"),
            "MII":        _kv_get(known, "mii"),
            "2PN":        _kv_get(known, "pn2"),
            "Blastocist": _kv_get(known, "blasts"),
            "Khor.kach":  _kv_get(known, "good"),
            "Euploidnykh":_kv_get(known, "euploid"),
        }.items() if v}

        # Collect cluster recommendation text
        _ca = res.get("cluster_analysis", {})
        _dom_info = _ca.get("clusters", {}).get(_ca.get("dominant_cluster", ""), {})
        _reco_text = ""
        if isinstance(_dom_info, dict):
            _reco_text = _dom_info.get("recommendation", "")

        # Collect warnings
        _warns = []
        import numpy as _np
        _p_cancel = _np.mean(res['sim_okk'] == 0)
        if _p_cancel > 0.05:
            _warns.append(f"Риск отмены цикла: {_p_cancel*100:.1f}%")

        with st.spinner("Формирование PDF (рендеринг графиков)..."):
            try:
                # 7.1: PDF использует тот же результат BEFE, что и экран.
                _befe_pdf = _ss.get("_pdf_befe") if _BEFE_OK else None
                _pdf_bytes = generate_patient_report(
                    patient_name   = pdf_patient_name or "Не указано",
                    patient_id     = pdf_patient_id   or "—",
                    age            = float(age),
                    amh            = float(amh),
                    afc            = int(afc),
                    bmi            = float(bmi),
                    attempt        = int(attempt),
                    sperm_source   = sperm_source,
                    known          = _known_clean,
                    res            = res,
                    eb             = _eb,
                    post           = res['posterior'],
                    fig_funnel     = _ss.get("_pdf_fig_funnel"),
                    fig_violin     = _ss.get("_pdf_fig_violin"),
                    fig_bar        = _ss.get("_pdf_fig_bar"),
                    fig_pca        = _ss.get("_pdf_fig_pca"),
                    fig_bayes      = _ss.get("_pdf_fig_bayes"),
                    fig_attempts   = _ss.get("_pdf_fig_attempts"),
                    fig_risks      = _ss.get("_pdf_fig_risks"),
                    csdi_result    = _ss.get("csdi_result"),
                    clinic_name    = st.session_state.get("ivf_clinic_name", ""),
                    cluster_recommendations = _reco_text,
                    warnings_list  = _warns,
                    p_kat_raw      = _ss.get("_pdf_p_kat_raw"),
                    p_nvsa         = _ss.get("_pdf_p_nvsa"),        # совместимость
                    ci_kat         = _ss.get("_pdf_ci_kat",  (None, None)),
                    ci_nvsa        = _ss.get("_pdf_ci_nvsa", (None, None)),  # совместимость
                    p_gnn_ens      = _ss.get("_pdf_p_gnn_ens"),
                    p_gnn_raw      = _ss.get("_pdf_p_gnn_raw"),
                    w_gnn          = _ss.get("_pdf_w_gnn", 0.35),
                    fig_gnn        = _ss.get("_pdf_fig_gnn"),
                    befe_result    = _befe_pdf,
                    clinical_probability  = _clinical_probability,
                    cycle_probability     = _cycle_probability,
                    no_transfer_confirmed = _clinical_summary["no_transfer_confirmed"],
                )

                _fname = f"IVF_Report_{(pdf_patient_id or 'patient').replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.success(f"PDF готов — {len(_pdf_bytes)//1024} КБ")

                # ── DT Analytics: пишем строку только вместе с PDF ──────
                _analytics_record_id = _save_analytics(
                    _dt71,
                    clinic_name  = st.session_state.get("ivf_clinic_name", ""),
                    patient_name = pdf_patient_name or "",
                    patient_id   = pdf_patient_id   or "",
                )
                if _analytics_record_id:
                    st.caption(f"Аналитика сохранена · ID записи: `{_analytics_record_id}`")
                # ────────────────────────────────────────────────────────

                st.download_button(
                    label     = "⬇️ Скачать PDF-отчёт",
                    data      = _pdf_bytes,
                    file_name = _fname,
                    mime      = "application/pdf",
                    use_container_width=True,
                )
            except Exception as _pdf_exc:
                st.error(f"Ошибка генерации PDF: {_pdf_exc}")
                import traceback
                st.code(traceback.format_exc())
