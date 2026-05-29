"""
IVF Digital Twin v6.2 — Streamlit Clinical Application
Запуск: streamlit run app.py

Офлайн-лицензирование (RSA + AES-256):
  Лицензия проверяется локально без обращения к серверу.
  Выдача ключей — через generate_license.py (только у разработчика).
"""

import sys, os, warnings, math, csv, uuid as _uuid
from datetime import datetime, date
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import beta as beta_dist, norm, ks_2samp

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
            paper_bgcolor="white",
            plot_bgcolor="rgba(248,250,252,1)",
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


    """#RRGGBB → 'rgba(r,g,b,a)' — Plotly не принимает 8-значный HEX."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a:.2f})"

# ── Цветовые палитры ──────────────────────────────────────────
C = {
    "blue":   "#1565C0",
    "teal":   "#00695C",
    "green":  "#2E7D32",
    "orange": "#E65100",
    "red":    "#B71C1C",
    "purple": "#4A148C",
    "amber":  "#F57F17",
    "grey":   "#546E7A",
}

# Цвета для стадий воронки / violin (7 стадий)
STAGE_COLORS = ["#1565C0", "#1976D2", "#0288D1", "#00838F",
                "#2E7D32", "#558B2F", "#795548"]

# Возрастные группы
AGE_COLORS = {
    "<30":   "#1565C0",
    "30–35": "#2E7D32",
    "35–38": "#F57F17",
    "38–41": "#E65100",
    ">41":   "#B71C1C",
}

# Кластеры
CLUSTER_HEX = {0: "#1976D2", 1: "#C62828", 2: "#2E7D32"}
CLUSTER_NAMES = {0: "C0 Standard", 1: "C1 Poor", 2: "C2 High"}

# ── Базовый layout (применять через **LAYOUT) ─────────────────
_FONT = dict(family="Inter, Arial, sans-serif", size=12, color="#1C2833")
LAYOUT = dict(
    font=_FONT,
    paper_bgcolor="white",
    plot_bgcolor="rgba(248,250,252,1)",
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

# ── Настройка страницы ────────────────────────────────────────
st.set_page_config(
    page_title="IVF Digital Twin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Podklyuchaem kriptograficheskiy dvizhok
# -- PDF generator
try:
    from src.pdf_report import generate_patient_report
    _PDF_OK = True
except ImportError as _pe:
    _PDF_OK = False
    _PDF_ERR = str(_pe)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE_DIR, "src"))
try:
    from crypt_engine import verify_license, get_model_key
    _CRYPT_ENGINE_OK = True
except ImportError as _ce:
    _CRYPT_ENGINE_OK = False
    _CRYPT_ENGINE_ERR = str(_ce)

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

        st.markdown("## 🧬 IVF Digital Twin v6.2")
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
            st.info("📄 Лицензионный файл найден. Проверка...")
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
            "IVF Digital Twin · Sergeev et al., 2025 · embryossa@gmail.com · Research prototype"
        )


def _try_activate(key_str: str):
    """Проверяет и активирует лицензию."""
    if not key_str:
        st.warning("⚠️ Введите лицензионный ключ")
        return

    if not _CRYPT_ENGINE_OK:
        st.error(f"❌ Модуль криптографии недоступен: {_CRYPT_ENGINE_ERR}")
        st.info("Выполните: pip install cryptography")
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
        st.success(f"✅ Лицензия активирована! Добро пожаловать, {clinic_name}")
        st.rerun()
    else:
        st.error(f"❌ {reason}")
        # Если файл есть но ключ невалиден — удаляем
        if os.path.exists(_LICENSE_FILE):
            try:
                os.remove(_LICENSE_FILE)
            except Exception:
                pass


def check_license():
    """Проверяет офлайн-лицензию перед показом интерфейса."""
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

# ── Баннер лицензии в sidebar ─────────────────────────────────
_clinic = st.session_state.get("ivf_clinic_name", "")
_expires = st.session_state.get("ivf_expires")
if _clinic and _expires:
    _days_left = (_expires - date.today()).days
    if _days_left <= 14:
        st.sidebar.warning(
            f"⚠️ Лицензия истекает через **{_days_left} дн.** ({_expires})\n"
            "Обратитесь к поставщику для продления."
        )
    else:
        st.sidebar.success(
            f"✅ Клиника: **{_clinic}**\n"
            f"Лицензия до: {_expires} ({_days_left} дн.)"
        )


# ── подключаем основной pipeline (.py или скомпилированный .pyd) ──
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _src_dir)

_src_py  = os.path.join(_src_dir, "ivf_digital_twin.py")
_src_pyd = any(
    f.startswith("ivf_digital_twin") and f.endswith(".pyd")
    for f in os.listdir(_src_dir)
) if os.path.isdir(_src_dir) else False

if _src_pyd:
    # Скомпилированный бинарный модуль — просто импортируем
    import ivf_digital_twin as _ivf_mod
    globals().update({k: getattr(_ivf_mod, k)
                      for k in dir(_ivf_mod) if not k.startswith("__")})
elif os.path.exists(_src_py):
    # Обычный .py — выполняем через exec (совместимость)
    _g = globals()
    _orig_file = _g.get("__file__", "")
    _g["__file__"] = _src_py
    _pipeline_code = open(_src_py, encoding="utf-8").read()
    _pipeline_code = _pipeline_code.replace("if __name__ ==", "if False and __name__ ==")
    exec(compile(_pipeline_code, _src_py, "exec"), _g)
    _g["__file__"] = _orig_file
else:
    st.error("Критическая ошибка: ivf_digital_twin не найден (ни .py ни .pyd)")
    st.stop()

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

# Проверяем наличие скомпилированного .pyd для CSDI
_csdi_pyd = any(
    f.startswith("embryo_csdi_v3") and f.endswith(".pyd")
    for f in os.listdir(_src_dir)
) if os.path.isdir(_src_dir) else False

_csdi_candidates = [
    os.path.join(_src_dir, "embryo_csdi_v3.py"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "embryo_csdi_v3.py"),
]

if _csdi_pyd:
    try:
        import embryo_csdi_v3 as _csdi_mod
        globals().update({k: getattr(_csdi_mod, k)
                          for k in dir(_csdi_mod) if not k.startswith("__")})
        CSDI_AVAILABLE    = True
        _CSDI_CLASS_READY = True
    except Exception as _e:
        CSDI_LOAD_ERROR = str(_e)
else:
    for _cf in _csdi_candidates:
        if os.path.exists(_cf):
            _g2 = globals()
            _orig_file2 = _g2.get("__file__", "")
            try:
                _csdi_code = open(_cf, encoding="utf-8").read()
                _csdi_code = _csdi_code.replace("if __name__ ==", "if False and __name__ ==")
                _g2["__file__"] = _cf
                exec(compile(_csdi_code, _cf, "exec"), _g2)
                CSDI_AVAILABLE    = True
                _CSDI_CLASS_READY = True
            except Exception as _e:
                CSDI_LOAD_ERROR = str(_e)
            finally:
                _g2["__file__"] = _orig_file2
            break

# Директория с обученной моделью
_CSDI_MODEL_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "embryo_v3_model"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "embryo_v3_model"),
    "embryo_v3_model",
]

# ── CSS ───────────────────────────────────────────────────────
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

_ANALYTICS_DIR = _Path(os.path.dirname(os.path.abspath(__file__))) / "dt_analytics_data"
_ANALYTICS_CSV = _ANALYTICS_DIR / "dt_predictions.csv"

_ANALYTICS_COLUMNS = [
    # Идентификация
    "record_id", "timestamp", "clinic_name",
    # Пациент (заполняется при генерации PDF — трейсинг с реальными данными)
    "patient_name", "patient_id",
    # Входные данные пациента
    "age", "amh", "afc", "bmi", "attempt_number", "sperm_source", "follicles_tvp",
    # Известные mid-cycle значения (байесовское обновление)
    "known_okk", "known_mii", "known_pn2", "known_blasts", "known_good", "known_euploid",
    # Медианы воронки (MC)
    "med_okk", "med_mii", "med_pn2", "med_blasts", "med_good", "med_euploid", "med_warmed",
    # Перцентили (P2.5 / P97.5) для ключевых стадий
    "p025_okk", "p975_okk", "p025_blasts", "p975_blasts", "p025_good", "p975_good",
    # Ключевые прогнозы беременности
    "p_per_transfer", "p_cum_if_viable", "p_overall_cycle", "p_viable",
    "p_cancel_risk", "rate_ci_low", "rate_ci_high",
    # Байесовский posterior
    "bayes_mean", "bayes_ci_low", "bayes_ci_high", "bayes_prior_mean", "bayes_prior_type",
    # Нейросеть L3: KAT / NVSA
    "p_kat_raw", "p_nvsa", "ci_kat_low", "ci_kat_high", "ci_nvsa_low", "ci_nvsa_high",
    # CSDI L5 (заполняется при открытии вкладки Diffusion)
    "p_csdi", "csdi_ci_low", "csdi_ci_high",
    # Кластер L4
    "dominant_cluster", "cluster_c0_prob", "cluster_c1_prob", "cluster_c2_prob",
    # Риски
    "ohss_moderate", "ohss_severe", "ohss_any",
    "p_no_blast", "p_no_good_blast",
    # Банкинг (Esteves)
    "banking_p_per_mii", "banking_expected_euploid",
    # Реальный исход (заполняется позже вручную)
    "real_outcome", "outcome_date", "notes",
]


def _save_analytics(res, _eb, age, amh, afc, bmi, attempt,
                    sperm_source, follicles, known, clinic_name,
                    patient_name="", patient_id="",
                    csdi_result=None):
    """
    Записывает одну строку с результатами расчёта Digital Twin в master CSV.
    Вызывается один раз при успешной генерации PDF-отчёта.
    Возвращает record_id (str) или None при ошибке.
    """
    try:
        _ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

        post  = res.get("posterior", {})
        ca    = res.get("cluster_analysis", {})
        ohss  = res.get("ohss", {})
        empty = res.get("empty", {})
        _nn   = res.get("nn_prediction", {})
        _nvsa = res.get("nn_nvsa", {})

        probs   = ca.get("cluster_probs", {})
        ci_kat  = _nn.get("base_prob_ci",  (None, None))
        ci_nvsa = _nvsa.get("adjusted_ci", (None, None))

        # CSDI — может быть None если вкладка Diffusion ещё не открывалась
        _p_csdi    = None
        _csdi_ci_l = None
        _csdi_ci_h = None
        if csdi_result and isinstance(csdi_result, dict):
            _p_csdi    = csdi_result.get("P_pregnancy")
            _ci95      = csdi_result.get("CI_95", (None, None))
            _csdi_ci_l = _ci95[0] if _ci95 else None
            _csdi_ci_h = _ci95[1] if _ci95 else None

        # Банкинг — явная проверка типа, не через bool(_eb)
        _p_mii  = None
        _exp_eu = None
        if isinstance(_eb, dict):
            _p_mii = _eb.get("p_per_mii")          # всегда float в _compute_esteves_banking
            _fwd   = _eb.get("forward_at_median")   # None если mii_med == 0
            if isinstance(_fwd, dict):
                _exp_eu = _fwd.get("mean")

        # Перцентили из симуляций
        def _pct(arr, q):
            try:
                return int(np.percentile(arr, q))
            except Exception:
                return ""

        def _r4(v):
            """Округление до 4 знаков; None → пустая строка."""
            if v is None:
                return ""
            try:
                return round(float(v), 4)
            except Exception:
                return ""

        row = {
            # Идентификация
            "record_id":        str(_uuid.uuid4()),
            "timestamp":        datetime.now().isoformat(timespec="seconds"),
            "clinic_name":      clinic_name or "",
            # Пациент
            "patient_name":     patient_name or "",
            "patient_id":       patient_id   or "",
            # Входные данные
            "age":              age,
            "amh":              amh,
            "afc":              afc,
            "bmi":              bmi,
            "attempt_number":   attempt,
            "sperm_source":     sperm_source,
            "follicles_tvp":    follicles if follicles else "",
            # Known mid-cycle
            "known_okk":        known.okk     if known and known.okk     is not None else "",
            "known_mii":        known.mii     if known and known.mii     is not None else "",
            "known_pn2":        known.pn2     if known and known.pn2     is not None else "",
            "known_blasts":     known.blasts  if known and known.blasts  is not None else "",
            "known_good":       known.good    if known and known.good    is not None else "",
            "known_euploid":    known.euploid if known and known.euploid is not None else "",
            # Медианы воронки
            "med_okk":          res.get("okk_med", ""),
            "med_mii":          res.get("mii_med", ""),
            "med_pn2":          res.get("pn2_med", ""),
            "med_blasts":       res.get("blasts_med", ""),
            "med_good":         res.get("good_med", ""),
            "med_euploid":      res.get("euploid_med", ""),
            "med_warmed":       res.get("warmed_med", ""),
            # Перцентили
            "p025_okk":         _pct(res.get("sim_okk", []), 2.5),
            "p975_okk":         _pct(res.get("sim_okk", []), 97.5),
            "p025_blasts":      _pct(res.get("sim_blasts", []), 2.5),
            "p975_blasts":      _pct(res.get("sim_blasts", []), 97.5),
            "p025_good":        _pct(res.get("sim_good", []), 2.5),
            "p975_good":        _pct(res.get("sim_good", []), 97.5),
            # Прогнозы беременности
            "p_per_transfer":   _r4(res.get("p_per_transfer", 0)),
            "p_cum_if_viable":  _r4(res.get("p_cum_if_viable", 0)),
            "p_overall_cycle":  _r4(res.get("p_overall_cycle", 0)),
            "p_viable":         _r4(res.get("p_viable", 0)),
            "p_cancel_risk":    _r4(np.mean(res["sim_okk"] == 0)),
            "rate_ci_low":      _r4(res.get("rate_ci", (0, 0))[0]),
            "rate_ci_high":     _r4(res.get("rate_ci", (0, 0))[1]),
            # Байес
            "bayes_mean":       _r4(post.get("mean", 0)),
            "bayes_ci_low":     _r4(post.get("ci_low", 0)),
            "bayes_ci_high":    _r4(post.get("ci_high", 0)),
            "bayes_prior_mean": _r4(post.get("prior_mean", 0)),
            "bayes_prior_type": post.get("prior_type", ""),
            # Нейросеть
            "p_kat_raw":        _r4((_nn.get("base_prob_mean")))   if _nn.get("base_prob_mean")  is not None else "",
            "p_nvsa":           _r4(_nvsa.get("adjusted_mean"))    if _nvsa.get("adjusted_mean") is not None else "",
            "ci_kat_low":       _r4(ci_kat[0])  if ci_kat[0]  is not None else "",
            "ci_kat_high":      _r4(ci_kat[1])  if ci_kat[1]  is not None else "",
            "ci_nvsa_low":      _r4(ci_nvsa[0]) if ci_nvsa[0] is not None else "",
            "ci_nvsa_high":     _r4(ci_nvsa[1]) if ci_nvsa[1] is not None else "",
            # CSDI
            "p_csdi":           _r4(_p_csdi)    if _p_csdi    is not None else "",
            "csdi_ci_low":      _r4(_csdi_ci_l) if _csdi_ci_l is not None else "",
            "csdi_ci_high":     _r4(_csdi_ci_h) if _csdi_ci_h is not None else "",
            # Кластер
            "dominant_cluster": ca.get("dominant_cluster", ""),
            "cluster_c0_prob":  _r4(probs.get(0)) if probs.get(0) is not None else "",
            "cluster_c1_prob":  _r4(probs.get(1)) if probs.get(1) is not None else "",
            "cluster_c2_prob":  _r4(probs.get(2)) if probs.get(2) is not None else "",
            # Риски
            "ohss_moderate":    _r4(ohss.get("p_moderate_ohss", 0)),
            "ohss_severe":      _r4(ohss.get("p_severe_ohss", 0)),
            "ohss_any":         _r4(ohss.get("p_any_ohss", 0)),
            "p_no_blast":       _r4(empty.get("p_no_blast", 0)),
            "p_no_good_blast":  _r4(empty.get("p_no_good_blast", 0)),
            # Банкинг
            "banking_p_per_mii":        _r4(_p_mii),
            "banking_expected_euploid": _r4(_exp_eu),
            # Исход — заполняется позже
            "real_outcome":  "",
            "outcome_date":  "",
            "notes":         "",
        }

        write_header = not _ANALYTICS_CSV.exists()
        with open(_ANALYTICS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_ANALYTICS_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        return row["record_id"]

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
    return _load_gnn_model(base_dir=_BASE_DIR)

_gnn_bundle = _get_gnn_bundle()
st.sidebar.image(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo22.png"), width=80)
st.sidebar.title("IVF Digital Twin v6.2")
st.sidebar.caption("Sergeev et al., 2025")
st.sidebar.markdown("---")

st.sidebar.header("👩 Параметры пациентки")
age  = st.sidebar.number_input("Возраст (лет)", 18, 50, 35, 1)
amh  = st.sidebar.number_input("АМГ (нг/мл)", 0.01, 15.0, 2.50, 0.10,
                                 format="%.2f")
afc  = st.sidebar.number_input("АФС (антральные фолликулы)", 1, 60, 15, 1)
bmi  = st.sidebar.number_input("ИМТ (кг/м²)", 15.0, 45.0, 23.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Параметры цикла")
attempt = st.sidebar.number_input("Номер попытки ЭКО", 1, 10, 1, 1)
follicles = st.sidebar.number_input("Фолликулов на ТВП (0 = AFC)",
                                     0, 60, 0, 1)
follicles = None if follicles == 0 else int(follicles)

sperm_label = st.sidebar.selectbox(
    "Источник спермы (для банкинга)",
    ["Эякулят", "Тестикулярная (НОА)",
     "Тестикулярная (ОА)", "Эпидидимальная"],
    index=0,
    help="Модуль банкинга учитывает источник спермы при оценке выхода эуплоидных",
)
_sperm_map = {
    "Эякулят":               "ejaculate",
    "Тестикулярная (НОА)":   "testicular_NOA",
    "Тестикулярная (ОА)":    "testicular_OA",
    "Эпидидимальная":        "epididymal",
}
sperm_source = _sperm_map[sperm_label]

st.sidebar.markdown("---")
st.sidebar.header("🔬 Байесовское обновление (mid-cycle)")
st.sidebar.caption("Введите наблюдённые значения. Оставьте 0 = не наблюдалось.")

def optional_int(val): return int(val) if val > 0 else None

okk_obs    = st.sidebar.number_input("Получено ооцитов (ОКК)", 0, 60, 0)
mii_obs    = st.sidebar.number_input("MII ооцитов", 0, 60, 0)
pn2_obs    = st.sidebar.number_input("2PN зигот", 0, 50, 0)
blasts_obs = st.sidebar.number_input("Бластоцист всего", 0, 40, 0)
good_obs   = st.sidebar.number_input("Бластоцист хор. кач.", 0, 40, 0)
euploid_obs= st.sidebar.number_input("Эуплоидных (ПГТ-А)", 0, 30, 0)

known = KnownValues(
    okk     = optional_int(okk_obs),
    mii     = optional_int(mii_obs),
    pn2     = optional_int(pn2_obs),
    blasts  = optional_int(blasts_obs),
    good    = optional_int(good_obs),
    euploid = optional_int(euploid_obs),
)

st.sidebar.markdown("---")
st.sidebar.header("🏥 Данные клиники (prior)")
use_clinic = st.sidebar.checkbox("Использовать данные клиники", value=True)
if use_clinic:
    clinic_raw = st.sidebar.text_area(
        "Успехи / Переносы (по строке: 19/43)",
        value="19/43\n18/45\n20/65\n6/18\n13/26\n12/31\n19/47\n22/49\n25/58",
        height=160,
    )
    try:
        lines = [l.strip() for l in clinic_raw.strip().splitlines() if "/" in l]
        clinic_s = [int(l.split("/")[0]) for l in lines]
        clinic_t = [int(l.split("/")[1]) for l in lines]
        obs_rate = sum(clinic_s) / sum(clinic_t)
        st.sidebar.success(f"✓ {len(clinic_s)} батчей, факт. частота: "
                           f"{obs_rate*100:.1f}%")
    except:
        clinic_s, clinic_t = None, None
        st.sidebar.error("Неверный формат")
else:
    clinic_s, clinic_t = None, None

n_sim = st.sidebar.select_slider(
    "Итераций MC", options=[500, 1000, 2000, 5000], value=2000)

# ── Загрузка нейросети (L3) ───────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🤖 Нейросетевой модуль")

@st.cache_resource(show_spinner="Загрузка нейросетевых моделей...")
def get_nn_model():
    return load_nn_ensemble()

nn_model = get_nn_model()

if nn_model is not None:
    st.sidebar.success("✅ KAT (KAN + FT-Transformer) загружен")
else:
    if not NN_LIBS_AVAILABLE:
        if "dll_error" in NN_LIBS_ERROR:
            st.sidebar.error(
                "❌ **Ошибка DLL (fbgemm.dll)**\n\n"
                "Запустите `fix_torch_dll.bat`\n\n"
                "Работает FORTUNE+KPI без нейросети."
            )
        else:
            st.sidebar.warning(
                "⚠️ torch не установлен\n\n"
                "Запустите `fix_torch_dll.bat`\n\n"
                "Работает FORTUNE+KPI без нейросети."
            )
    else:
        st.sidebar.info(
            "ℹ️ Файлы моделей не найдены.\n\n"
            "Поместите в `src/` или `models/`:\n"
            "- `Prediction_KAN.pth`\n"
            "- `FTTransformer.joblib`\n"
            "- `KAT_calibrated_model.pkl`\n\n"
            "Работает FORTUNE+KPI без нейросети."
        )

# ── Загрузка CSDI Hybrid v3 (L5) ─────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🔬 Diffusion модуль (L5)")
CSDI_MODEL_LOAD_ERROR = ""

@st.cache_resource(show_spinner="Загрузка CSDI Hybrid v3...")
def get_csdi_model():
    global CSDI_MODEL_LOAD_ERROR
    if not _CSDI_CLASS_READY:
        return None
    for _d in _CSDI_MODEL_DIRS:
        _cfg = os.path.join(_d, "config.json")
        # Accept either plain .pt or encrypted .pt.enc
        _wts_plain = os.path.join(_d, "csdi_weights.pt")
        _wts_enc   = os.path.join(_d, "csdi_weights.pt.enc")
        if os.path.isfile(_cfg) and (os.path.isfile(_wts_plain) or os.path.isfile(_wts_enc)):
            try:
                return EmbryoHybridV3.load(_d)
            except Exception as _e:
                CSDI_MODEL_LOAD_ERROR = str(_e)
                return None
    return None

csdi_model = get_csdi_model()

# ── CSDI runner — на уровне модуля, не внутри with-блока.
# st.cache_data не работает корректно когда функция переопределяется
# при каждом рендере и захватывает PyTorch-модель из closure:
# Streamlit не может её хешировать и возвращает первый результат навсегда.
# session_state решает проблему: пересчёт только при реальном изменении
# входных данных пациента (сравнение кортежа из 7 чисел).
def _csdi_run_and_cache(patient: dict, key: tuple):
    """Запускает mc_sample только если ключ изменился.
    Результат хранится в st.session_state — живёт в рамках сессии.
    """
    if (st.session_state.get("_csdi_last_key") != key
            or "csdi_result" not in st.session_state):
        with st.spinner("Генерация CSDI-траекторий (DDIM, 50 шагов)..."):
            result = csdi_model.mc_sample(patient, n_samples=1000)
        st.session_state["csdi_result"]    = result
        st.session_state["_csdi_last_key"] = key
    return st.session_state["csdi_result"]


if csdi_model is not None:
    st.sidebar.success(f"✅ CSDI Hybrid v3 загружен "
                       f"(порог: {csdi_model.best_threshold:.2f})")
elif not _CSDI_CLASS_READY:
    st.sidebar.info(
        "ℹ️ CSDI Hybrid v3 не загружен.\n\n"
        f"Причина: `{CSDI_LOAD_ERROR or 'src/embryo_csdi_v3.py не найден'}`\n\n"
        "Проверьте файл `src/embryo_csdi_v3.py` и зависимости L5."
    )
elif CSDI_MODEL_LOAD_ERROR:
    st.sidebar.info(
        "ℹ️ Модель CSDI найдена, но не загрузилась.\n\n"
        f"Причина: `{CSDI_MODEL_LOAD_ERROR}`\n\n"
        "Проверьте файлы в `models/embryo_v3_model/`."
    )
else:
    st.sidebar.info(
        "ℹ️ Модель не найдена.\n\n"
        "Поместите папку `embryo_v3_model/` в `models/`.\n\n"
        "Обучение: `python src/embryo_csdi_v3.py`"
    )

# ── Статус GNN модели (Graph Transformer) ─────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🕸️ Graph модуль (GAT)")

if _gnn_bundle.get('available'):
    st.sidebar.success("✅ GNN (Graph Transformer) загружен")
elif not _GNN_IMPORT_OK:
    st.sidebar.warning(
        "⚠️ torch-geometric не установлен\n\n"
        "Запустите `INSTALL.bat` (шаг 8) или:\n"
        "```\npip install torch-scatter torch-sparse \\\n"
        "  torch-cluster torch-spline-conv \\\n"
        "  -f https://data.pyg.org/whl/torch-2.5.1+cpu.html\n"
        "pip install torch-geometric\n```\n\n"
        "GAT Ансамбль будет недоступен."
    )
else:
    _gnn_err_short = _gnn_bundle.get('error', 'Файл не найден')[:80]
    st.sidebar.info(
        "ℹ️ GNN модель не загружена.\n\n"
        f"Причина: `{_gnn_err_short}`\n\n"
        "Поместите `gnn_ivf_model.pt` в `models/`.\n\n"
        "Обучение: `python gnn_ivf_562.py clinical_protocols.xlsx`"
    )

run_btn = st.sidebar.button("▶ Запустить расчёт", use_container_width=True,
                             type="primary")

# ── ГЛАВНАЯ СТРАНИЦА ──────────────────────────────────────────
st.title("IVF Digital Twin")
st.markdown("""
<div class="disclaimer">
⚠️ <b>Только для поддержки клинического решения.</b> Все прогнозы являются
вероятностными оценками на основе опубликованных моделей. Окончательное
решение принимает врач-репродуктолог.
<br><i>IVF Digital Twin v6.2 · Sergeev et al., 2025</i>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    # If we have cached results, skip the welcome screen and proceed to show results + PDF
    if st.session_state.get("_pdf_res") is None:
        col1, col2, col3 = st.columns(3)
        col1.info("← Введите данные пациентки в панели слева")
        col2.info("Нажмите **▶ Запустить расчёт**")
        col3.info("Получите полный отчёт с графиками")

        with st.expander("ℹ️ О системе"):
            st.markdown("""
        **IVF Digital Twin v6.2** — интегрированная система прогнозирования
        исходов ЭКО, объединяющая 6 независимых слоёв оценки.

        *Sergeev et al., 2025. Personal research project.*

        | Слой | Метод |
        |---|---|
        | L1 Стохастический pipeline | ZINB + биномиальные фильтры (S1–S6b) |
        | L2 Ансамбль на перенос | FORTUNE + KPIScore (логит-взвешивание) |
        | L3 Нейросеть | KAN + FT-Transformer + Venn-Abers (KAT) |
        | L4 Кластер | Ближайший центроид 18D (Sergeev et al.) |
        | L5 Лабораторный прогноз | CSDI-Transformer + LightGBM + Conformal PI |
        | L6 Граф пациентов | Graph Attention Transformer (GAT) + ансамбль с KAT |

        **Байесовский posterior** с пациент-зависимым prior (Beta-регрессия).
        **Трёхуровневая декомпозиция** вероятности беременности.
        """)
        st.stop()
    else:
        # Restore cached results so the rest of the page renders normally
        res         = st.session_state["_pdf_res"]
        _eb         = st.session_state.get("_pdf_eb")
        known       = st.session_state.get("_pdf_known", {})
        sperm_source = st.session_state.get("_pdf_sperm", "")

# ── РАСЧЁТ ───────────────────────────────────────────────────
if run_btn:
    patient = PatientInput(female_age=float(age), amh=float(amh),
                           afc=int(afc), bmi=float(bmi))

    with st.spinner(f"Выполняется {n_sim} итераций Monte Carlo..."):
        np.random.seed(42)
        res = run_pipeline_extended(
            patient, known=known,
            attempt_number=int(attempt),
            follicles=follicles,
            nn_model=nn_model,
            clinic_real_successes=clinic_s,
            clinic_real_trials=clinic_t,
            max_attempts_curve=6,
            n=n_sim,
        )
        st.session_state["_pdf_res"]     = res
        st.session_state["_pdf_known"]   = known
        st.session_state["_pdf_age"]     = float(age)
        st.session_state["_pdf_amh"]     = float(amh)
        st.session_state["_pdf_afc"]     = int(afc)
        st.session_state["_pdf_bmi"]     = float(bmi)
        st.session_state["_pdf_attempt"] = int(attempt)
        st.session_state["_pdf_sperm"]   = sperm_source
    _nn_s = res.get('nn_prediction', {})
    _nv_s = res.get('nn_nvsa', {})
    st.session_state["_pdf_p_kat_raw"] = _nn_s.get('base_prob_mean')
    st.session_state["_pdf_p_nvsa"]    = _nv_s.get('adjusted_mean')
    st.session_state["_pdf_ci_kat"]    = _nn_s.get('base_prob_ci',  (None, None))
    st.session_state["_pdf_ci_nvsa"]   = _nv_s.get('adjusted_ci',   (None, None))
else:
    # Use cached values (user interacting with PDF form after calculation)
    res          = st.session_state["_pdf_res"]
    known        = st.session_state.get("_pdf_known", {})
    sperm_source = st.session_state.get("_pdf_sperm", "")

# ── BANKING MODULE (Esteves model) ───────────────────────────
def _compute_esteves_banking(patient_age, sperm_src, res_pipeline):
    """
    Esteves et al. model: probability of euploid blastocyst per MII oocyte.
    Combines: fertilisation × D5-blast × euploidy (age + sperm dependent).

    Sources:
      Fertilisation by sperm source: Esteves 2022, Palermo 2022
      D5 blastulation: Romanski 2022
      Euploidy by age: Franasiak 2014 + Armstrong 2023
    """
    from scipy.stats import binom as _binom

    # ── Fertilisation rate by sperm source ──────────────────
    fert_by_source = {
        "ejaculate":       0.76,
        "testicular_NOA":  0.56,
        "testicular_OA":   0.68,
        "epididymal":      0.66,
    }
    fert_r = fert_by_source.get(sperm_src, 0.76)

    # ── D5 blastulation rate (age-adjusted) ─────────────────
    if patient_age < 35:
        blast_r = 0.48
    elif patient_age < 38:
        blast_r = 0.44
    elif patient_age < 41:
        blast_r = 0.38
    else:
        blast_r = 0.30

    # ── Euploidy rate by age (Franasiak 2014) ───────────────
    eupl_table = {
        (0,  35): 0.68,
        (35, 37): 0.57,
        (37, 39): 0.48,
        (39, 41): 0.38,
        (41, 43): 0.29,
        (43, 99): 0.18,
    }
    eupl_r = 0.30
    for (lo, hi), v in eupl_table.items():
        if lo <= patient_age < hi:
            eupl_r = v
            break

    # p per MII = fertilisation × blastulation × euploidy
    p_per_mii = fert_r * blast_r * eupl_r

    # ── Forward table ────────────────────────────────────────
    mii_median = int(res_pipeline.get('mii_med', 0))
    forward = None
    if mii_median > 0:
        dist = [_binom.pmf(k, mii_median, p_per_mii) for k in range(mii_median+1)]
        mean_e  = mii_median * p_per_mii
        med_e   = float(_binom.ppf(0.50, mii_median, p_per_mii))
        forward = {"mean": mean_e, "median": med_e, "pmf": dist}

    # ── How many euploid needed for pregnancy target ─────────
    # Uses cumulative transfer model: P(preg|k_euploid) = 1-(1-p_xfer)^k
    p_xfer = float(res_pipeline.get('p_per_transfer', 0.35))
    euploid_for_preg = {}
    for target_p in [0.50, 0.70, 0.90]:
        if p_xfer <= 0:
            euploid_for_preg[target_p] = None
            continue
        import math as _math
        k = _math.ceil(_math.log(1 - target_p) / _math.log(1 - min(p_xfer, 0.9999)))
        euploid_for_preg[target_p] = max(1, k)

    # ── Inverse table: MII needed for k euploid at confidence ──
    k_targets   = [1, 2, 3, 4, 5]
    confidences = [0.70, 0.80, 0.90]
    mii_table   = {}
    for k in k_targets:
        mii_table[k] = {}
        for cf in confidences:
            found = None
            for n in range(k, 501):
                if 1 - _binom.cdf(k - 1, n, p_per_mii) >= cf:
                    found = n
                    break
            mii_table[k][cf] = found

    return {
        "p_per_mii":         p_per_mii,
        "age":               patient_age,
        "sperm_source":      sperm_src,
        "fert_r":            fert_r,
        "blast_r":           blast_r,
        "eupl_r":            eupl_r,
        "patient_mii_median":mii_median if mii_median > 0 else None,
        "forward_at_median": forward,
        "euploid_for_preg":  euploid_for_preg,
        "k_targets":         k_targets,
        "confidences":       confidences,
        "mii_table":         mii_table,
    }

if run_btn:
    _eb = _compute_esteves_banking(float(age), sperm_source, res)
    st.session_state["_pdf_eb"] = _eb
else:
    _eb = st.session_state.get("_pdf_eb")

# ── БЛОК РЕЗУЛЬТАТОВ ─────────────────────────────────────────
st.markdown("---")
st.header("Результаты")

ca   = res['cluster_analysis']
post = res['posterior']
dom  = ca['dominant_cluster']

# ── Ключевые метрики ──────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("На перенос", f"{res['p_per_transfer']*100:.1f}%",
          help="Вероятность беременности при одном переносе")
c2.metric("Если цикл viable", f"{res['p_cum_if_viable']*100:.1f}%",
          help="Кумулятивная при ≥1 эмбрионе для переноса")
c3.metric("Успех цикла", f"{res['p_overall_cycle']*100:.1f}%",
          help="От начала стимуляции, включая риск пустого цикла")
# KAT raw (чистый выход нейросети)
_nn_pred = res.get('nn_prediction', {})
_nn_nvsa = res.get('nn_nvsa', {})
_p_kat_raw  = _nn_pred.get('base_prob_mean', None)
_p_nvsa     = _nn_nvsa.get('adjusted_mean',  None)
_ci_kat     = _nn_pred.get('base_prob_ci',   (None, None))
_ci_nvsa    = _nn_nvsa.get('adjusted_ci',    (None, None))

# ── GNN / GAT Ансамбль ────────────────────────────────────────
_gnn_result = {'available': False, 'gnn_prob': None, 'ensemble_prob': None, 'w_gnn': 0.35}
if _gnn_bundle.get('available') and run_btn:
    try:
        _gnn_feats = _build_gnn_features(
            age       = float(age),
            afc       = int(afc),
            attempt   = int(attempt),
            res       = res,
            known     = known,
            p_kat_raw = _p_kat_raw,
        )
        _gnn_result = _predict_gnn(_gnn_bundle, _gnn_feats, prai_score=_p_kat_raw)
        st.session_state['_gnn_result'] = _gnn_result
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
    except Exception as _gnn_exc:
        _gnn_result = {'available': False, 'error': str(_gnn_exc),
                       'gnn_prob': None, 'ensemble_prob': None, 'w_gnn': 0.35}
elif not run_btn:
    _gnn_result = st.session_state.get('_gnn_result',
                  {'available': False, 'gnn_prob': None,
                   'ensemble_prob': None, 'w_gnn': 0.35})

_p_gnn_ens  = _gnn_result.get('ensemble_prob')   # итоговый скор для отображения
_p_gnn_raw  = _gnn_result.get('gnn_prob')
_w_gnn      = _gnn_result.get('w_gnn', 0.35)

# Сохраняем в session_state для PDF
st.session_state['_pdf_p_gnn_ens'] = _p_gnn_ens
st.session_state['_pdf_p_gnn_raw'] = _p_gnn_raw
st.session_state['_pdf_w_gnn']     = _w_gnn

if _p_kat_raw is not None:
    c4.metric("KAT (ансамбль NN)",
              f"{_p_kat_raw*100:.1f}%",
              help=(f"Чистый выход нейросетевого ансамбля KAN+FT-Transformer. "
                    f"95% CI: {_ci_kat[0]*100:.1f}–{_ci_kat[1]*100:.1f}%"
                    if _ci_kat[0] is not None else "Чистый выход нейросетевого ансамбля"))
else:
    c4.metric("KAT (ансамбль NN)", "—", help="Нейросеть не загружена")

# c5 — GAT Ensemble (вместо NVSA)
if _p_gnn_ens is not None:
    c5.metric(
        "GAT Ансамбль",
        f"{_p_gnn_ens*100:.1f}%",
        help=(f"Graph Attention Transformer + KAT ансамбль. "
              f"GNN: {_p_gnn_raw*100:.1f}%  |  "
              f"w_GNN={_w_gnn:.2f} · w_KAT={1-_w_gnn:.2f}  |  "
              f"AUC(CV)≈0.66 на обучающей когорте")
    )
elif _gnn_bundle.get('available'):
    c5.metric("GAT Ансамбль", "—", help="Нажмите Рассчитать для получения предсказания GNN")
else:
    _gnn_err = _gnn_bundle.get('error', 'Модель не найдена')
    c5.metric("GAT Ансамбль", "н/д",
              help=f"GNN модель недоступна: {_gnn_err[:80]}")

p_cancel = np.mean(res['sim_okk'] == 0)
if p_cancel > 0.05:
    st.warning(f"⚠️ Риск отмены цикла (ZINB нулевые значения): "
               f"**{p_cancel*100:.1f}%**")

# ── Вкладки ───────────────────────────────────────────────────
tabs = st.tabs(["🔬 Pipeline", "📈 Беременность", "🧠 Кластер",
                "📉 Байес + попытки", "⚠️ Риски", "🏦 Банкинг",
                "🧬 Diffusion", "🕸️ GAT Graph"])

# ── TAB 1: Pipeline ───────────────────────────────────────────
with tabs[0]:
    col_f, col_v = st.columns([1, 2])

    with col_f:
        st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">Воронка (медианы)</p>', unsafe_allow_html=True)
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
        st.plotly_chart(funnel, use_container_width=True)
        st.session_state["_pdf_fig_funnel"] = funnel

    with col_v:
        st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">Распределения по стадиям</p>', unsafe_allow_html=True)
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
        st.plotly_chart(vfig, use_container_width=True)
        st.session_state["_pdf_fig_violin"] = vfig

    st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">95% интервалы</p>', unsafe_allow_html=True)
    pct = lambda arr, q: int(np.percentile(arr, q))
    table_data = {
        "Стадия": stages,
        "P2.5": [pct(a, 2.5) for a in arrays],
        "Медиана": meds,
        "P97.5": [pct(a, 97.5) for a in arrays],
    }
    st.dataframe(table_data, use_container_width=True, hide_index=True)

# ── TAB 2: Беременность ───────────────────────────────────────
with tabs[1]:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">Шансы ≥k беременностей в цикле</p>', unsafe_allow_html=True)
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
        st.plotly_chart(bar_fig, use_container_width=True)
        st.session_state["_pdf_fig_bar"] = bar_fig

    with col_b:
        st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">FORTUNE vs KPI vs Ансамбль</p>', unsafe_allow_html=True)
        comp_fig = go.Figure()
        _comp_palette = [
            ("FORTUNE",  res["sim_p_fortune"],  C["blue"],   0.55),
            ("KPI",      res["sim_p_kpi"],      C["orange"], 0.55),
            ("Ансамбль", res["sim_p_combined"], C["green"],  0.75),
        ]
        for label, arr, col, alpha in _comp_palette:
            comp_fig.add_trace(go.Histogram(
                x=arr * 100, opacity=0.65, name=label,
                marker=dict(
                    color=hex_rgba(col, alpha),
                    line=dict(color=hex_rgba(col, 0.90), width=0.8),
                ),
                xbins=dict(size=2),
            ))
        comp_fig.update_layout(
            **LAYOUT,
            barmode="overlay",
            height=400,
            margin=dict(l=65, r=30, t=60, b=60),
            xaxis=dict(title="Вероятность на перенос (%)"),
            yaxis=dict(title="Частота", gridcolor="rgba(200,210,220,0.35)"),
        )
        comp_fig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        comp_fig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        st.plotly_chart(comp_fig, use_container_width=True)

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

# ── TAB 3: Кластер ────────────────────────────────────────────
with tabs[2]:
    col_pca, col_info = st.columns([3, 2])

    with col_pca:
        st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">PCA(2) — кластерная принадлежность</p>', unsafe_allow_html=True)
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
        st.plotly_chart(pca_fig, use_container_width=True)
        st.session_state["_pdf_fig_pca"] = pca_fig

    with col_info:
        st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">Распределение по кластерам</p>', unsafe_allow_html=True)
        probs = ca['cluster_probs']
        for c in (0,1,2):
            info = CLUSTER_INTERPRETATIONS[c]
            css  = ["cluster-c0","cluster-c1","cluster-c2"][c]
            mark = " ← доминирует" if c == dom else ""
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

# ── TAB 4: Байес + попытки ────────────────────────────────────
with tabs[3]:
    col_bay, col_att = st.columns(2)

    with col_bay:
        st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">Байесовский posterior</p>', unsafe_allow_html=True)
        x = np.linspace(0.001, 0.999, 400)
        pp = beta_dist.pdf(x, post['posterior_alpha'], post['posterior_beta'])
        pr = beta_dist.pdf(x, post['prior_alpha'],     post['prior_beta'])
        bfig = go.Figure()
        bfig.add_trace(go.Scatter(
            x=x * 100, y=pr, mode="lines",
            name=f"Prior (mean {post['prior_mean']*100:.1f}%)",
            line=dict(color=hex_rgba(C["grey"], 0.65), dash="dot", width=2),
        ))
        bfig.add_trace(go.Scatter(
            x=x * 100, y=pp, mode="lines", fill="tozeroy",
            name=f"Posterior (mean {post['mean']*100:.1f}%)",
            line=dict(color=hex_rgba(C["green"], 1.0), width=2.5),
            fillcolor=hex_rgba(C["green"], 0.12),
        ))
        bfig.add_vline(
            x=post["mean"] * 100,
            line_dash="dash",
            line_color=hex_rgba(C["red"], 0.80),
            line_width=1.8,
            annotation_text=f"Mean {post['mean']*100:.1f}%",
            annotation_font=dict(family="Inter, Arial, sans-serif", size=11,
                                 color=C["red"]),
            annotation_bgcolor="rgba(255,255,255,0.85)",
        )
        bfig.add_vrect(
            x0=post["ci_low"] * 100, x1=post["ci_high"] * 100,
            fillcolor=hex_rgba(C["green"], 0.07), layer="below",
            line_width=0,
        )
        bfig.update_layout(
            **LAYOUT,
            height=400,
            margin=dict(l=65, r=30, t=60, b=60),
            xaxis=dict(title="Вероятность (%)"),
            yaxis=dict(title="Плотность", gridcolor="rgba(200,210,220,0.35)"),
        )
        bfig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        bfig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        st.plotly_chart(bfig, use_container_width=True)
        st.session_state["_pdf_fig_bayes"] = bfig
        st.markdown(f"""
        **Prior:** {post['prior_type']}, mean {post['prior_mean']*100:.1f}%,
        κ={post['prior_kappa']:.0f} |
        **95% CI:** {post['ci_low']*100:.1f}–{post['ci_high']*100:.1f}%
        """)

    with col_att:
        st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">Вероятность по попыткам ЭКО</p>', unsafe_allow_html=True)
        curve = res['attempt_curve']
        afig = go.Figure()
        # CI-полоса (снизу)
        afig.add_traces([go.Scatter(
            x=curve["attempts"] + curve["attempts"][::-1],
            y=[p * 100 for p in curve["p_hi"]] +
              [p * 100 for p in curve["p_lo"][::-1]],
            fill="toself",
            fillcolor=hex_rgba(C["blue"], 0.10),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
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
            marker=dict(
                size=11,
                color=hex_rgba(C["blue"], 0.90),
                line=dict(width=1.5, color="white"),
            ),
            text=[f"{p*100:.1f}%" for p in curve["p_mean"]],
            textposition="top center",
            textfont=dict(family="Inter, Arial, sans-serif", size=11),
        ))
        afig.update_layout(
            **LAYOUT,
            height=400,
            margin=dict(l=65, r=30, t=65, b=60),
            xaxis=dict(
                title="Номер попытки",
                tickmode="array", tickvals=curve["attempts"],
            ),
            yaxis=dict(title="Вероятность беременности (%)",
                       gridcolor="rgba(200,210,220,0.35)"),
        )
        afig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        afig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        st.plotly_chart(afig, use_container_width=True)
        st.session_state["_pdf_fig_attempts"] = afig
        st.caption(f"Снижение per-attempt: α={curve['decay_alpha']:.2f}  "
                   f"(Malizia et al. NEJM 2009)")

# ── TAB 5: Риски ─────────────────────────────────────────────
with tabs[4]:
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">Профиль рисков</p>', unsafe_allow_html=True)
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
                color=[hex_rgba(c, 0.78) for c in _risk_cols],
                line=dict(color=[hex_rgba(c, 0.95) for c in _risk_cols], width=1.5),
            ),
            text=[f"{v:.1f}%" for v in rv],
            textposition="outside",
            textfont=dict(family="Inter, Arial, sans-serif", size=12),
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
        st.plotly_chart(rfig, use_container_width=True)
        st.session_state["_pdf_fig_risks"] = rfig

    with col_r2:
        st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">Распределение Ооцитов (ZINB)</p>', unsafe_allow_html=True)
        okk_arr = res['sim_okk']
        p_zero  = np.mean(okk_arr == 0)
        ofig = px.histogram(okk_arr[okk_arr > 0], nbins=30, opacity=0.75,
                            labels={"value": "Число ооцитов"})
        ofig.update_traces(
            marker=dict(
                color=hex_rgba(C["blue"], 0.72),
                line=dict(color=hex_rgba(C["blue"], 0.90), width=0.8),
            )
        )
        ofig.update_layout(
            **LAYOUT,
            height=400,
            margin=dict(l=65, r=30, t=55, b=60),
            xaxis=dict(title="Число ооцитов"),
            yaxis=dict(title="Частота", gridcolor="rgba(200,210,220,0.35)"),
        )
        ofig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        ofig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
        st.plotly_chart(ofig, use_container_width=True)

        st.markdown(f"""
        | Показатель | Значение |
        |---|---|
        | Риск отмены цикла | **{p_zero*100:.1f}%** |
        | Медиана ООЦ | **{int(np.median(okk_arr))}** |
        | 5-й – 95-й перцентиль | {int(np.percentile(okk_arr,5))} – {int(np.percentile(okk_arr,95))} |
        """)

        if p_zero > 0.05:
            st.error(f"⛔ Высокий риск отмены цикла: {p_zero*100:.1f}%")
        elif p_zero > 0.02:
            st.warning(f"⚠️ Умеренный риск отмены: {p_zero*100:.1f}%")
        else:
            st.success(f"✅ Риск отмены низкий: {p_zero*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# ── TAB 6: Банкинг ────────────────────────────────────────────
with tabs[5]:
    eb = _eb
    if not eb:
        st.info("Модуль банкинга недоступен для этого расчёта.")
    else:
        st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">Эуплоидность и банкинг ооцитов</p>', unsafe_allow_html=True)
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
            st.plotly_chart(f1, use_container_width=True)

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
            st.plotly_chart(f2, use_container_width=True)

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
                    f"✅ При MII медиане {eb['patient_mii_median']} ожидается "
                    f"~{eb['forward_at_median']['mean']:.1f} эуплоидных — "
                    f"достаточно для цели 50% ({need50})")
            else:
                st.warning(
                    f"⚠️ При MII медиане {eb['patient_mii_median']} ожидается "
                    f"~{eb['forward_at_median']['mean']:.1f} эуплоидных — "
                    f"для 50% нужно {need50}. Рекомендуется банкинг "
                    f"дополнительных циклов.")

        with st.expander("ℹ️ Параметры модели банкинга"):
            st.markdown(f"""
            | Параметр | Значение |
            |---|---|
            | Источник спермы | {sperm_label} |
            | Коэфф. оплодотворения | {eb['fert_r']*100:.0f}% |
            | Коэфф. бластуляции D5 | {eb['blast_r']*100:.0f}% |
            | Частота эуплоидности (возраст {eb['age']:.0f}) | {eb['eupl_r']*100:.0f}% |
            | P(эуплоид/MII) итог | {eb['p_per_mii']*100:.1f}% |

            *Esteves et al. 2022; Franasiak et al. 2014; Romanski et al. 2022*
            """)
        st.caption("p — вероятность на один MII ооцит; объединяет оплодотворение "
                   "× бластуляцию × эуплоидность. Независимая модель планирования, "
                   "не заменяет основной pipeline.")

# ── TAB 7: Лабораторный прогноз (CSDI Hybrid v3 — L5) ─────────
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">L5 · Лабораторный прогноз (CSDI Hybrid v3)</p>', unsafe_allow_html=True)
    st.markdown("""
    Гибридная генеративная модель обучена на лабораторном этапе ЭКО
    и генерирует синтетические пары **(Число Bl, Число Bl хор.кач.)** без
    параметрических допущений MC-пайплайна. Частоты вычисляются аналитически
    из сгенерированных пар — это устраняет независимый дрейф числителя и
    знаменателя. Исход беременности предсказывает откалиброванный LightGBM + Platt.
    """)

    if csdi_model is None:
        # ── Модель не загружена ───────────────────────────────
        st.markdown("""
        <div class="diff-warn">
        <b>⚠️ CSDI Hybrid v3 не загружен</b><br><br>
        Для активации этой вкладки необходимо:<br>
        1. Убедиться, что <code>src/embryo_csdi_v3.py</code> присутствует<br>
        2. Обучить модель: <code>python src/embryo_csdi_v3.py</code><br>
        3. Скопировать папку <code>embryo_v3_model/</code> в <code>models/</code><br><br>
        MC-результаты (вкладки 1–6) работают независимо.
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Формируем patient dict для CSDI ──────────────────
        _foll_count = follicles if follicles is not None else int(afc)
        _okk_med    = max(1, int(res['okk_med']))
        _mii_med    = max(1, int(res['mii_med']))
        _pn2_med    = max(1, int(res['pn2_med']))
        _okk_rate   = min(1.0, _okk_med / max(_foll_count, 1))
        _fert_rate  = min(1.0, _pn2_med / max(_mii_med, 1))
        _kpi        = float(res['kpi_score_median'])

        _patient_csdi = {
            "Количество фолликулов":  float(_foll_count),
            "Число ОКК":              float(_okk_med),
            "Число инсеминированных": float(_mii_med),
            "2 pN":                   float(_pn2_med),
            "Частота получения ОКК":  _okk_rate,
            "Частота оплодотворения": _fert_rate,
            "KPIScore":               _kpi,
        }

        # Полный ключ по всем 7 кондиционирующим признакам CSDI.
        # Это гарантирует пересчёт при изменении любого входного параметра.
        _csdi_key = (
            _foll_count, _okk_med, _mii_med, _pn2_med,
            round(_okk_rate, 4), round(_fert_rate, 4), round(_kpi, 2)
        )

        _csdi_res = _csdi_run_and_cache(_patient_csdi, _csdi_key)

        _csdi_df   = _csdi_res['samples']
        _p_csdi    = _csdi_res['P_pregnancy']
        _p_mc      = res['p_per_transfer']
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
        _pred_label = "✅ Благоприятный" if _pred_ok else "⚠️ Осторожный"
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
                x=_mc_bl, name="MC pipeline", opacity=0.68,
                marker=dict(
                    color=hex_rgba(C["blue"], 0.68),
                    line=dict(color=hex_rgba(C["blue"], 0.90), width=0.8),
                ),
                xbins=dict(size=1),
            ))
            blfig.add_trace(go.Histogram(
                x=_csdi_bl, name="CSDI", opacity=0.68,
                marker=dict(
                    color=hex_rgba(C["green"], 0.68),
                    line=dict(color=hex_rgba(C["green"], 0.90), width=0.8),
                ),
                xbins=dict(size=1),
            ))
            blfig.update_layout(
                **LAYOUT,
                barmode="overlay",
                height=340,
                margin=dict(l=60, r=30, t=50, b=55),
                xaxis=dict(title="Число бластоцист"),
                yaxis=dict(title="Частота"),
            )
            blfig.add_annotation(
                text=f"KS={_ks_bl:.3f}  p={_p_bl:.3f}  {'✓ схожи' if _p_bl > 0.05 else '≠ различны'}",
                xref="paper", yref="paper", x=0.99, y=0.99,
                showarrow=False,
                font=dict(size=10, color=C["green"] if _p_bl > 0.05 else C["red"]),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ccc", borderwidth=1, borderpad=4,
            )
            blfig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            blfig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            st.plotly_chart(blfig, use_container_width=True)

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
                x=_mc_tgbdr * 100, name="MC pipeline", opacity=0.68,
                marker=dict(
                    color=hex_rgba(C["blue"], 0.68),
                    line=dict(color=hex_rgba(C["blue"], 0.90), width=0.8),
                ),
                xbins=dict(size=2),
            ))
            tfig.add_trace(go.Histogram(
                x=_csdi_tgbdr * 100, name="CSDI", opacity=0.68,
                marker=dict(
                    color=hex_rgba(C["green"], 0.68),
                    line=dict(color=hex_rgba(C["green"], 0.90), width=0.8),
                ),
                xbins=dict(size=2),
            ))
            tfig.update_layout(
                **LAYOUT,
                barmode="overlay",
                height=340,
                margin=dict(l=60, r=30, t=50, b=55),
                xaxis=dict(title="TGBDR (%)"),
                yaxis=dict(title="Частота"),
            )
            tfig.add_annotation(
                text=f"KS={_ks_tgbdr:.3f}  p={_p_tgbdr:.3f}  {'✓ схожи' if _p_tgbdr > 0.05 else '≠ различны'}",
                xref="paper", yref="paper", x=0.99, y=0.99,
                showarrow=False,
                font=dict(size=10, color=C["green"] if _p_tgbdr > 0.05 else C["red"]),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ccc", borderwidth=1, borderpad=4,
            )
            tfig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            tfig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            st.plotly_chart(tfig, use_container_width=True)

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
                x=_mc_gb, name="MC", opacity=0.68,
                marker=dict(
                    color=hex_rgba(C["blue"], 0.68),
                    line=dict(color=hex_rgba(C["blue"], 0.90), width=0.8),
                ),
                xbins=dict(size=1),
            ))
            gbfig.add_trace(go.Histogram(
                x=_csdi_gb, name="CSDI", opacity=0.68,
                marker=dict(
                    color=hex_rgba(C["green"], 0.68),
                    line=dict(color=hex_rgba(C["green"], 0.90), width=0.8),
                ),
                xbins=dict(size=1),
            ))
            gbfig.update_layout(
                **LAYOUT,
                barmode="overlay",
                height=280,
                xaxis=dict(title="Число бластоцист хор. кач."),
                yaxis=dict(title="Частота"),
            )
            gbfig.add_annotation(
                text=f"KS={_ks_gb:.3f}  p={_p_gb:.3f}",
                xref="paper", yref="paper", x=0.99, y=0.99,
                showarrow=False, font=dict(size=10),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ccc", borderwidth=1, borderpad=4,
            )
            gbfig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            gbfig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            st.plotly_chart(gbfig, use_container_width=True)

        with col_d4:
            st.markdown("**Blast rate — MC vs CSDI**")
            _mc_br   = res['sim_blasts'] / np.maximum(res['sim_pn2'], 1)
            _mc_br   = np.clip(_mc_br, 0, 1)
            _csdi_br = _csdi_df["Частота формирования бластоцист"].values
            _ks_br, _p_br = ks_2samp(_mc_br, _csdi_br)

            brfig = go.Figure()
            brfig.add_trace(go.Histogram(
                x=_mc_br * 100, name="MC", opacity=0.68,
                marker=dict(
                    color=hex_rgba(C["blue"], 0.68),
                    line=dict(color=hex_rgba(C["blue"], 0.90), width=0.8),
                ),
                xbins=dict(size=2),
            ))
            brfig.add_trace(go.Histogram(
                x=_csdi_br * 100, name="CSDI", opacity=0.68,
                marker=dict(
                    color=hex_rgba(C["green"], 0.68),
                    line=dict(color=hex_rgba(C["green"], 0.90), width=0.8),
                ),
                xbins=dict(size=2),
            ))
            brfig.update_layout(
                **LAYOUT,
                barmode="overlay",
                height=280,
                xaxis=dict(title="Blast rate (%)"),
                yaxis=dict(title="Частота"),
            )
            brfig.add_annotation(
                text=f"KS={_ks_br:.3f}  p={_p_br:.3f}",
                xref="paper", yref="paper", x=0.99, y=0.99,
                showarrow=False, font=dict(size=10),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ccc", borderwidth=1, borderpad=4,
            )
            brfig.update_xaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            brfig.update_yaxes(gridcolor="rgba(200,210,220,0.35)", zeroline=False, tickfont=dict(size=11))
            st.plotly_chart(brfig, use_container_width=True)

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
                "✅ схожи" if _p_bl    > 0.05 else "⚠️ различны",
                "✅ схожи" if _p_tgbdr > 0.05 else "⚠️ различны",
                "✅ схожи" if _p_gb    > 0.05 else "⚠️ различны",
                "✅ схожи" if _p_br    > 0.05 else "⚠️ различны",
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
            _icon = "✅"
            _verdict = "подтверждает"
            _color_word = "сходство"
        else:
            _box_class = "diff-warn"
            _icon = "⚠️"
            _verdict = "не подтверждает"
            _color_word = "расхождение"

        st.markdown(f"""
        <div class="{_box_class}">
        <b>{_icon} Интерпретация верификации</b><br><br>
        CSDI Hybrid v3 (1000 траекторий, DDIM 50 шагов) генерирует
        эмбриологический каскад без параметрических допущений MC-пайплайна.<br><br>
        KS-тест <b>{_verdict}</b> статистическое {_color_word}
        финальных распределений: {_n_pass}/4 переменных прошли порог p&nbsp;>&nbsp;0.05.<br><br>
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
        with st.expander("🔧 Параметры CSDI-условия (conditioning)"):
            st.markdown(f"""
            CSDI-модель обусловлена на upstream-результатах MC:

            | Параметр | Значение (из MC) |
            |---|---|
            | Фолликулов | {_foll_count} |
            | ОКК (MC медиана) | {_okk_med} |
            | MII как инсеминированные | {_mii_med} |
            | 2PN (MC медиана) | {_pn2_med} |
            | Частота получения ОКК | {_okk_rate:.2f} |
            | Частота оплодотворения | {_fert_rate:.2f} |
            | KPIScore (MC медиана) | {_kpi:.1f} |

            *Генерация: 1000 траекторий, DDIM 50 шагов*
            """)

# ── TAB 8: GAT Graph ──────────────────────────────────────────
with tabs[7]:
    st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;margin:0 0 6px 0">🕸️ Graph Attention Transformer — граф клинических соседей</p>', unsafe_allow_html=True)

    if not _gnn_bundle.get('available'):
        _err = _gnn_bundle.get('error', 'Модель не загружена')
        st.info(f"ℹ️ GNN модель недоступна: {_err}\n\n"
                f"Поместите `gnn_ivf_model.pt` в папку `models/` и перезапустите приложение.")

    elif _p_gnn_raw is None:
        st.info("▶ Нажмите **Запустить расчёт** чтобы получить предсказание Graph Transformer.")

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
            st.plotly_chart(_fig_gnn_tab, use_container_width=True)
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
                    st.plotly_chart(_fig_gnn_tab, use_container_width=True)
                else:
                    st.warning("Не удалось построить граф: данные соседей недоступны.")
            except Exception as _gnn_fig_err:
                st.error(f"Ошибка построения графа: {_gnn_fig_err}")

        # ── Пояснение ─────────────────────────────────────────
        with st.expander("ℹ️ Как читать этот график"):
            st.markdown(f"""
**Левая панель — сетевой граф:**
- ⭐ **Звезда** в центре — текущая пациентка
- ⚫ **Круги** — 10 клинически наиболее похожих пациентов из обучающей когорты
- **Цвет** узла: 🟢 зелёный = высокая GNN-вероятность, 🔴 красный = низкая
- **Размер** узла и **толщина** ребра ∝ косинусное сходство профилей
- Сходство вычисляется по клиническим показателям: возраст, ОКК, бластоцисты,
  частоты оплодотворения / бластуляции и др. *(без учёта KAT-скора)*

**Правая панель — распределение GNN-вероятностей соседей:**
- Каждый бар = один сосед, отсортированы по вероятности
- 🔵 Пунктирная линия = вероятность **текущей пациентки** по Graph Transformer
- Серая линия = медиана вероятностей среди соседей

**Интерпретация:**
Если пациентка попадает в область высоких вероятностей среди похожих случаев —
это дополнительный аргумент в пользу оптимистичного прогноза.
Если её позиция ниже медианы соседей — Graph Transformer выявляет
дополнительные неблагоприятные паттерны относительно похожих пациентов.

> Ансамбль: **{_w_gnn:.0%}×GNN + {1-_w_gnn:.0%}×KAT**
            """)

        # ── Технические детали (для исследователя) ───────────
        with st.expander("🔧 Технические параметры модели"):
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
st.caption("IVF Digital Twin v6.2  ·  Sergeev et al., 2025  ·  "
           "embryossa@gmail.com  ·  "
           "Research prototype — not for standalone clinical use")


# ══════════════════════════════════════════════════════════
#  PDF ОТЧЁТ
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.header("📄 Экспорт отчёта")

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
            "📄 Сформировать PDF",
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
                )

                _fname = f"IVF_Report_{(pdf_patient_id or 'patient').replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.success(f"✅ PDF готов — {len(_pdf_bytes)//1024} КБ")

                # ── DT Analytics: пишем строку только вместе с PDF ──────
                _analytics_record_id = _save_analytics(
                    res          = res,
                    _eb          = _eb,
                    age          = float(age),
                    amh          = float(amh),
                    afc          = int(afc),
                    bmi          = float(bmi),
                    attempt      = int(attempt),
                    sperm_source = sperm_source,
                    follicles    = follicles,
                    known        = known,
                    clinic_name  = st.session_state.get("ivf_clinic_name", ""),
                    patient_name = pdf_patient_name or "",
                    patient_id   = pdf_patient_id   or "",
                    csdi_result  = _ss.get("csdi_result"),
                )
                if _analytics_record_id:
                    st.caption(f"📊 Аналитика сохранена · ID записи: `{_analytics_record_id}`")
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
