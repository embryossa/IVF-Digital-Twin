# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
IVF Digital Twin v6.2 — Вычислительное ядро (без Streamlit)
=============================================================
Этот модуль даёт пакетному скрипту и app.py общий код:
  - load_nn_model()          — KAT (KAN + FT-Transformer)
  - load_csdi_model()        — CSDI Hybrid v3 (L5)
  - load_gnn_bundle()        — Graph Attention Transformer (L6)
  - predict_single_patient() — полный расчёт L1–L6 для одной пациентки
  - save_analytics_record()  — запись строки в dt_predictions.csv
    (дублирует _save_analytics из app.py, но без зависимостей на st)

Размещение: src/ivf_core.py
Запуск отдельно (тест):
    python src/ivf_core.py
"""

from __future__ import annotations
import os, sys, csv, uuid as _uuid, warnings, math
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import numpy as np

warnings.filterwarnings("ignore")

# ── Пути ─────────────────────────────────────────────────────────────────────
_THIS_FILE = os.path.abspath(__file__)
_SRC_DIR   = os.path.dirname(_THIS_FILE)           # …/src/
_BASE_DIR  = os.path.dirname(_SRC_DIR)             # …/ (рядом с app.py)

# src/ должен быть в sys.path для относительных импортов
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

# ══════════════════════════════════════════════════════════════════════════════
#  ПОДКЛЮЧЕНИЕ PIPELINE (ivf_digital_twin.py / .pyd)
#  Повторяет логику app.py строки 302-328
# ══════════════════════════════════════════════════════════════════════════════
_src_py  = os.path.join(_SRC_DIR, "ivf_digital_twin.py")
_src_pyd = any(
    f.startswith("ivf_digital_twin") and f.endswith(".pyd")
    for f in os.listdir(_SRC_DIR)
) if os.path.isdir(_SRC_DIR) else False

if _src_pyd:
    import ivf_digital_twin as _ivf_mod
    globals().update({k: getattr(_ivf_mod, k)
                      for k in dir(_ivf_mod) if not k.startswith("__")})
elif os.path.exists(_src_py):
    _g = globals()
    _orig_file = _g.get("__file__", "")
    _g["__file__"] = _src_py
    _pipeline_code = open(_src_py, encoding="utf-8").read()
    _pipeline_code = _pipeline_code.replace("if __name__ ==", "if False and __name__ ==")
    exec(compile(_pipeline_code, _src_py, "exec"), _g)
    _g["__file__"] = _orig_file
else:
    raise RuntimeError(
        "ivf_digital_twin не найден ни как .py, ни как .pyd в src/\n"
        f"Ожидается: {_src_py}"
    )

# ── Уже подключены: PatientInput, KnownValues, run_pipeline_extended,
#    load_nn_ensemble, NN_MODEL_PATHS, NN_LIBS_AVAILABLE — из pipeline


# ══════════════════════════════════════════════════════════════════════════════
#  ПОДКЛЮЧЕНИЕ CSDI Hybrid v3 (L5)
#  Повторяет логику app.py строки 347-390
# ══════════════════════════════════════════════════════════════════════════════
_CSDI_CLASS_READY = False
CSDI_LOAD_ERROR   = ""

_csdi_pyd = any(
    f.startswith("embryo_csdi_v3") and f.endswith(".pyd")
    for f in os.listdir(_SRC_DIR)
) if os.path.isdir(_SRC_DIR) else False

_csdi_candidates = [
    os.path.join(_SRC_DIR, "embryo_csdi_v3.py"),
    os.path.join(_BASE_DIR, "embryo_csdi_v3.py"),
]

if _csdi_pyd:
    try:
        import embryo_csdi_v3 as _csdi_mod
        globals().update({k: getattr(_csdi_mod, k)
                          for k in dir(_csdi_mod) if not k.startswith("__")})
        _CSDI_CLASS_READY = True
    except Exception as _e:
        CSDI_LOAD_ERROR = str(_e)
else:
    for _cf in _csdi_candidates:
        if os.path.exists(_cf):
            _g2 = globals()
            _orig2 = _g2.get("__file__", "")
            try:
                _csdi_code = open(_cf, encoding="utf-8").read()
                _csdi_code = _csdi_code.replace("if __name__ ==", "if False and __name__ ==")
                _g2["__file__"] = _cf
                exec(compile(_csdi_code, _cf, "exec"), _g2)
                _CSDI_CLASS_READY = True
            except Exception as _e:
                CSDI_LOAD_ERROR = str(_e)
            finally:
                _g2["__file__"] = _orig2
            break

_CSDI_MODEL_DIRS = [
    os.path.join(_BASE_DIR, "models", "embryo_v3_model"),
    os.path.join(_BASE_DIR, "embryo_v3_model"),
    "embryo_v3_model",
]

# ── GNN ────────────────────────────────────────────────────────────────────
_GNN_IMPORT_OK  = False
_GNN_LOAD_ERROR = ""
try:
    from gnn_predictor import load_gnn_model    as _load_gnn_model
    from gnn_predictor import predict_gnn       as _predict_gnn
    from gnn_predictor import build_patient_features as _build_gnn_features
    _GNN_IMPORT_OK = True
except ImportError as _e:
    _GNN_LOAD_ERROR = str(_e)


# ══════════════════════════════════════════════════════════════════════════════
#  КЕШИ МОДЕЛЕЙ (загружаются один раз на процесс)
# ══════════════════════════════════════════════════════════════════════════════
_NN_CACHE:   object = "NOT_LOADED"   # None = загружали, не нашли
_CSDI_CACHE: object = "NOT_LOADED"
_GNN_CACHE:  dict   = None


def load_nn_model():
    """KAT (KAN + FT-Transformer). None если недоступны веса/torch."""
    global _NN_CACHE
    if _NN_CACHE == "NOT_LOADED":
        _NN_CACHE = load_nn_ensemble()  # из ivf_digital_twin.py
    return _NN_CACHE


def get_nn_model_info(nn_model=None) -> dict:
    """
    Возвращает человекочитаемую информацию о загруженном KAT-слое:
    KAN/FT веса ансамбля, наличие калибровки и строку для вывода/CSV.

    Работает мягко: если модель недоступна или wrapper не содержит этих
    атрибутов, не ломает расчёт, а возвращает пустые значения.
    """
    if nn_model is None:
        nn_model = load_nn_model()

    info = {
        "available": nn_model is not None,
        "kan_weight": None,
        "ft_weight": None,
        "calibrated": False,
        "source_suffix": "",
    }
    if nn_model is None:
        return info

    # В обновлённом ivf_digital_twin wrapper обычно содержит ensemble_model.
    ens = getattr(nn_model, "ensemble_model", None)
    if ens is None:
        ens = getattr(nn_model, "model", None)

    try:
        raw = getattr(ens, "raw_weights", None) if ens is not None else None
        if raw is not None:
            # Новый KAT: raw_weights -> softmax -> [KAN, FT]
            if "torch" in globals():
                w = torch.softmax(raw.detach().cpu(), dim=0).numpy()  # noqa: F821
            else:
                arr = np.asarray(raw.detach().cpu().numpy(), dtype=float)
                exp = np.exp(arr - np.max(arr))
                w = exp / exp.sum()
            info["kan_weight"] = float(w[0])
            info["ft_weight"] = float(w[1])
        else:
            # Старый формат: отдельные kan_weight / ft_weight без softmax.
            kw = getattr(ens, "kan_weight", None) if ens is not None else None
            fw = getattr(ens, "ft_weight", None) if ens is not None else None
            if kw is not None and fw is not None:
                kw = float(kw.detach().cpu().numpy().ravel()[0])
                fw = float(fw.detach().cpu().numpy().ravel()[0])
                total = kw + fw
                if total > 0:
                    info["kan_weight"] = kw / total
                    info["ft_weight"] = fw / total
    except Exception:
        pass

    # Возможные имена атрибутов в разных wrapper-версиях.
    info["calibrated"] = any(
        hasattr(nn_model, name)
        for name in ("isotonic", "calibrator", "calibration_model", "ir")
    )

    if info["kan_weight"] is not None and info["ft_weight"] is not None:
        info["source_suffix"] = (
            f" | weights: KAN={info['kan_weight']:.3f}, "
            f"FT={info['ft_weight']:.3f}"
        )
    if info["calibrated"]:
        info["source_suffix"] += " | isotonic calibrated"

    return info


def load_csdi_model():
    """CSDI Hybrid v3. None если нет модели или torch."""
    global _CSDI_CACHE
    if _CSDI_CACHE == "NOT_LOADED":
        if not _CSDI_CLASS_READY:
            _CSDI_CACHE = None
            return None
        for _d in _CSDI_MODEL_DIRS:
            _cfg      = os.path.join(_d, "config.json")
            _wts_p    = os.path.join(_d, "csdi_weights.pt")
            _wts_enc  = os.path.join(_d, "csdi_weights.pt.enc")
            if os.path.isfile(_cfg) and (os.path.isfile(_wts_p) or
                                          os.path.isfile(_wts_enc)):
                try:
                    _CSDI_CACHE = EmbryoHybridV3.load(_d)  # noqa: F821
                    break
                except Exception as _e:
                    print(f"[CORE] CSDI load error: {_e}")
                    _CSDI_CACHE = None
        if _CSDI_CACHE == "NOT_LOADED":
            _CSDI_CACHE = None
    return _CSDI_CACHE


def load_gnn_bundle():
    """GNN (Graph Attention Transformer). {'available': False} если нет модели."""
    global _GNN_CACHE
    if _GNN_CACHE is None:
        if not _GNN_IMPORT_OK:
            _GNN_CACHE = {"available": False, "error": _GNN_LOAD_ERROR}
        else:
            _GNN_CACHE = _load_gnn_model(base_dir=_BASE_DIR)
    return _GNN_CACHE


# ══════════════════════════════════════════════════════════════════════════════
#  ESTEVES BANKING (из app.py строки 932-1060)
# ══════════════════════════════════════════════════════════════════════════════
def _compute_esteves_banking(patient_age: float, sperm_src: str,
                              res_pipeline: dict) -> dict:
    """
    Esteves et al. банкинг-модель: P(euploid blastocyst | MII oocyte).
    Скопировано из app.py функции _compute_esteves_banking.
    """
    from scipy.stats import binom as _binom

    fert_by_source = {
        "ejaculate":      0.76, "testicular_NOA": 0.56,
        "testicular_OA":  0.68, "epididymal":     0.66,
    }
    fert_r = fert_by_source.get(sperm_src, 0.76)

    if patient_age < 35:   blast_r = 0.48
    elif patient_age < 38: blast_r = 0.44
    elif patient_age < 41: blast_r = 0.38
    else:                  blast_r = 0.30

    eupl_table = {
        (0,  35): 0.68, (35, 37): 0.57, (37, 39): 0.48,
        (39, 41): 0.38, (41, 43): 0.29, (43, 99): 0.18,
    }
    eupl_r = 0.38
    for (lo, hi), rate in eupl_table.items():
        if lo <= patient_age < hi:
            eupl_r = rate
            break

    p_per_mii = fert_r * blast_r * eupl_r

    # Forwards: сколько эуплоидных при медиане MII пациентки
    mii_med = int(res_pipeline.get("mii_med", 0)) or 0
    fwd = None
    if mii_med > 0:
        n_samples = 20_000
        draws = np.random.binomial(mii_med, p_per_mii, n_samples)
        p_at_least = {k: float(np.mean(draws >= k))
                      for k in range(0, min(mii_med, 15) + 2)}
        fwd = {
            "mii":          mii_med,
            "mean":         float(draws.mean()),
            "median":       float(np.median(draws)),
            "p80":          float(np.percentile(draws, 80)),
            "p_at_least":   p_at_least,
        }

    return {
        "p_per_mii":          p_per_mii,
        "p_transfer_used":    res_pipeline.get("p_per_transfer", 0.45),
        "forward_at_median":  fwd,
        "patient_mii_median": mii_med,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ОСНОВНАЯ ФУНКЦИЯ: ПОЛНЫЙ РАСЧЁТ ОДНОЙ ПАЦИЕНТКИ (L1–L6)
# ══════════════════════════════════════════════════════════════════════════════
def predict_single_patient(
    age:          float,
    amh:          float  = 2.0,
    afc:          int    = 12,
    bmi:          float  = 23.0,
    attempt:      int    = 1,
    follicles:    Optional[int]   = None,
    sperm_source: str             = "ejaculate",
    known_okk:    Optional[int]   = None,
    known_mii:    Optional[int]   = None,
    known_pn2:    Optional[int]   = None,
    known_blasts: Optional[int]   = None,
    known_good:   Optional[int]   = None,
    known_euploid:Optional[int]   = None,
    clinic_successes: Optional[List[int]] = None,
    clinic_trials:    Optional[List[int]] = None,
    n_sim:        int    = 2000,
    seed:         int    = 42,
) -> dict:
    """
    Запускает полный расчёт Digital Twin (L1–L6) для одной пациентки.
    Точно воспроизводит порядок вызовов из блока `if run_btn:` в app.py.

    Возвращает словарь:
        res           — dict из run_pipeline_extended (L1-L4 + байес + Esteves)
        eb            — dict Esteves banking (_compute_esteves_banking)
        csdi_result   — dict из csdi_model.mc_sample() или None
        gnn_result    — dict из predict_gnn() или {'available': False, ...}
        nn_available  — bool
        csdi_available— bool
        gnn_available — bool
        p_kat_raw     — float или None
        p_nvsa        — float или None
        p_csdi        — float или None
        p_gnn_raw     — float или None
        p_gnn_ens     — float или None (w_gnn × GNN + (1-w_gnn) × KAT)
    """
    np.random.seed(seed)

    # ── PatientInput / KnownValues ────────────────────────────────────────
    patient = PatientInput(  # noqa: F821
        female_age=float(age),
        amh=float(amh),
        afc=int(afc),
        bmi=float(bmi),
    )
    known = KnownValues(  # noqa: F821
        okk=known_okk,
        mii=known_mii,
        pn2=known_pn2,
        blasts=known_blasts,
        good=known_good,
        euploid=known_euploid,
    )

    # ── L1–L4 + байес + Esteves + кластер ────────────────────────────────
    nn_model = load_nn_model()
    res = run_pipeline_extended(  # noqa: F821
        patient, known=known,
        attempt_number=int(attempt),
        follicles=follicles,
        nn_model=nn_model,
        clinic_real_successes=clinic_successes,
        clinic_real_trials=clinic_trials,
        max_attempts_curve=6,
        sperm_source=sperm_source,
        n=n_sim,
    )

    # ── Esteves banking (как в app.py строки 932+) ───────────────────────
    eb = _compute_esteves_banking(float(age), sperm_source, res)

    # ── L3: из nn_prediction (уже внутри run_pipeline_extended) ──────────
    _nn   = res.get("nn_prediction", {})
    _nvsa = res.get("nn_nvsa", {})
    p_kat_raw = _nn.get("base_prob_mean")
    p_nvsa    = _nvsa.get("adjusted_mean")
    nn_info   = get_nn_model_info(nn_model)

    # ── L5: CSDI Hybrid v3 ───────────────────────────────────────────────
    csdi_model  = load_csdi_model()
    csdi_result = None
    p_csdi      = None
    if csdi_model is not None:
        try:
            _foll_count = follicles if follicles is not None else int(afc)
            _okk_med  = max(1, int(res["okk_med"]))
            _mii_med  = max(1, int(res["mii_med"]))
            _pn2_med  = max(1, int(res["pn2_med"]))
            _okk_rate = min(1.0, _okk_med / max(_foll_count, 1))
            _fert_rate = min(1.0, _pn2_med / max(_mii_med, 1))
            _kpi = float(res["kpi_score_median"])

            patient_csdi = {
                "Количество фолликулов":  float(_foll_count),
                "Число ОКК":              float(_okk_med),
                "Число инсеминированных": float(_mii_med),
                "2 pN":                   float(_pn2_med),
                "Частота получения ОКК":  _okk_rate,
                "Частота оплодотворения": _fert_rate,
                "KPIScore":               _kpi,
            }
            csdi_result = csdi_model.mc_sample(patient_csdi, n_samples=1000)
            p_csdi = csdi_result.get("P_pregnancy")
        except Exception as _e:
            print(f"[CORE] CSDI inference error: {_e}")

    # ── L6: GNN ──────────────────────────────────────────────────────────
    gnn_bundle  = load_gnn_bundle()
    gnn_result  = {"available": False, "gnn_prob": None,
                   "ensemble_prob": None, "w_gnn": 0.35}
    p_gnn_raw   = None
    p_gnn_ens   = None
    if gnn_bundle.get("available"):
        try:
            gnn_feats = _build_gnn_features(  # noqa: F821
                age=float(age),
                afc=int(afc),
                attempt=int(attempt),
                res=res,
                known=known,
                p_kat_raw=p_kat_raw,
            )
            gnn_result = _predict_gnn(  # noqa: F821
                gnn_bundle, gnn_feats, prai_score=p_kat_raw
            )
            p_gnn_raw = gnn_result.get("gnn_prob")
            p_gnn_ens = gnn_result.get("ensemble_prob")
        except Exception as _e:
            print(f"[CORE] GNN inference error: {_e}")

    return {
        "res":             res,
        "eb":              eb,
        "csdi_result":     csdi_result,
        "gnn_result":      gnn_result,
        "known":           known,
        # Удобные shortcut-поля для CSV-записи
        "nn_available":    nn_model is not None,
        "csdi_available":  csdi_model is not None,
        "gnn_available":   gnn_bundle.get("available", False),
        "p_kat_raw":       p_kat_raw,
        "p_nvsa":          p_nvsa,
        "nn_info":         nn_info,
        "p_csdi":          p_csdi,
        "p_gnn_raw":       p_gnn_raw,
        "p_gnn_ens":       p_gnn_ens,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ЗАПИСЬ АНАЛИТИКИ — точная копия _save_analytics из app.py
#  но без зависимостей на Streamlit
# ══════════════════════════════════════════════════════════════════════════════

_ANALYTICS_COLUMNS = [
    "record_id", "timestamp", "clinic_name",
    "patient_name", "patient_id",
    "age", "amh", "afc", "bmi", "attempt_number", "sperm_source", "follicles_tvp",
    "known_okk", "known_mii", "known_pn2", "known_blasts", "known_good", "known_euploid",
    "med_okk", "med_mii", "med_pn2", "med_blasts", "med_good", "med_euploid", "med_warmed",
    "p025_okk", "p975_okk", "p025_blasts", "p975_blasts", "p025_good", "p975_good",
    "p_per_transfer", "p_cum_if_viable", "p_overall_cycle", "p_viable",
    "p_cancel_risk", "rate_ci_low", "rate_ci_high",
    "bayes_mean", "bayes_ci_low", "bayes_ci_high", "bayes_prior_mean", "bayes_prior_type",
    "p_kat_raw", "p_nvsa", "ci_kat_low", "ci_kat_high", "ci_nvsa_low", "ci_nvsa_high",
    "p_csdi", "csdi_ci_low", "csdi_ci_high",
    # GNN (новые колонки относительно app.py — добавляем)
    "p_gnn_raw", "p_gnn_ens", "w_gnn",
    "dominant_cluster", "cluster_c0_prob", "cluster_c1_prob", "cluster_c2_prob",
    "ohss_moderate", "ohss_severe", "ohss_any",
    "p_no_blast", "p_no_good_blast",
    "banking_p_per_mii", "banking_expected_euploid",
    # слои доступности (для отладки согласованности)
    "nn_source", "csdi_available", "gnn_available",
    # реальные данные из выгрузки (для сравнения)
    "real_pn2", "real_cleav", "real_bl", "real_goodbl", "real_cryo",
    "real_outcome", "outcome_date", "notes",
]


def save_analytics_record(
    result:       dict,
    age:          float,
    amh:          float,
    afc:          int,
    bmi:          float,
    attempt:      int,
    sperm_source: str,
    follicles:    Optional[int],
    clinic_name:  str = "",
    patient_name: str = "",
    patient_id:   str = "",
    real_pn2:     object = "",
    real_cleav:   object = "",
    real_bl:      object = "",
    real_goodbl:  object = "",
    real_cryo:    object = "",
    real_outcome: str = "",
    notes:        str = "",
    analytics_csv: Optional[str] = None,
) -> Optional[str]:
    """
    Формирует строку и дописывает её в dt_predictions.csv.
    result — словарь из predict_single_patient().
    Возвращает record_id или None при ошибке.
    """
    try:
        res        = result["res"]
        eb         = result["eb"]
        known      = result["known"]
        csdi_result= result.get("csdi_result")
        gnn_result = result.get("gnn_result", {})
        post       = res.get("posterior", {})
        ca         = res.get("cluster_analysis", {})
        ohss       = res.get("ohss", {})
        empty      = res.get("empty", {})
        _nn        = res.get("nn_prediction", {})
        _nvsa      = res.get("nn_nvsa", {})
        _nn_info   = result.get("nn_info", {})

        probs   = ca.get("cluster_probs", {})
        ci_kat  = _nn.get("base_prob_ci",  (None, None))
        ci_nvsa = _nvsa.get("adjusted_ci", (None, None))

        _p_csdi = _csdi_ci_l = _csdi_ci_h = None
        if csdi_result and isinstance(csdi_result, dict):
            _p_csdi = csdi_result.get("P_pregnancy")
            _ci95   = csdi_result.get("CI_95", (None, None))
            _csdi_ci_l = _ci95[0] if _ci95 else None
            _csdi_ci_h = _ci95[1] if _ci95 else None

        _p_mii = _exp_eu = None
        if isinstance(eb, dict):
            _p_mii = eb.get("p_per_mii")
            _fwd   = eb.get("forward_at_median")
            if isinstance(_fwd, dict):
                _exp_eu = _fwd.get("mean")

        def _pct(arr, q):
            try:    return int(np.percentile(arr, q))
            except: return ""

        def _r4(v):
            if v is None: return ""
            try:    return round(float(v), 4)
            except: return ""

        def _knw(attr):
            if known is None: return ""
            v = getattr(known, attr, None)
            return v if v is not None else ""

        row = {
            "record_id":       str(_uuid.uuid4()),
            "timestamp":       datetime.now().isoformat(timespec="seconds"),
            "clinic_name":     clinic_name or "",
            "patient_name":    patient_name or "",
            "patient_id":      patient_id   or "",
            "age":             age,
            "amh":             amh,
            "afc":             afc,
            "bmi":             bmi,
            "attempt_number":  attempt,
            "sperm_source":    sperm_source,
            "follicles_tvp":   follicles if follicles else "",
            "known_okk":       _knw("okk"),
            "known_mii":       _knw("mii"),
            "known_pn2":       _knw("pn2"),
            "known_blasts":    _knw("blasts"),
            "known_good":      _knw("good"),
            "known_euploid":   _knw("euploid"),
            "med_okk":         res.get("okk_med", ""),
            "med_mii":         res.get("mii_med", ""),
            "med_pn2":         res.get("pn2_med", ""),
            "med_blasts":      res.get("blasts_med", ""),
            "med_good":        res.get("good_med", ""),
            "med_euploid":     res.get("euploid_med", ""),
            "med_warmed":      res.get("warmed_med", ""),
            "p025_okk":        _pct(res.get("sim_okk", []), 2.5),
            "p975_okk":        _pct(res.get("sim_okk", []), 97.5),
            "p025_blasts":     _pct(res.get("sim_blasts", []), 2.5),
            "p975_blasts":     _pct(res.get("sim_blasts", []), 97.5),
            "p025_good":       _pct(res.get("sim_good", []), 2.5),
            "p975_good":       _pct(res.get("sim_good", []), 97.5),
            "p_per_transfer":  _r4(res.get("p_per_transfer", 0)),
            "p_cum_if_viable": _r4(res.get("p_cum_if_viable", 0)),
            "p_overall_cycle": _r4(res.get("p_overall_cycle", 0)),
            "p_viable":        _r4(res.get("p_viable", 0)),
            "p_cancel_risk":   _r4(float(np.mean(res["sim_okk"] == 0))),
            "rate_ci_low":     _r4(res.get("rate_ci", (0, 0))[0]),
            "rate_ci_high":    _r4(res.get("rate_ci", (0, 0))[1]),
            "bayes_mean":      _r4(post.get("mean", 0)),
            "bayes_ci_low":    _r4(post.get("ci_low", 0)),
            "bayes_ci_high":   _r4(post.get("ci_high", 0)),
            "bayes_prior_mean":_r4(post.get("prior_mean", 0)),
            "bayes_prior_type":post.get("prior_type", ""),
            "p_kat_raw":       _r4(_nn.get("base_prob_mean")),
            "p_nvsa":          _r4(_nvsa.get("adjusted_mean")),
            "ci_kat_low":      _r4(ci_kat[0])  if ci_kat[0]  is not None else "",
            "ci_kat_high":     _r4(ci_kat[1])  if ci_kat[1]  is not None else "",
            "ci_nvsa_low":     _r4(ci_nvsa[0]) if ci_nvsa[0] is not None else "",
            "ci_nvsa_high":    _r4(ci_nvsa[1]) if ci_nvsa[1] is not None else "",
            "p_csdi":          _r4(_p_csdi)    if _p_csdi    is not None else "",
            "csdi_ci_low":     _r4(_csdi_ci_l) if _csdi_ci_l is not None else "",
            "csdi_ci_high":    _r4(_csdi_ci_h) if _csdi_ci_h is not None else "",
            # GNN
            "p_gnn_raw":       _r4(gnn_result.get("gnn_prob")),
            "p_gnn_ens":       _r4(gnn_result.get("ensemble_prob")),
            "w_gnn":           _r4(gnn_result.get("w_gnn", 0.35)),
            # Кластер
            "dominant_cluster":ca.get("dominant_cluster", ""),
            "cluster_c0_prob": _r4(probs.get(0)) if probs.get(0) is not None else "",
            "cluster_c1_prob": _r4(probs.get(1)) if probs.get(1) is not None else "",
            "cluster_c2_prob": _r4(probs.get(2)) if probs.get(2) is not None else "",
            # Риски
            "ohss_moderate":   _r4(ohss.get("p_moderate_ohss", 0)),
            "ohss_severe":     _r4(ohss.get("p_severe_ohss", 0)),
            "ohss_any":        _r4(ohss.get("p_any_ohss", 0)),
            "p_no_blast":      _r4(empty.get("p_no_blast", 0)),
            "p_no_good_blast": _r4(empty.get("p_no_good_blast", 0)),
            # Банкинг
            "banking_p_per_mii":        _r4(_p_mii),
            "banking_expected_euploid": _r4(_exp_eu),
            # Слои доступности
            "nn_source":      (_nn.get("source", "") + _nn_info.get("source_suffix", "")),
            "csdi_available": int(result.get("csdi_available", False)),
            "gnn_available":  int(result.get("gnn_available", False)),
            # Реальные данные из выгрузки (для сравнения)
            "real_pn2":       real_pn2     if real_pn2     != "" else "",
            "real_cleav":     real_cleav   if real_cleav   != "" else "",
            "real_bl":        real_bl      if real_bl      != "" else "",
            "real_goodbl":    real_goodbl  if real_goodbl  != "" else "",
            "real_cryo":      real_cryo    if real_cryo    != "" else "",
            "real_outcome":   real_outcome or "",
            "outcome_date":   "",
            "notes":          notes or "",
        }

        # Куда пишем
        if analytics_csv:
            csv_path = Path(analytics_csv)
        else:
            csv_path = (Path(_BASE_DIR) / "dt_analytics_data" / "dt_predictions.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Проверяем заголовок: если файл существует но схема не совпадает —
        # архивируем старый файл и начинаем новый (данные не теряются)
        need_header = True
        if csv_path.exists():
            try:
                with open(csv_path, "r", encoding="utf-8") as _chk:
                    existing_cols = _chk.readline().strip().split(",")
                if existing_cols == _ANALYTICS_COLUMNS:
                    need_header = False
                else:
                    from datetime import datetime as _dt2
                    _arc = csv_path.with_name(
                        csv_path.stem + f"_schema_v{len(existing_cols)}_"
                        + _dt2.now().strftime("%Y%m%d_%H%M%S") + ".csv"
                    )
                    csv_path.rename(_arc)
                    print(f"[CORE] Schema mismatch ({len(existing_cols)} vs "
                          f"{len(_ANALYTICS_COLUMNS)} cols): archived -> {_arc.name}")
                    need_header = True
            except Exception:
                need_header = True

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_ANALYTICS_COLUMNS)
            if need_header:
                writer.writeheader()
            writer.writerow(row)

        return row["record_id"]

    except Exception as exc:
        print(f"[CORE] save_analytics_record error: {exc}")
        return None


# ── Быстрый тест-прогон ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== IVF Core self-test ===")
    result = predict_single_patient(
        age=34, amh=2.1, afc=13, bmi=23,
        attempt=1, follicles=15,
        known_okk=10, known_mii=9,
    )
    r = result["res"]
    print(f"okk_med={r['okk_med']}  mii_med={r['mii_med']}  "
          f"blasts_med={r['blasts_med']}  good_med={r['good_med']}")
    print(f"p_per_transfer={r['p_per_transfer']:.4f}  "
          f"p_overall_cycle={r['p_overall_cycle']:.4f}")
    print(f"bayes_mean={r['posterior']['mean']:.4f}")
    nn_info = result.get("nn_info", {})
    nn_suffix = nn_info.get("source_suffix", "")
    print(f"NN source : {r['nn_prediction']['source']}{nn_suffix}")
    if nn_info.get("kan_weight") is not None:
        print(f"KAT weights: KAN={nn_info['kan_weight']:.3f}  FT={nn_info['ft_weight']:.3f}")
    print(f"CSDI      : {result['p_csdi']}")
    print(f"GNN ens   : {result['p_gnn_ens']}")
    print(f"cluster   : {r['cluster_analysis']['dominant_cluster']}")
    print("=== DONE ===")
