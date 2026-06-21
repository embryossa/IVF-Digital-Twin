"""
llm_consultant.py — IVF Digital Twin v7.0
LLM-слой консультанта поверх ансамбля. Offline-safe (Ollama localhost).

Три уровня:
  Tier 0 — нарратор: pre-classified JSON → клинический текст для врача
  Tier 1 — аналитик ансамбля: raw per-layer outputs → матрица согласованности
  Tier 2 — агент: function calling над движком DT (gemma4)

Принцип: Python классифицирует, LLM объясняет — не наоборот.
LLM не вычисляет вероятности и не ставит диагнозов.
Зависимости: стандартная библиотека + requests.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterator, List, Optional

import requests

# ──────────────────────────────────────────────────────────────────────────
#  КОНФИГУРАЦИЯ
# ──────────────────────────────────────────────────────────────────────────
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")  # 127.0.0.1, не localhost (обход IPv6 в Windows)
MEDGEMMA    = os.environ.get("DT_LLM_NARRATOR", "medgemma1.5")  # Tier 0
GEMMA4      = os.environ.get("DT_LLM_AGENT", "gemma4")          # Tier 2

# Таймауты — переопределяются переменными окружения для медленных CPU.
# DT_LLM_TIMEOUT_READ задаёт read-таймаут в секундах (по умолчанию 1200).
# Для GPU можно снизить до 300; для медленных CPU увеличить до 1800.
_TIMEOUT_READ   = int(os.environ.get("DT_LLM_TIMEOUT_READ",  1200))
_TIMEOUT        = (5, _TIMEOUT_READ)
_WARMUP_TIMEOUT = (5, int(os.environ.get("DT_LLM_WARMUP_TIMEOUT", 900)))

_KEEP_ALIVE    = os.environ.get("DT_LLM_KEEP_ALIVE", "1h")
_TEMPERATURE   = 0.15
_NARRATOR_TEMP = 0.35         # нарратор — живой язык; числа из контекста, не придумываются
_NARRATOR_THINK = False  # chain-of-thought отключён — на CPU только съедает время

# Куда писать аудит (рядом с существующим analytics-пайплайном).
_AUDIT_DIR  = os.environ.get("DT_LLM_AUDIT_DIR", "dt_analytics_data")
_AUDIT_FILE = os.path.join(_AUDIT_DIR, "llm_consult_log.jsonl")



# ──────────────────────────────────────────────────────────────────────────
#  ВСПОМОГАТЕЛЬНЫЙ ЭКСТРАКТОР: k-NN СОСЕДИ GAT
# ──────────────────────────────────────────────────────────────────────────
# Клинически читаемые имена ключевых признаков (Graph_FEATURES).
_FEAT_RU: Dict[str, str] = {
    "Age":        "возраст",
    "attempt":    "попытка",
    "afc":        "АФЧ",
    "OCC":        "ОКК",
    "insem":      "MII",
    "two_pn":     "2PN",
    "Bl":         "бластоцисты",
    "Good_Bl":    "хорошие_Bl",
    "KPIScore":   "KPI",
    "fert_rate":  "частота_опл",
    "blast_rate": "частота_бласт",
}
# Признаки, выводимые в карточке соседа (по приоритету важности)
_KEY_FEATS = ["Age", "attempt", "afc", "OCC", "insem", "two_pn",
              "Bl", "Good_Bl", "KPIScore"]


def _extract_gat_neighbors(g: Dict[str, Any],
                            max_n: int = 7) -> Optional[Dict[str, Any]]:
    """Извлекает top-N ближайших соседей из results GAT-предсказания.

    Возвращает dict для вставки в LLM-контекст или None если данные недоступны.
    Работает как для Tier 0, так и для Tier 1.
    """
    gnn_r = g.get("_gnn_result") or {}
    nb    = gnn_r.get("neighbors")
    if not isinstance(nb, dict):
        return None

    sims      = nb.get("sims")
    probs     = nb.get("probs")
    neigh_raw = nb.get("neigh_raw")
    pat_raw   = nb.get("pat_raw")

    if sims is None or probs is None:
        return None

    # Имена признаков берём из загруженного бандла (приоритет) или
    # из feat_labels neighbors_data (только первые 5 граф-признаков)
    bundle   = g.get("_gnn_bundle") or {}
    feat_list: List[str] = bundle.get("features") or nb.get("feat_labels") or []

    try:
        import numpy as np
        sims_np  = np.asarray(sims,  dtype=float)
        probs_np = np.asarray(probs, dtype=float)

        n = min(max_n, len(sims_np))

        # ── Карточки соседей ──────────────────────────────────────────────
        neighbors_list = []
        for i in range(n):
            card: Dict[str, Any] = {
                "ранг":                i + 1,
                "косинусное_сходство": round(float(sims_np[i]), 3),
                "GNN_P_беременность_%": round(float(probs_np[i]) * 100, 1),
            }
            # Клинические признаки — из inverse-transformed матрицы
            if neigh_raw is not None and feat_list:
                row = np.asarray(neigh_raw)[i]
                feats: Dict[str, Any] = {}
                for fname in _KEY_FEATS:
                    if fname in feat_list:
                        idx = feat_list.index(fname)
                        v   = float(row[idx])
                        # Округление: целочисленные признаки — int
                        ru  = _FEAT_RU.get(fname, fname)
                        feats[ru] = (int(round(v))
                                     if fname in ("Age", "attempt", "afc",
                                                  "OCC", "insem", "two_pn",
                                                  "Bl", "Good_Bl")
                                     else round(v, 3))
                if feats:
                    card["признаки"] = feats
            neighbors_list.append(card)

        # ── Профиль пациентки для сравнения ──────────────────────────────
        patient_profile: Dict[str, Any] = {}
        if pat_raw is not None and feat_list:
            row = np.asarray(pat_raw)[0]
            for fname in _KEY_FEATS:
                if fname in feat_list:
                    idx = feat_list.index(fname)
                    v   = float(row[idx])
                    ru  = _FEAT_RU.get(fname, fname)
                    patient_profile[ru] = (
                        int(round(v))
                        if fname in ("Age", "attempt", "afc",
                                     "OCC", "insem", "two_pn",
                                     "Bl", "Good_Bl")
                        else round(v, 3)
                    )

        # ── Сводка ───────────────────────────────────────────────────────
        summary: Dict[str, Any] = {
            "число_соседей":            n,
            "косинусное_сходство_мин":  round(float(sims_np[:n].min()), 3),
            "косинусное_сходство_макс": round(float(sims_np[:n].max()), 3),
            "GNN_P_медиана_%":          round(float(np.median(probs_np[:n])) * 100, 1),
            "GNN_P_мин_%":              round(float(probs_np[:n].min()) * 100, 1),
            "GNN_P_макс_%":             round(float(probs_np[:n].max()) * 100, 1),
            "GNN_P_стд_%":              round(float(probs_np[:n].std())  * 100, 1),
        }

        return {
            "сводка":   summary,
            "пациентка_профиль": patient_profile or None,
            "соседи":   neighbors_list,
            "пояснение": (
                "Каждый сосед — реальный исторический цикл ЭКО из обучающей "
                "базы (~16К протоколов), наиболее похожий на текущий по "
                "клиническим признакам (косинусное сходство по Graf-пространству). "
                "GNN_P_беременность — прогноз модели для СОСЕДА (не пациентки). "
                "Профиль пациентки приведён для сравнения."
            ),
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════
#  PRE-COMPUTATION LAYER (Tier 0 — перед LLM)
#  Python-классификатор: все клинические категории вычисляются детерминированно
#  ДО обращения к LLM. LLM получает уже готовые метки и только пишет текст.
#
#  Принцип: Neural pipeline → deterministic clinical facts → LLM narration
#  LLM не переклассифицирует числа — она объясняет уже размеченную картину.
# ══════════════════════════════════════════════════════════════════════════

def build_interpretation_flags(g: Dict[str, Any]) -> Dict[str, Any]:
    """Детерминированная классификация всех клинических категорий.

    Не вызывает LLM. Результат — готовые метки для нарратора.
    Пороги отражают принятые клинические диапазоны ЭКО.
    """
    res  = g.get("res") or {}
    befe = g.get("_befe_res")

    def _f(v):
        try: return float(v) if v is not None else None
        except: return None

    flags: Dict[str, Any] = {}

    # ── Уровень прогноза беременности ─────────────────────────────────────
    p = _f(getattr(befe, "posterior", None)) if befe else None
    if p is not None:
        pp = p * 100
        if pp >= 55:
            flags["prognosis_level"] = "favorable"
            flags["prognosis_hint"]  = "выше среднего — ≈ каждый второй перенос"
        elif pp >= 40:
            flags["prognosis_level"] = "good"
            flags["prognosis_hint"]  = "умеренно благоприятный — ≈ каждый 2–3-й перенос"
        elif pp >= 25:
            flags["prognosis_level"] = "moderate"
            flags["prognosis_hint"]  = "умеренный — ≈ каждый 3–4-й перенос"
        else:
            flags["prognosis_level"] = "low"
            flags["prognosis_hint"]  = "ниже среднего — менее одного из четырёх"

    # ── Ширина ДИ BEFE (статистическая неопределённость) ─────────────────
    ci_lo = _f(getattr(befe, "ci_low",  None)) if befe else None
    ci_hi = _f(getattr(befe, "ci_high", None)) if befe else None
    if ci_lo is not None and ci_hi is not None:
        ci_w = round((ci_hi - ci_lo) * 100, 1)
        flags["CI_width_pp"] = ci_w
        if ci_w < 15:
            flags["CI_category"] = "low_statistical_uncertainty"
        elif ci_w < 30:
            flags["CI_category"] = "moderate_statistical_uncertainty"
        else:
            flags["CI_category"] = "high_statistical_uncertainty"

    # ── Разброс ансамбля (модельная неопределённость) ─────────────────────
    _mp = []
    for _v in [res.get("p_per_transfer"),
               (res.get("nn_prediction") or {}).get("base_prob_mean"),
               g.get("_p_gnn_ens")]:
        try:
            if _v is not None: _mp.append(float(_v))
        except: pass
    if len(_mp) >= 2:
        sp = round((max(_mp) - min(_mp)) * 100, 1)
        flags["ensemble_spread_pp"] = sp
        # [CALIB] Wider bands: a 3-estimator (L1/L3/L6) spread naturally exceeds
        # 10pp because the mechanistic prior differs from the neural heads, so the
        # old 10/20 cut-points read "disagreement" almost always. Raw spread_pp is
        # still reported unchanged; only the label boundary moves.
        if sp < 15:
            flags["agreement_category"] = "high_model_agreement"
        elif sp < 30:
            flags["agreement_category"] = "moderate_model_agreement"
        else:
            flags["agreement_category"] = "high_model_disagreement"

    # ── OOD-статус ────────────────────────────────────────────────────────
    if befe is not None:
        ood_c = getattr(befe, "ood_clinical",    False)
        ood_e = getattr(befe, "ood_embryology",  False)
        ood_f = getattr(befe, "ood_final",       False)
        if ood_f:
            flags["OOD_status"] = "out_of_distribution"
            flags["OOD_subspace"] = (
                "clinical+embryological" if (ood_c and ood_e)
                else "clinical" if ood_c else "embryological"
            )
        else:
            flags["OOD_status"] = "in_distribution"

    # ── Итоговая оценка надёжности (A / B / C) ────────────────────────────
    rel = getattr(befe, "reliability", None) if befe else None
    ood = flags.get("OOD_status") == "out_of_distribution"
    ood_sub = flags.get("OOD_subspace", "")
    ci_cat  = flags.get("CI_category", "")
    agr_cat = flags.get("agreement_category", "")
    # [CALIB] OOD no longer forces C unconditionally: a mild single-subspace OOD
    # with acceptable reliability is treated as caution (grade B), while dual-
    # subspace OOD or genuinely low reliability still yields C. Reliability cut-
    # points lowered (A: 70->60, B: 45->35) to match the recalibrated bands.
    if ood and (ood_sub == "clinical+embryological"
                or (rel is not None and rel < 35)):
        flags["confidence_grade"] = "C"
    elif (not ood
          and ci_cat == "low_statistical_uncertainty"
          and agr_cat == "high_model_agreement"
          and (rel is None or rel >= 60)):
        flags["confidence_grade"] = "A"
    elif (ci_cat != "high_statistical_uncertainty"
          and agr_cat != "high_model_disagreement"
          and (rel is None or rel >= 35)):
        flags["confidence_grade"] = "B"
    else:
        flags["confidence_grade"] = "C"

    # ── Риск OHSS ─────────────────────────────────────────────────────────
    # Пороги откалиброваны консервативно: LLM не должна переоценивать риски.
    # Тяжёлый: любой ≥ 10% — реальная клиническая угроза.
    # Умеренный: тяжёлый 5–9% ИЛИ умеренный ≥ 20% — наблюдение.
    # Низкий: стандартное ведение без специальных мер.
    ohss  = res.get("ohss") or {}
    p_sev = _f(ohss.get("p_severe_ohss"))
    p_mod = _f(ohss.get("p_moderate_ohss"))
    if p_sev is not None and p_sev >= 0.10:
        flags["OHSS_risk"]          = "high"
        flags["OHSS_management"]    = "имеет смысл обсудить freeze-all как приоритетную опцию"
    elif ((p_sev is not None and p_sev >= 0.05)
          or (p_mod is not None and p_mod >= 0.20)):
        flags["OHSS_risk"]          = "moderate"
        flags["OHSS_management"]    = "мониторинг после пункции; freeze-all стоит держать в голове"
    else:
        flags["OHSS_risk"]          = "low"
        flags["OHSS_management"]    = "стандартное наблюдение, специальных мер не требуется"

    # ── Риск пустого цикла ────────────────────────────────────────────────
    # ≥ 35%: существенный — влияет на консультирование.
    # 20–34%: умеренный — стоит обсудить с пациенткой.
    # < 20%: низкий — стандартная культивация.
    empty = res.get("empty") or {}
    p_nb  = _f(empty.get("p_no_blast"))
    if p_nb is not None:
        if p_nb >= 0.35:
            flags["empty_cycle_risk"]        = "high"
            flags["empty_cycle_management"]  = "обсудить с пациенткой заранее; может быть основанием для пересмотра тактики"
        elif p_nb >= 0.20:
            flags["empty_cycle_risk"]        = "moderate"
            flags["empty_cycle_management"]  = "стоит учитывать при консультировании"
        else:
            flags["empty_cycle_risk"]        = "low"
            flags["empty_cycle_management"]  = "стандартный прогноз эмбриологии"

    # ── Фенотип ответа ────────────────────────────────────────────────────
    ca    = g.get("ca") or res.get("cluster_analysis") or {}
    cl    = ca.get("dominant_cluster")
    probs = ca.get("cluster_probs") or {}
    if cl is not None:
        _labels = {0: "standard_responder",
                   1: "poor_responder",
                   2: "high_responder"}
        flags["response_phenotype"] = _labels.get(int(cl), "unknown")
        _conf = probs.get(int(cl))
        if _conf is not None:
            flags["phenotype_confidence_pct"] = round(float(_conf) * 100, 1)

    # ── Стратегия банкинга MII ────────────────────────────────────────────
    eb  = g.get("_eb")
    age = _f(g.get("age"))
    if isinstance(eb, dict):
        fwd   = eb.get("forward_at_median") or {}
        efp   = eb.get("euploid_for_preg")  or {}
        exp_e = _f(fwd.get("mean"))
        k50   = efp.get(0.50)
        try:
            import math as _math
            cyc50 = (_math.ceil(k50 / exp_e)
                     if k50 and exp_e and exp_e > 0 else None)
        except: cyc50 = None
        if cyc50 == 1 or (exp_e and k50 and exp_e >= k50):
            flags["banking_strategy"] = "immediate_transfer_feasible"
        elif cyc50 == 2:
            flags["banking_strategy"] = "two_cycles_recommended"
        elif cyc50 is not None and cyc50 >= 3:
            flags["banking_strategy"] = "extended_accumulation"
        else:
            flags["banking_strategy"] = "insufficient_data"
        # Срочность: возраст ≥ 38 повышает приоритет накопления
        if age is not None and age >= 38:
            flags["banking_age_urgency"] = "high"
        elif age is not None and age >= 35:
            flags["banking_age_urgency"] = "moderate"
        else:
            flags["banking_age_urgency"] = "low"

    # ── Управление циклом ─────────────────────────────────────────────────
    p_cancel = _f(res.get("p_cancel_risk"))
    if p_cancel is not None:
        if p_cancel >= 0.15:
            flags["cycle_management"] = "high_cancellation_risk"
        elif p_cancel >= 0.07:
            flags["cycle_management"] = "monitor_closely"
        else:
            flags["cycle_management"] = "standard_cycle_expected"

    return flags


def build_narrative_context(g: Dict[str, Any]) -> Dict[str, Any]:
    """Компактный пред-классифицированный контекст для нарратора Tier 0.

    Заменяет build_clinical_context() как вход к LLM.
    Ключевые свойства:
      • JSON в ~3× меньше build_clinical_context() → меньше токенов → быстрее
      • Все категории вычислены в Python ДО вызова LLM
      • LLM объясняет метки, а не переоценивает числа
    """
    res  = g.get("res") or {}
    befe = g.get("_befe_res")

    def _f(v):
        try: return float(v) if v is not None else None
        except: return None
    def _pv(v, nd=1):
        x = _f(v)
        return round(x * 100, nd) if x is not None else None

    flags = build_interpretation_flags(g)

    # ── Основа: пациент ───────────────────────────────────────────────────
    ctx: Dict[str, Any] = {
        "patient": {
            "age":        _f(g.get("age")),
            "amh_ng_ml":  _f(g.get("amh")),
            "afc":        _f(g.get("afc")),
            "attempt":    g.get("attempt_number"),
        }
    }

    # ── Главный прогноз с предклассифицированными метками ─────────────────
    if befe is not None:
        ctx["main_forecast"] = {
            "P_pct":         _pv(getattr(befe, "posterior", None)),
            "CI_low_pct":    _pv(getattr(befe, "ci_low",    None)),
            "CI_high_pct":   _pv(getattr(befe, "ci_high",   None)),
            "CI_width_pp":   flags.get("CI_width_pp"),
            # ↓ уже вычисленные категории — LLM только объясняет
            "level":         flags.get("prognosis_level"),
            "interpretation":flags.get("prognosis_hint"),
        }

    # ── Динамика цикла ────────────────────────────────────────────────────
    ctx["cycle"] = {
        "P_overall_pct":  _pv(res.get("p_overall_cycle")),
        "P_cancel_pct":   _pv(res.get("p_cancel_risk")),
        "P_viable_pct":   _pv(res.get("p_viable")),
        "management":     flags.get("cycle_management"),
    }

    # ── Фенотип ───────────────────────────────────────────────────────────
    ctx["response_phenotype"] = {
        "label":          flags.get("response_phenotype"),
        "confidence_pct": flags.get("phenotype_confidence_pct"),
    }

    # ── Риски с предклассифицированными уровнями ──────────────────────────
    ohss  = res.get("ohss")  or {}
    empty = res.get("empty") or {}
    ctx["risks"] = {
        "OHSS": {
            "level":         flags.get("OHSS_risk"),
            "management":    flags.get("OHSS_management"),   # pre-computed action hint
            "P_mod_pct":     _pv(ohss.get("p_moderate_ohss")),
            "P_sev_pct":     _pv(ohss.get("p_severe_ohss")),
        },
        "empty_cycle": {
            "level":         flags.get("empty_cycle_risk"),
            "management":    flags.get("empty_cycle_management"),
            "P_no_blast_pct": _pv(empty.get("p_no_blast")),
        },
    }

    # ── Банкинг MII ───────────────────────────────────────────────────────
    eb = g.get("_eb")
    if isinstance(eb, dict):
        fwd = eb.get("forward_at_median") or {}
        efp = eb.get("euploid_for_preg")  or {}
        mt  = eb.get("mii_table")         or {}
        k50, k70 = efp.get(0.50), efp.get(0.70)
        try:
            import math as _math
            exp_e = _f(fwd.get("mean"))
            c50   = (_math.ceil(k50 / exp_e) if k50 and exp_e and exp_e > 0 else None)
            c70   = (_math.ceil(k70 / exp_e) if k70 and exp_e and exp_e > 0 else None)
        except: c50 = c70 = None
        _mii_target = (mt.get(min(k50 or 2, max(eb.get("k_targets", [2])))) or {}).get(0.80)
        ctx["banking_mii"] = {
            "MII_this_cycle":         int(eb.get("patient_mii_median") or 0) or None,
            "euploids_expected":      round(exp_e, 1) if exp_e else None,
            "euploids_for_P50":       k50,
            "cycles_for_P50":         c50,
            "cycles_for_P70":         c70,
            "mii_target_80pct":       int(_mii_target) if _mii_target else None,
            "strategy":               flags.get("banking_strategy"),
            "age_urgency":            flags.get("banking_age_urgency"),
        }

    # ── Надёжность прогноза с итоговой оценкой ────────────────────────────
    ctx["prediction_confidence"] = {
        "grade":          flags.get("confidence_grade"),       # A / B / C
        "CI_category":    flags.get("CI_category"),
        "agreement":      flags.get("agreement_category"),
        "BEFE_reliability": getattr(befe, "reliability", None) if befe else None,
        "BEFE_band":      getattr(befe, "reliability_band", None) if befe else None,
        "OOD_status":     flags.get("OOD_status"),
        "OOD_subspace":   flags.get("OOD_subspace"),
    }

    # ── Исторические аналоги GAT (только сводка — не полный список) ───────
    nb_full = _extract_gat_neighbors(g, max_n=7)
    if nb_full:
        sv = nb_full.get("сводка") or {}
        top = (nb_full.get("соседи") or [{}])[0]
        ctx["historical_analogues"] = {
            "n":                  sv.get("число_соседей"),
            "similarity_range":   (f"{sv.get('косинусное_сходство_мин')}–"
                                   f"{sv.get('косинусное_сходство_макс')}"),
            "GNN_P_range_pct":    f"{sv.get('GNN_P_мин_%')}–{sv.get('GNN_P_макс_%')}",
            "GNN_P_median_pct":   sv.get("GNN_P_медиана_%"),
            "closest": {
                "similarity":     top.get("косинусное_сходство"),
                "GNN_P_pct":      top.get("GNN_P_беременность_%"),
                **(top.get("признаки") or {}),
            },
        }

    # [IMP STIM] Protocol/guideline grounding — deterministic, graceful fallback.
    # Adds dose nomogram + matched published recommendations so the narrator can
    # explain the L7/BEFE prognosis IN CONCORDANCE with guidelines, not in isolation.
    try:
        from protocol_guidance import build_protocol_guidance
        _pg = build_protocol_guidance(g)
        if _pg:
            ctx["protocol_guidance"] = _pg
    except Exception:
        pass  # feature optional; never break the existing narrator

    return ctx


# ──────────────────────────────────────────────────────────────────────────
#  СИСТЕМНЫЙ ПРОМПТ (Tier 0) — v3
#  Короткий (≈150 токенов): категории уже вычислены Python-кодом,
#  LLM только пишет связный клинический текст.
# ──────────────────────────────────────────────────────────────────────────
_SYSTEM_NARRATOR = """Ты — клинический консультант IVF Digital Twin. Читатель — лечащий врач.

Получаешь pre-classified JSON: все категории (уровень прогноза, риски, надёжность,
стратегия банкинга) уже вычислены Python-кодом. Твоя задача — написать связный
клинический нарратив, объясняющий эти категории врачу.

ПРАВИЛА (нарушение недопустимо):
1. Числа — ТОЛЬКО из JSON. Ничего не изобретать.
2. НЕ переклассифицировать. Если JSON говорит "moderate" — объясняй "moderate".
3. Не назначай лечение и не формулируй директивные рекомендации.
   Используй: «можно рассмотреть», «имеет смысл обсудить», «стоит учитывать»,
   «может быть основанием для». Не используй: «показан», «провести», «назначить».
4. Тон: коллега-консультант, активный залог, без канцеляризмов.
5. Отвечай на русском. Термины BEFE, MII, OHSS, ПГТ-А — свободно.

РАЗДЕЛЫ (пропускай при отсутствии данных):
### 1. Главный прогноз — объясни level и что он означает для этой пациентки
### 2. Сценарии цикла — развилки, риск отмены, что отличает хороший исход от плохого
### 3. Фенотип ответа — ожидания от стимуляции, нюансы протокола
### 4. Риски — для каждого риска используй management из JSON как ориентир формулировки
### 5. Банкинг MII — логика strategy, возрастная срочность, цепочка MII→эуплоиды
### 6. Исторические аналоги — что говорят похожие случаи из базы (если есть)
### 7. Надёжность прогноза — объясни confidence_grade (A/B/C) и оба источника
         неопределённости (CI_category + agreement) в клиническом смысле
### 8. Итог — 2–3 предложения: что главное для этого конкретного случая

### 9. Протокол стимуляции и гайдлайны (ТОЛЬКО если в JSON есть protocol_guidance)
         Свяжи фенотип ответа и риск СГЯ из основного прогноза с блоком
         protocol_guidance: какой протокол и какой ДИАПАЗОН стартовой дозы
         предлагает номограмма и почему. Дозу подавай только как ориентир
         («по номограмме — N–M МЕ»), никогда как назначение. Каждое клиническое
         утверждение по протоколу подкрепляй ссылкой из
         protocol_guidance.гайдлайны (поле citation). Числа дозы бери
         исключительно из этого блока. Если follitropin-дельта = null —
         не упоминай её."""


# ──────────────────────────────────────────────────────────────────────────
#  ИЗВЛЕЧЕНИЕ КОНТЕКСТА ИЗ globals app.py
#  (те же обращения, что в patient_brief.py — числа совпадают один в один)
# ──────────────────────────────────────────────────────────────────────────
def _pct(x: Optional[float], nd: int = 1) -> Optional[float]:
    """Доля 0..1 → процент, округлённый. None → None."""
    if x is None:
        return None
    try:
        return round(float(x) * 100, nd)
    except (TypeError, ValueError):
        return None


# Единый источник заключения CSDI — переиспользуем логику patient_brief,
# чтобы не было расхождения значений. Если импорт не удался — мягкий фолбэк.
try:
    from patient_brief import _csdi_conclusion as _pb_csdi  # type: ignore
except Exception:  # pragma: no cover
    _pb_csdi = None


def build_clinical_context(g: Dict[str, Any],
                           include_trp: bool = False) -> Dict[str, Any]:
    """Собирает структурированный контекст из окружения app.py для нарратора (Tier 0).

    Каждое поле извлекается защищённо: отсутствие любого значения не ломает
    сборку, а просто не попадает в контекст.

    include_trp: устарело, игнорируется. TRP убран из нарратора —
    это независимый стратегический модуль, не связанный с конкретным циклом.
    """
    res: Dict[str, Any] = g.get("res") or {}
    ctx: Dict[str, Any] = {}

    # ── Параметры пациента ────────────────────────────────────────────────
    def _num(key):
        v = g.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    ctx["пациент"] = {
        "возраст":        _num("age"),
        "AMH_нг_мл":      _num("amh"),
        "AFC":            _num("afc"),
        "BMI":            _num("bmi"),
        "номер_попытки":  g.get("attempt_number"),
    }

    # ── BEFE (Layer 7) — главная вероятность ──────────────────────────────
    befe = g.get("_befe_res")
    if befe is not None and getattr(befe, "posterior", None) is not None:
        ctx["BEFE_layer7"] = {
            "вероятность_беременности_проц": _pct(befe.posterior),
            "диапазон_95_проц": [_pct(getattr(befe, "ci_low", None)),
                                  _pct(getattr(befe, "ci_high", None))],
            "пояснение": "Главный итоговый прогноз на перенос (байесовское "
                         "слияние доказательств L3/L6).",
        }

    # ── Шансы по циклу ────────────────────────────────────────────────────
    ctx["цикл"] = {
        "успех_всего_цикла_проц":          _pct(res.get("p_overall_cycle")),
        "суммарный_шанс_при_переносе_всех_проц": _pct(res.get("p_cum_if_viable")),
        "вероятность_на_перенос_проц":     _pct(res.get("p_per_transfer")),
        "риск_отмены_цикла_проц":          _pct(res.get("p_cancel_risk")),
    }

    # ── CSDI (Layer 5) ────────────────────────────────────────────────────
    if _pb_csdi is not None:
        try:
            csdi = _pb_csdi(g)
        except Exception:
            csdi = None
        if csdi is not None:
            ctx["CSDI_layer5"] = {
                "оценка_проц": _pct(csdi.get("p")),
                "заключение":  ("благоприятное" if csdi.get("favorable")
                                else "осторожное"),
                "диапазон_проц": [_pct(csdi.get("ci_low")),
                                   _pct(csdi.get("ci_high"))],
                "пояснение": "Независимая проверка прогноза эмбриологического "
                             "этапа (диффузионная модель + LightGBM).",
            }

    # ── GAT (Layer 6) ─────────────────────────────────────────────────────
    p_gnn = g.get("_p_gnn_ens")
    if p_gnn is not None:
        ctx["GAT_layer6"] = {
            "вероятность_успеха_переноса_проц": _pct(p_gnn),
            "пояснение": "Граф клинических соседей (ансамбль на ~16K циклов).",
        }
        nb_ctx = _extract_gat_neighbors(g)
        if nb_ctx is not None:
            ctx["GAT_layer6"]["kNN_соседи"] = nb_ctx

    # ── Неопределённость ансамбля (количественная) ───────────────────────
    # Ключевая фишка Digital Twin: неопределённость рассчитана, а не декларирована.
    if befe is not None and getattr(befe, "posterior", None) is not None:
        _ci_lo = getattr(befe, "ci_low",  None)
        _ci_hi = getattr(befe, "ci_high", None)
        _ci_w  = (round((_ci_hi - _ci_lo) * 100, 1)
                  if _ci_lo is not None and _ci_hi is not None else None)
        # Разброс ансамбля: собираем доступные оценки (L1, L3, L6)
        _mp: List[float] = []
        _p_l1 = res.get("p_per_transfer")
        _nn_b  = (res.get("nn_prediction") or {}).get("base_prob_mean")
        _p_gnn = g.get("_p_gnn_ens")
        for _v in [_p_l1, _nn_b, _p_gnn]:
            try:
                if _v is not None:
                    _mp.append(float(_v))
            except (TypeError, ValueError):
                pass
        _spread = (round((max(_mp) - min(_mp)) * 100, 1)
                   if len(_mp) >= 2 else None)
        ctx["неопределённость"] = {
            "CI_BEFE_ширина_пп":          _ci_w,
            "разброс_ансамбля_пп":        _spread,
            "надёжность_BEFE_0_100":      getattr(befe, "reliability",     None),
            "полоса_надёжности":          getattr(befe, "reliability_band", None),
            "prior_pull_проц":            _pct(getattr(befe, "prior_pull",    None)),
            "evidence_pull_проц":         _pct(getattr(befe, "evidence_pull", None)),
            "OOD_клинический":            getattr(befe, "ood_clinical",  None),
            "OOD_эмбриологический":       getattr(befe, "ood_embryology", None),
            "пояснение": (
                "CI_BEFE_ширина_пп — ширина 95% ДИ в процентных пунктах: "
                "<15 узкий (высокая точность), 15–30 умеренный, >30 широкий. "
                "разброс_ансамбля_пп — max-min между L1/L3/L6 оценщиками. "
                "Эти два числа вместе дают количественную меру неопределённости."
            ),
        }

    # ── Банкинг (накопление MII) ──────────────────────────────────────────
    eb = g.get("_eb")
    if isinstance(eb, dict):
        try:
            import math as _math
            p_mii   = eb.get("p_per_mii")
            fwd     = eb.get("forward_at_median") or {}
            efp     = eb.get("euploid_for_preg")  or {}
            mt      = eb.get("mii_table")         or {}
            kmax    = max(eb.get("k_targets", [5]))
            mii_med = eb.get("patient_mii_median")

            exp_eu  = fwd.get("mean")   # ожидаемых эуплоидов из медианного MII этого цикла

            def _cycles(k_need):
                if not k_need or not exp_eu or exp_eu <= 0:
                    return None
                return _math.ceil(k_need / exp_eu)

            k50, k70, k90 = efp.get(0.50), efp.get(0.70), efp.get(0.90)

            # Обратная таблица: MII нужно накопить для k эуплоидных @ 80%
            mii_for = {}
            for _k in [1, 2, 3]:
                _entry = (mt.get(min(_k, kmax)) or {}).get(0.80)
                mii_for[f"{_k}_эупл_80%"] = (
                    int(_entry) if _entry is not None else "более 200"
                )

            ctx["банкинг"] = {
                # ── Ключевые параметры модели Esteves ─────────────────────
                "P_эуплоид_на_MII_проц":        _pct(p_mii),
                "цепочка": "фертилизация × бластуляция × эуплоидия по возрасту",
                # ── Текущий цикл ──────────────────────────────────────────
                "MII_ожидается_этот_цикл":      (int(mii_med) if mii_med else None),
                "эуплоидных_бластоцист_из_этих_MII": (
                    round(exp_eu, 1) if exp_eu is not None else None
                ),
                # ── Сколько эуплоидных бластоцист нужно для P-беременности
                # (из накопленного пула; каждый перенос независим)
                "эуплоидных_эмбрионов_для_P50": k50,
                "эуплоидных_эмбрионов_для_P70": k70,
                "эуплоидных_эмбрионов_для_P90": k90,
                # ── Сколько циклов стимуляции нужно накопить MII ─────────
                # (циклов с медианным ответом этого пациента)
                "циклов_стимуляции_для_P50":    _cycles(k50),
                "циклов_стимуляции_для_P70":    _cycles(k70),
                # ── Обратная таблица: суммарно MII нужно собрать ─────────
                # чтобы получить k эуплоидных бластоцист с P=80%
                "суммарно_MII_для_k_эуплоидных_80%": mii_for,
                "пояснение": (
                    "БАНКИНГ — накопление MII ООЦИТОВ по нескольким циклам "
                    "стимуляции. Из накопленных MII после оплодотворения, "
                    "культивирования и ПГТ-А получают эуплоидные бластоцисты. "
                    "'суммарно_MII_для_k_эуплоидных_80%' — сколько MII нужно "
                    "собрать СУММАРНО по всем циклам, чтобы с 80% уверенностью "
                    "иметь k эуплоидных эмбрионов для переноса. "
                    "'циклов_стимуляции_для_P50/70' — сколько стимуляций с "
                    "медианным ответом ЭТОГО пациента даст достаточно MII."
                ),
            }
        except Exception:
            pass

    # ── Тип ответа на стимуляцию (кластер L4) ────────────────────────────
    ca = g.get("ca") or res.get("cluster_analysis")
    if ca is not None:
        try:
            probs = ca.get("cluster_probs", {}) or {}
            if probs:
                top = max(probs, key=probs.get)
                interp = g.get("CLUSTER_INTERPRETATIONS") or {}
                name_map = {0: "Standard Responder", 1: "Poor Responder",
                            2: "High Responder"}
                eng = interp.get(top, {}).get("name", name_map.get(top, f"C{top}"))
                ru = {"Standard Responder": "стандартный (типичный) ответ",
                      "Poor Responder":     "сниженный (слабый) ответ",
                      "High Responder":     "высокий ответ"}.get(eng, eng)
                ctx["тип_ответа_кластер"] = {
                    "тип": ru,
                    "вероятность_отнесения_проц": _pct(float(probs[top])),
                }
        except Exception:
            pass

    # ── Риски (применимость прогноза на перенос) ─────────────────────────
    empty = res.get("empty") or {}
    p_no_blast = empty.get("p_no_blast")
    if p_no_blast is not None:
        ctx["риски"] = {
            "нет_ни_одной_бластоцисты_проц":       _pct(p_no_blast),
            "нет_хороших_бластоцист_проц":         _pct(empty.get("p_no_good_blast")),
            "OHSS_умеренный_проц":                 _pct(
                (res.get("ohss") or {}).get("p_moderate_ohss")),
            "OHSS_тяжёлый_проц":                   _pct(
                (res.get("ohss") or {}).get("p_severe_ohss")),
        }

    return ctx


# ──────────────────────────────────────────────────────────────────────────
#  НИЗКОУРОВНЕВЫЙ ВЫЗОВ OLLAMA
# ──────────────────────────────────────────────────────────────────────────
# Отдельная сессия, игнорирующая HTTP_PROXY/HTTPS_PROXY/.netrc из окружения:
# запрос к 127.0.0.1 должен идти НАПРЯМУЮ, а не через корпоративный прокси.
_SESSION = requests.Session()
_SESSION.trust_env = False
_NO_PROXY = {"http": None, "https": None}


class OllamaError(RuntimeError):
    """Сервер ответил, но с ошибкой (неверная модель, 4xx/5xx и т.п.)."""


def health_check() -> bool:
    """True, если Ollama отвечает на loopback."""
    try:
        r = _SESSION.get(f"{OLLAMA_HOST}/api/tags", timeout=(3, 5),
                         proxies=_NO_PROXY)
        return r.status_code == 200
    except requests.RequestException:
        return False


def list_models() -> List[str]:
    """Возвращает список моделей, загруженных в локальный Ollama.

    Пример: ['medgemma1.5', 'gemma4:27b', 'gemma3:12b', ...]
    Пустой список если Ollama недоступна.
    """
    try:
        r = _SESSION.get(f"{OLLAMA_HOST}/api/tags", timeout=(3, 8),
                         proxies=_NO_PROXY)
        r.raise_for_status()
        return sorted(m.get("name", "") for m in r.json().get("models", []))
    except requests.RequestException:
        return []


def warmup(model: str = MEDGEMMA) -> tuple[bool, str]:
    """Прогревает (загружает в память) модель один раз, с большим таймаутом.

    Вызывать при старте app.py (под спиннером). После прогрева модель держится
    в памяти keep_alive времени, и обычные consult() отвечают быстро.

    Возвращает (ok, message). При неудаче message содержит причину.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "ok"}],
        "stream": False,
        "keep_alive": _KEEP_ALIVE,
        "think": False,
        "options": {"num_predict": 1},   # сгенерировать 1 токен — нужна только загрузка
    }
    try:
        r = _SESSION.post(f"{OLLAMA_HOST}/api/chat", json=body,
                          timeout=_WARMUP_TIMEOUT, proxies=_NO_PROXY)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:300]}"
        return True, f"Модель «{model}» загружена и готова."
    except requests.exceptions.ConnectionError:
        return False, _OFFLINE_MSG
    except requests.exceptions.Timeout:
        return False, (f"Модель «{model}» не успела загрузиться за "
                       f"{_WARMUP_TIMEOUT[1]} c — вероятно, не хватает ресурсов "
                       f"(CPU/RAM). Проверьте `ollama run {model}` и `ollama ps`.")
    except requests.RequestException as exc:
        return False, f"Ошибка прогрева: {exc}"


def _chat(messages: List[Dict[str, Any]],
          model: str,
          tools: Optional[List[Dict]] = None,
          stream: bool = False,
          temperature: float = _TEMPERATURE,
          num_predict: Optional[int] = None,
          think: Optional[bool] = None):
    """Вызов /api/chat. При stream=False возвращает dict message;
    при stream=True — генератор кусков текста."""
    opts: Dict[str, Any] = {"temperature": temperature}
    if num_predict is not None:
        opts["num_predict"] = num_predict
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "keep_alive": _KEEP_ALIVE,
        "options": opts,
    }
    if think is not None:
        body["think"] = think
    if tools:
        body["tools"] = tools

    if not stream:
        r = _SESSION.post(f"{OLLAMA_HOST}/api/chat", json=body,
                          timeout=_TIMEOUT, proxies=_NO_PROXY)
        if r.status_code != 200:
            raise OllamaError(f"HTTP {r.status_code}: {r.text[:500]}")
        return r.json().get("message", {})

    def _gen() -> Iterator[str]:
        with _SESSION.post(f"{OLLAMA_HOST}/api/chat", json=body,
                           timeout=_TIMEOUT, stream=True,
                           proxies=_NO_PROXY) as r:
            if r.status_code != 200:
                raise OllamaError(f"HTTP {r.status_code}: {r.text[:500]}")
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break
    return _gen()


_OFFLINE_MSG = ("Локальная LLM недоступна (Ollama не запущена на "
                f"{OLLAMA_HOST}). Запустите `ollama serve` и убедитесь, что "
                "модель загружена (`ollama list`).")


# ──────────────────────────────────────────────────────────────────────────
#  TIER 0 — НАРРАТОР
# ──────────────────────────────────────────────────────────────────────────
_DEFAULT_QUESTION = (
    "Составь развёрнутое клиническое резюме по всем разделам: главный прогноз "
    "с объяснением, сценарии развития цикла, фенотип ответа, клиническая картина "
    "рисков, логика и стратегия банкинга, исторические аналоги GAT, природа "
    "неопределённости ансамбля, итоговые клинические соображения."
)

_QUESTION_CONCISE = (
    "Краткое клиническое резюме для врача: прогноз с объяснением, ключевой "
    "сценарий цикла, главный риск и как его распознать, логика банкинга одной "
    "фразой, неопределённость ансамбля количественно."
)

# Подсказки по длине — добавляются к запросу перед отправкой.
_STYLE_HINT = {
    "full":      "",
    "narrative": ("\n\nПиши развёрнуто — каждый раздел полным абзацем с объяснением "
                  "механизмов и сценариев. Объём: ~400–550 слов суммарно, "
                  "не больше — только суть."),
    "concise":   ("\n\nФОРМАТ: компактно, до ~200 слов. По 1–2 предложения на раздел, "
                  "без повторов и воды."),
}

# num_predict — потолок токенов на ВЫХОДЕ модели.
# Важно: если think=False игнорируется (старый Ollama), токены рассуждений
# тоже входят в этот бюджет. Запас ~800 токенов на возможные рассуждения.
# Переопределяется через DT_LLM_NP_NARRATIVE / DT_LLM_NP_CONCISE.
_STYLE_NUM_PREDICT = {
    "full":      int(os.environ.get("DT_LLM_NP_NARRATIVE", 2500)),
    "narrative": int(os.environ.get("DT_LLM_NP_NARRATIVE", 2500)),
    "concise":   int(os.environ.get("DT_LLM_NP_CONCISE",   1200)),
}


_THINK_BLOCK = [
    re.compile(r"<unused94>.*?<unused95>", re.DOTALL),  # формат рассуждения Gemma/MedGemma
    re.compile(r"<think>.*?</think>", re.DOTALL),
]


def _strip_thinking(text: str) -> str:
    """Убирает остаточные блоки рассуждения, если они просочились в content."""
    for pat in _THINK_BLOCK:
        text = pat.sub("", text)
    if "<unused95>" in text:                       # есть закрывающий — берём хвост
        text = text.rsplit("<unused95>", 1)[-1]
    elif text.lstrip().startswith(("<unused94>", "<think>")):  # незакрытый блок — ответа нет
        s = text.lstrip()
        for t in ("<unused94>", "<think>"):
            if s.startswith(t):
                text = s[len(t):]
                break
    text = re.sub(r"^\s*thought\b", "", text)
    return text.strip()


_OPEN_TAGS  = ("<unused94>", "<think>")
_CLOSE_TAGS = ("<unused95>", "</think>")


def _filter_thinking_stream(chunks: Iterator[str]) -> Iterator[str]:
    """Поточно подавляет блок рассуждения: на экран идёт только сам ответ.

    Пока не ясно, есть ли блок рассуждения, копит буфер. Если ответ начинается
    с открывающего тега — ждёт закрывающий и пускает всё, что после него.
    Если рассуждения нет — сразу переходит в прозрачный режим.

    Защита от обрыва потока внутри think-блока: если поток завершился до
    закрывающего тега (num_predict исчерпан до конца рассуждения), блок
    подавляется полностью — наружу идёт только сигнальное сообщение.
    """
    buf = ""
    passthrough = False
    in_think = None  # None=ещё не решили, True=внутри рассуждения, False=рассуждения нет
    for ch in chunks:
        if passthrough:
            yield ch
            continue
        buf += ch
        if in_think is None:
            s = buf.lstrip()
            if s == "" or any(t.startswith(s) and s != t for t in _OPEN_TAGS):
                continue  # буфер ещё может оказаться началом тега — ждём
            if s.startswith(_OPEN_TAGS):
                in_think = True
                buf = ""   # сбрасываем открывающий тег — он нам не нужен
            else:
                in_think = False
                passthrough = True
                yield buf
                buf = ""
                continue
        if in_think:
            # Ищем закрывающий тег — но не в огромном буфере целиком,
            # а только в хвосте (последние 20 символов + новый кусок),
            # чтобы не держать всё рассуждение в памяти.
            tail = buf[-20:]
            for close in _CLOSE_TAGS:
                if close in tail:
                    after = buf.split(close, 1)[1]
                    passthrough = True
                    buf = ""
                    if after:
                        yield after
                    break
            else:
                # Не копим весь think-буфер — сбрасываем, держим только хвост
                # для поиска закрывающего тега через границу чанков.
                if len(buf) > 40:
                    buf = buf[-40:]

    # ── Хвост: поток завершился ───────────────────────────────────────────
    if passthrough:
        return  # нормальный выход — ответ уже был передан

    if in_think:
        # Поток оборвался ВНУТРИ блока рассуждений (num_predict исчерпан
        # до закрывающего тега). Не выводим сырое рассуждение наружу.
        yield ("\n\n⚠ Генерация прервана внутри блока рассуждений модели "
               "(num_predict исчерпан). Увеличьте DT_LLM_NP_NARRATIVE или "
               "используйте style='concise'.")
        return

    # Поток завершился без think-блока — отдаём буфер как есть
    if buf:
        cleaned = _strip_thinking(buf)
        if cleaned:
            yield cleaned


def _build_messages(ctx: Dict[str, Any], question: str) -> List[Dict[str, str]]:
    """Собирает список сообщений для /api/chat.

    Префикс /no_think подавляет блок рассуждений у Gemma-семейства на уровне
    модели (работает независимо от версии Ollama и поддержки параметра think).
    """
    ctx_json = json.dumps(ctx, ensure_ascii=False, indent=2)
    user = (
        "/no_think\n"                         # Gemma-level thinking suppression
        "Контекст пациента (использовать ТОЛЬКО эти числа):\n"
        f"```json\n{ctx_json}\n```\n\nЗапрос: {question}"
    )
    return [{"role": "system", "content": _SYSTEM_NARRATOR},
            {"role": "user",   "content": user}]


def consult(g: Dict[str, Any],
            question: Optional[str] = None,
            model: str = MEDGEMMA,
            include_trp: bool = True,
            style: str = "full",
            num_predict: Optional[int] = None,
            audit: bool = True) -> str:
    """Tier 0: вернуть готовый текст резюме (блокирующий вызов).

    style : "full" — развёрнуто по разделам; "concise" — компактно (быстрее на CPU).
    """
    if question is None:
        question = (_QUESTION_CONCISE if style == "concise" else _DEFAULT_QUESTION)
    question += _STYLE_HINT.get(style, "")
    if num_predict is None:
        num_predict = _STYLE_NUM_PREDICT.get(style)
    ctx = build_narrative_context(g)  # pre-classified compact context
    # [IMP] narrator cache (CPU win)
    _cache = _ck = None
    try:
        import llm_cache as _cache
        _ck = _cache.make_key("narrator_v1", ctx, question, model, style)
        _hit = _cache.get(_ck)
        if _hit:
            return _hit
    except Exception:
        _cache = _ck = None
    try:
        msg = _chat(_build_messages(ctx, question), model=model,
                    stream=False, temperature=_NARRATOR_TEMP,
                    num_predict=num_predict, think=_NARRATOR_THINK)
        text = _strip_thinking(msg.get("content", "").strip())
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return _OFFLINE_MSG
    except (requests.RequestException, OllamaError) as exc:
        return f"Ошибка обращения к модели «{model}»: {exc}"
    if _cache is not None and _ck is not None and text and text != _OFFLINE_MSG:
        _cache.set(_ck, text)
    if audit:
        _fr = None
        try:
            import faithfulness as _fc
            _fr = _fc.check_faithfulness(text, ctx)
        except Exception:
            pass
        _audit_log(ctx, question, text, model, faithfulness=_fr)
    return text


def consult_stream(g: Dict[str, Any],
                   question: Optional[str] = None,
                   model: str = MEDGEMMA,
                   include_trp: bool = True,
                   style: str = "full",
                   num_predict: Optional[int] = None) -> Iterator[str]:
    """Tier 0: потоковый генератор (для st.write_stream)."""
    if question is None:
        question = (_QUESTION_CONCISE if style == "concise" else _DEFAULT_QUESTION)
    question += _STYLE_HINT.get(style, "")
    if num_predict is None:
        num_predict = _STYLE_NUM_PREDICT.get(style)
    ctx = build_narrative_context(g)  # pre-classified compact context

    # [IMP] narrator cache (CPU win): identical context → return stored text.
    try:
        import llm_cache as _cache
        _ck = _cache.make_key("narrator_v1", ctx, question, model, style)
    except Exception:
        _cache, _ck = None, None
    if _cache is not None and _ck is not None:
        _hit = _cache.get(_ck)
        if _hit:
            yield _hit
            return

    collected: List[str] = []
    try:
        raw = _chat(_build_messages(ctx, question), model=model,
                    stream=True, temperature=_NARRATOR_TEMP,
                    num_predict=num_predict, think=_NARRATOR_THINK)
        for piece in _filter_thinking_stream(raw):
            collected.append(piece)
            yield piece
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        yield _OFFLINE_MSG
        return
    except (requests.RequestException, OllamaError) as exc:
        yield f"Ошибка обращения к модели «{model}»: {exc}"
        return
    _full = "".join(collected)
    # [IMP] store in cache (skip error/offline placeholders)
    if (_cache is not None and _ck is not None and _full
            and _full != _OFFLINE_MSG and not _full.startswith("Ошибка")):
        _cache.set(_ck, _full)
    # [IMP] grounding faithfulness self-check
    _fr = None
    try:
        import faithfulness as _fc
        _fr = _fc.check_faithfulness(_full, ctx)
    except Exception:
        pass
    _audit_log(ctx, question, _full, model, faithfulness=_fr)


# ──────────────────────────────────────────────────────────────────────────
#  АУДИТ (расширение существующего analytics-пайплайна)
# ──────────────────────────────────────────────────────────────────────────
def _audit_log(ctx: Dict[str, Any], question: str, answer: str,
               model: str, faithfulness: Optional[Dict[str, Any]] = None) -> None:
    """Пишет JSONL-запись: вход LLM + выход рядом с прогнозом. Не падает."""
    try:
        os.makedirs(_AUDIT_DIR, exist_ok=True)
        import datetime as _dt
        rec = {
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "context": ctx,
            "question": question,
            "answer": answer,
        }
        if faithfulness is not None:
            rec["faithfulness"] = faithfulness   # [IMP] grounding self-check
        with open(_AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass  # аудит не должен влиять на консультацию


# ══════════════════════════════════════════════════════════════════════════
#  TIER 1 — ENSEMBLE ANALYST
#  Получает СЫРЫЕ выходы всех 7 слоёв Digital Twin (те же данные, что идут
#  в аналитический CSV dt_analytics_ready_*.csv) и выдаёт:
#    • матрицу согласованности ансамбля
#    • оценку разброса и биологической правдоподобности
#    • якорный анализ CSDI vs нейросетевых оценщиков
#    • наиболее обоснованный диапазон вероятностей с ранжированной неопределённостью
#
#  В отличие от Tier 0 (нарратор BEFE-резюме), Tier 1 — количественный
#  аналитик РАСХОЖДЕНИЙ между моделями. Его ценность — там, где Tier 0
#  молчит: объективная картина того, насколько согласован ансамбль.
# ══════════════════════════════════════════════════════════════════════════

_SYSTEM_ANALYST = """Ты — аналитик нейросетевого ансамбля IVF Digital Twin.
Читатель — разработчик или опытный клиницист. Ты НЕ нарратор: не пересказываешь
итоговый вывод L7. Ты количественный аналитик, который интерпретирует РАСХОЖДЕНИЯ
между слоями и выбирает наиболее обоснованный прогноз.

СТРОГИЕ ПРАВИЛА:
1. Используй ТОЛЬКО числа из предоставленного JSON. Ничего не изобретать.
2. Недоступные модели ("доступна": false или null) → «н/д» в матрице.
3. Тон — аналитический, краткий. Никаких клинических нарративов и рекомендаций пациенту.
4. Согласованность: разброс P% < 10 пп = высокая, 10–20 пп = средняя, > 20 пп = низкая.
5. L5 CSDI = биологический якорь (диффузионная симуляция эмбриологии).
   Расхождение CSDI vs нейросетевых (L3/L6) > 10 пп = значимый конфликт, объяснить.
6. Ширина CI: < 15 пп = узкий (высокая точность), 15–30 пп = умеренный, > 30 пп = широкий.
7. Если P_нет_бластоцист > 30% — прогноз на перенос условно применим; отметить.
8. OOD_клинический → снижает доверие к нейросетевым оценкам L3/L6.
   OOD_эмбриологический → снижает вес CSDI в слиянии (L7 BEFE).
9. Отвечать на русском.
10. Не делать клинических назначений и финальных терапевтических рекомендаций.
    Итог Tier 1 — только о согласованности моделей, диапазоне вероятности
    и источниках неопределённости. Клиническое решение — за врачом.

ФОРМАТ — строго 5 блоков, без введения и послесловия:

[МАТРИЦА АНСАМБЛЯ]
Таблица строк: L1 MC-prior | L3 KAT-FT | L3 KAN | L5 CSDI | L6 GAT-ens | L7 BEFE
Колонки: Слой, P%, CI (низ–верх), Ширина CI, Статус

[СОГЛАСОВАННОСТЬ]
1–2 предложения: диапазон P% по ансамблю, оценка согласованности, выброс если есть.

[ИСТОРИЧЕСКИЕ АНАЛОГИ GAT]
Если kNN_соседи есть — 2–3 предложения: сколько реальных похожих циклов из базы,
диапазон косинусного сходства, диапазон их GNN-прогнозов. Перечисли 2–3 ближайших
аналога с числовым сходством и ключевыми признаками (возраст, АФЧ, ОКК, MII,
бластоцисты). Если kNN недоступен — «GAT соседи не извлечены».

[ЯКОРЬ CSDI]
1–2 предложения: согласован ли L5 с нейросетевыми (L3/L6)? Направление и величина
расхождения. Если CSDI недоступна — явно отметить.

[ВЫВОД И НЕОПРЕДЕЛЁННОСТЬ]
— Наиболее обоснованный диапазон P% (из наиболее согласованных / лучше
  откалиброванных моделей) с кратким обоснованием.
— Ранг неопределённости: ВЫСОКАЯ / СРЕДНЯЯ / НИЗКАЯ (и почему — 1 предложение).
— Если OOD или риск отсутствия бластоцист влияет на применимость — 1 фраза."""


def build_ensemble_context(g: Dict[str, Any]) -> Dict[str, Any]:
    """Извлекает сырые выходы всех слоёв Digital Twin из globals app.py.

    Цель — минимальная обработка для аналитика ансамбля, не нарратора.
    Структура зеркалит аналитический CSV (dt_analytics_ready_*.csv), но
    снабжена метаданными ролей для LLM-интерпретации.
    """
    res: Dict[str, Any] = g.get("res") or {}

    def _f(v) -> Optional[float]:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _pv(v, nd: int = 1) -> Optional[float]:
        x = _f(v)
        return round(x * 100, nd) if x is not None else None

    # ── Пациент (для проверки биологической правдоподобности) ─────────────
    ctx: Dict[str, Any] = {
        "пациент": {
            "возраст":       _f(g.get("age")),
            "AMH_нг_мл":     _f(g.get("amh")),
            "AFC":           _f(g.get("afc")),
            "BMI":           _f(g.get("bmi")),
            "номер_попытки": g.get("attempt_number"),
        }
    }

    # ── L1/L2: Monte Carlo + FORTUNE/KPI (механистический приор) ─────────
    rate_ci = res.get("rate_ci") or (None, None)
    ctx["L1_MC_prior"] = {
        "P_per_transfer_%":   _pv(res.get("p_per_transfer")),
        "P_overall_cycle_%":  _pv(res.get("p_overall_cycle")),
        "P_cancel_risk_%":    _pv(res.get("p_cancel_risk")),
        "клиника_CI_low_%":   _pv(rate_ci[0] if len(rate_ci) > 0 else None),
        "клиника_CI_high_%":  _pv(rate_ci[1] if len(rate_ci) > 1 else None),
        "med_blasts":         _f(res.get("blasts_med")),
        "med_good_blasts":    _f(res.get("good_med")),
        "роль": ("Механистический приор (MC + FORTUNE/KPI воронка). "
                 "L2=mean(sim_p_combined)≡p_per_transfer — не самостоятельный эксперт BEFE."),
    }

    # ── L3: KAT нейросетевой ансамбль (FT-Transformer + KAN) ─────────────
    _nn   = res.get("nn_prediction") or {}
    _nvsa = res.get("nn_nvsa") or {}
    ci_kat  = _nn.get("base_prob_ci")  or (None, None)
    ci_nvsa = _nvsa.get("adjusted_ci") or (None, None)
    p_kat  = _f(g.get("_p_kat_raw") or _nn.get("base_prob_mean"))
    p_nvsa = _f(g.get("_p_nvsa")    or _nvsa.get("adjusted_mean"))
    ctx["L3_KAT"] = {
        "P_FT_Transformer_%": _pv(p_kat),
        "P_KAN_%":            _pv(p_nvsa),
        "CI_FT_low_%":        _pv(ci_kat[0])  if ci_kat[0]  is not None else None,
        "CI_FT_high_%":       _pv(ci_kat[1])  if ci_kat[1]  is not None else None,
        "CI_KAN_low_%":       _pv(ci_nvsa[0]) if ci_nvsa[0] is not None else None,
        "CI_KAN_high_%":      _pv(ci_nvsa[1]) if ci_nvsa[1] is not None else None,
        "τ_база":             2.4,
        "доступна":           p_kat is not None,
        "роль": ("Лучшая калибровка в ансамбле (τ=2.4). "
                 "FT-Transformer — empirical evidence для BEFE L7. "
                 "KAN — независимый второй оценщик в L3."),
    }

    # ── L4: Кластер (фенотип ответа — модулирует надёжность BEFE) ─────────
    ca    = g.get("ca") or res.get("cluster_analysis") or {}
    probs = ca.get("cluster_probs") or {}
    ctx["L4_кластер"] = {
        "доминирующий":    ca.get("dominant_cluster"),
        "P_C0_Standard_%": _pv(probs.get(0)),
        "P_C1_Poor_%":     _pv(probs.get(1)),
        "P_C2_High_%":     _pv(probs.get(2)),
        "пояснение": "C0=Standard, C1=Poor Responder, C2=High Responder. Влияет на надёжность L7.",
    }

    # ── L5: CSDI Hybrid v3 (диффузионная симуляция) ───────────────────────
    # Используем ту же логику извлечения, что и в Tier 0 / patient_brief.
    _csdi_p = _csdi_ci_l = _csdi_ci_h = None
    if _pb_csdi is not None:
        try:
            _csdi_d = _pb_csdi(g)
            if _csdi_d:
                _csdi_p    = _f(_csdi_d.get("p"))
                _csdi_ci_l = _f(_csdi_d.get("ci_low"))
                _csdi_ci_h = _f(_csdi_d.get("ci_high"))
        except Exception:
            pass
    ctx["L5_CSDI"] = {
        "P_pregnancy_%": _pv(_csdi_p),
        "CI_low_%":      _pv(_csdi_ci_l),
        "CI_high_%":     _pv(_csdi_ci_h),
        "доступна":      _csdi_p is not None,
        "роль": ("Биологический якорь: диффузионная симуляция эмбриологии + LightGBM. "
                 "Независима от нейросетевых оценщиков. AUROC=0.661, ECE=0.029. "
                 "Расхождение с L3/L6 > 10 пп = значимый конфликт механики с NN-прогнозом."),
    }

    # ── L6: GAT (граф клинических соседей) ────────────────────────────────
    _gnn_r  = g.get("_gnn_result") or {}
    p_gnn_r = _f(g.get("_p_gnn_raw") or _gnn_r.get("gnn_prob"))
    p_gnn_e = _f(g.get("_p_gnn_ens") or _gnn_r.get("ensemble_prob"))
    w_gnn   = _f(g.get("_w_gnn")     or _gnn_r.get("w_gnn"))
    ctx["L6_GAT"] = {
        "P_raw_%":      _pv(p_gnn_r),
        "P_ensemble_%": _pv(p_gnn_e),
        "w_gnn":        round(w_gnn, 3) if w_gnn is not None else None,
        "доступна":     p_gnn_r is not None,
        "роль": ("Граф k-NN по клиническим признакам (~16K протоколов). "
                 "ensemble = w_gnn×GAT + (1−w_gnn)×KAT."),
    }
    _nb = _extract_gat_neighbors(g)
    if _nb is not None:
        ctx["L6_GAT"]["kNN_соседи"] = _nb

    # ── L7: BEFE (Байесовское слияние доказательств) ──────────────────────
    befe = g.get("_befe_res")
    if befe is not None:
        ctx["L7_BEFE"] = {
            "P_posterior_%":        _pv(getattr(befe, "posterior",      None)),
            "CI_low_%":             _pv(getattr(befe, "ci_low",         None)),
            "CI_high_%":            _pv(getattr(befe, "ci_high",        None)),
            "CI_source":            getattr(befe, "ci_source",          None),
            "надёжность_0_100":     getattr(befe, "reliability",        None),
            "полоса_надёжности":    getattr(befe, "reliability_band",   None),
            "prior_pull_%":         _pv(getattr(befe, "prior_pull",     None)),
            "evidence_pull_%":      _pv(getattr(befe, "evidence_pull",  None)),
            "OOD_клинический":      getattr(befe, "ood_clinical",       None),
            "OOD_эмбриологический": getattr(befe, "ood_embryology",     None),
            "OOD_финальный":        getattr(befe, "ood_final",          None),
            "OOD_заметка":          getattr(befe, "ood_note",           None),
            "доступна":             True,
            "роль": ("Точностно-взвешенное слияние L3+L6 (Bayesian logit-pooling). "
                     "Главный итоговый оценщик. CI = клинический Beta-posterior."),
        }
    else:
        ctx["L7_BEFE"] = {"доступна": False}

    # ── Риски применимости прогноза на перенос ────────────────────────────
    empty = res.get("empty") or {}
    ohss  = res.get("ohss")  or {}
    ctx["риски_применимости"] = {
        "P_нет_бластоцист_%":         _pv(empty.get("p_no_blast")),
        "P_нет_хороших_бластоцист_%": _pv(empty.get("p_no_good_blast")),
        "P_OHSS_умеренный_%":         _pv(ohss.get("p_moderate_ohss")),
        "P_OHSS_тяжёлый_%":           _pv(ohss.get("p_severe_ohss")),
        "пояснение": "P_нет_бластоцист > 30% → прогноз на перенос условно применим.",
    }

    return ctx


_ENSEMBLE_QUESTION = (
    "Проведи анализ ансамбля по схеме из системного промпта: матрица всех доступных "
    "слоёв, оценка согласованности, исторические аналоги GAT k-NN (если есть), "
    "якорный анализ CSDI, вывод с наиболее обоснованным диапазоном вероятностей "
    "и ранжированной неопределённостью."
)


def analyse_ensemble(g: Dict[str, Any],
                     model: str = MEDGEMMA,
                     audit: bool = True) -> str:
    """Tier 1: Блокирующий анализ ансамбля по сырым выходам всех слоёв.

    Использовать вместо consult() когда нужна оценка СОГЛАСОВАННОСТИ моделей,
    а не нарратив итогового прогноза BEFE.
    """
    ctx = build_ensemble_context(g)
    ctx_json = json.dumps(ctx, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": _SYSTEM_ANALYST},
        {"role": "user",   "content":
            f"Данные ансамбля (использовать ТОЛЬКО эти числа):\n"
            f"```json\n{ctx_json}\n```\n\n{_ENSEMBLE_QUESTION}"},
    ]
    try:
        msg  = _chat(messages, model=model, stream=False,
                     temperature=_NARRATOR_TEMP, think=_NARRATOR_THINK)
        text = _strip_thinking(msg.get("content", "").strip())
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return _OFFLINE_MSG
    except (requests.RequestException, OllamaError) as exc:
        return f"Ошибка обращения к модели «{model}»: {exc}"
    if audit:
        _audit_log(ctx, _ENSEMBLE_QUESTION, text, model)
    return text


def analyse_ensemble_stream(g: Dict[str, Any],
                            model: str = MEDGEMMA) -> Iterator[str]:
    """Tier 1: Потоковый анализ ансамбля (для st.write_stream).

    Пример подключения в app.py (новая вкладка после BEFE):
        try:
            import llm_consultant as LC
            with st.expander("Анализ ансамбля (LLM)"):
                if st.button("Анализировать согласованность", key="_ens_go"):
                    st.write_stream(LC.analyse_ensemble_stream(globals()))
        except Exception as _e:
            st.caption(f"LLM-аналитик недоступен: {_e}")
    """
    ctx = build_ensemble_context(g)
    ctx_json = json.dumps(ctx, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": _SYSTEM_ANALYST},
        {"role": "user",   "content":
            f"Данные ансамбля (использовать ТОЛЬКО эти числа):\n"
            f"```json\n{ctx_json}\n```\n\n{_ENSEMBLE_QUESTION}"},
    ]
    collected: List[str] = []
    try:
        raw = _chat(messages, model=model, stream=True,
                    temperature=_NARRATOR_TEMP, think=_NARRATOR_THINK)
        for piece in _filter_thinking_stream(raw):
            collected.append(piece)
            yield piece
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        yield _OFFLINE_MSG
        return
    except (requests.RequestException, OllamaError) as exc:
        yield f"Ошибка обращения к модели «{model}»: {exc}"
        return
    _audit_log(ctx, _ENSEMBLE_QUESTION, "".join(collected), model)


# ──────────────────────────────────────────────────────────────────────────
#  MULTI-TURN DIALOG (Tier 0 или Tier 1 с историей переписки)
# ──────────────────────────────────────────────────────────────────────────
def chat_stream(g: Dict[str, Any],
                history: List[Dict[str, str]],
                question: str,
                tier: int = 0,
                style: str = "concise",
                model: str = MEDGEMMA) -> Iterator[str]:
    """Потоковый ответ с учётом истории диалога (для st.write_stream).

    Контекст Digital Twin передаётся на каждый запрос заново — модель
    «без памяти», но всегда работает с актуальными данными пациента.

    Args:
        g       : globals() из app.py
        history : список {"role": "user"|"assistant", "content": str}
                  — предыдущие сообщения диалога (без начального системного).
        question: новый вопрос пользователя.
        tier    : 0 = нарратор (Tier 0), 1 = аналитик ансамбля (Tier 1).
        style   : для tier=0 — "narrative" | "concise"; для tier=1 игнорируется.
        model   : имя модели Ollama.
    """
    if tier == 0:
        ctx    = build_narrative_context(g)  # pre-classified compact context
        system = _SYSTEM_NARRATOR
        seed_q = _DEFAULT_QUESTION + _STYLE_HINT.get(style, "")
    else:
        ctx    = build_ensemble_context(g)
        system = _SYSTEM_ANALYST
        seed_q = _ENSEMBLE_QUESTION

    ctx_json = json.dumps(ctx, ensure_ascii=False, indent=2)

    # Первое сообщение — контекст + начальный вопрос (роль seed)
    seed_user = (
        f"/no_think\n"
        f"Контекст пациента (использовать ТОЛЬКО эти числа):\n"
        f"```json\n{ctx_json}\n```\n\n{seed_q}"
    )
    messages: List[Dict[str, str]] = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": seed_user},
    ]
    # История предыдущих ходов (assistant + user поочерёдно)
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    # Новый вопрос
    messages.append({"role": "user", "content": f"/no_think\n{question}"})

    # Для follow-up вопросов используем concise-бюджет — ответ уже был полным
    np_followup = _STYLE_NUM_PREDICT.get("concise", 1200)

    collected: List[str] = []
    try:
        raw = _chat(messages, model=model, stream=True,
                    temperature=_NARRATOR_TEMP,
                    num_predict=np_followup,
                    think=_NARRATOR_THINK)
        for piece in _filter_thinking_stream(raw):
            collected.append(piece)
            yield piece
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        yield _OFFLINE_MSG
        return
    except (requests.RequestException, OllamaError) as exc:
        yield f"Ошибка обращения к модели «{model}»: {exc}"
        return
    _audit_log(ctx, question, "".join(collected), model)


#  Демонстрационный scaffold: один инструмент — пересчёт TRP.
#  Использует gemma4 (нативный tool use); medgemma на это не рассчитана.
# ──────────────────────────────────────────────────────────────────────────
_TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_trp_simulation",
        "description": ("Пересчитать Совокупный репродуктивный потенциал (TRP) "
                        "при изменённых параметрах планирования. Возвращает "
                        "шанс ≥1 беременности за горизонт, ожидаемое число "
                        "попыток и риск закрытия биологического окна."),
        "parameters": {
            "type": "object",
            "properties": {
                "max_future_cycles": {"type": "integer",
                    "description": "Сколько будущих циклов готова пройти пациентка (1-8)."},
                "desired_children": {"type": "integer",
                    "description": "Желаемое число детей: 1 или 2."},
            },
            "required": ["max_future_cycles"],
        },
    },
}, {
    # [IMP STIM] second tool — deterministic stimulation protocol/dose.
    "type": "function",
    "function": {
        "name": "recommend_stimulation_protocol",
        "description": ("Детерминированно рассчитать по номограмме фенотип ответа, "
                        "риск СГЯ, предлагаемый протокол, целевой выход ооцитов и "
                        "ДИАПАЗОН стартовой дозы гонадотропина, плюс сопоставленные "
                        "опубликованные рекомендации. Поддержка решения, НЕ назначение."),
        "parameters": {
            "type": "object",
            "properties": {
                "protocol_pref": {"type": "string",
                    "description": "auto | antagonist | agonist (по умолчанию auto)."},
            },
            "required": [],
        },
    },
}]


def _execute_tool(name: str, args: Dict[str, Any], g: Dict[str, Any]) -> str:
    """Выполняет инструмент ВАШИМ кодом и возвращает результат строкой."""
    # [IMP STIM]
    if name == "recommend_stimulation_protocol":
        from protocol_guidance import build_protocol_guidance
        gg = dict(g)
        gg["protocol_pref"] = args.get("protocol_pref", "auto")
        block = build_protocol_guidance(gg)
        return json.dumps(block or {"error": "insufficient_inputs"},
                          ensure_ascii=False)
    if name == "run_trp_simulation":
        from trp_engine import compute_trp, TRPInput
        res = g.get("res") or {}
        inp = TRPInput(
            age=float(g.get("age", 0) or 0),
            amh=float(g.get("amh", 0) or 0),
            afc=int(g.get("afc", 0) or 0),
            bmi=float(g.get("bmi", 24.0) or 24.0),
            max_future_cycles=int(args.get("max_future_cycles", 6)),
            desired_children=int(args.get("desired_children", 1)),
            p_base_override=res.get("p_overall_cycle"),
        )
        out = compute_trp(inp)
        return json.dumps({
            "шанс_не_менее_1_беременности_проц": round(out.p_success_total * 100, 1),
            "ожидаемо_попыток": (None if out.expected_cycles_to_success != out.expected_cycles_to_success
                                 else round(out.expected_cycles_to_success, 1)),
            "биологическое_окно_лет": round(out.window_years_p50, 1),
            "риск_окно_закроется_раньше_проц": round(out.p_window_closes_first * 100, 1),
        }, ensure_ascii=False)
    return json.dumps({"error": f"unknown tool {name}"}, ensure_ascii=False)


def consult_agentic(g: Dict[str, Any],
                    question: str,
                    model: str = GEMMA4,
                    max_steps: int = 4) -> str:
    """Tier 2: даёт модели вызывать инструменты движка и формулирует ответ."""
    ctx = build_clinical_context(g, include_trp=False)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_NARRATOR +
         "\n\nПри необходимости вызывай инструменты для пересчёта сценариев. "
         "Числа из результатов инструментов также приводи дословно."},
        {"role": "user", "content":
            f"Контекст:\n```json\n{json.dumps(ctx, ensure_ascii=False)}\n```\n\n{question}"},
    ]
    try:
        for _ in range(max_steps):
            msg = _chat(messages, model=model, tools=_TOOLS, stream=False)
            calls = msg.get("tool_calls") or []
            if not calls:
                return msg.get("content", "").strip()
            messages.append(msg)
            for call in calls:
                fn = call.get("function", {})
                result = _execute_tool(fn.get("name", ""),
                                       fn.get("arguments", {}) or {}, g)
                messages.append({"role": "tool", "content": result})
        # исчерпан лимит шагов — последний обычный ответ
        final = _chat(messages, model=model, stream=False)
        return final.get("content", "").strip()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return _OFFLINE_MSG
    except (requests.RequestException, OllamaError) as exc:
        return f"Ошибка обращения к модели «{model}»: {exc}"

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Быстрая проверка соединения с Ollama:
    #   python llm_consultant.py
    print("Ollama доступна:", health_check())
    print("Загруженные модели:", list_models())

