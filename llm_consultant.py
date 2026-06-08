"""
llm_consultant.py — IVF Digital Twin v7.0
═══════════════════════════════════════════════════════════════════════════
Локальный LLM-слой консультанта поверх Digital Twin. Работает через Ollama
на localhost — без выхода в интернет (offline-safe для клиник).

АРХИТЕКТУРНАЯ РОЛЬ
    Presentation / interaction слой. НЕ оценщик вероятности, НЕ участвует
    в fusion. Ни KAT, ни одна из моделей не затрагиваются.

    • Tier 0  — нарратор: L7-summary → текст              (MEDGEMMA)
               Клиническое резюме итогового BEFE-вывода.

    • Tier 1  — аналитик ансамбля: raw per-layer outputs  (MEDGEMMA)
               Сырые выходы всех слоёв (≡ аналитический CSV).
               Согласованность, CSDI-якорь, OOD, ранжированная
               неопределённость. Ценность: то, что L7 не видит.

    • Tier 2  — агент: function calling над движком DT     (GEMMA4)

ЖЁСТКОЕ ПРАВИЛО
    LLM НИКОГДА не вычисляет и не выдумывает числа. Все вероятности и CI
    берутся дословно из переданного контекста. Это закреплено в системном
    промпте и поддержано низкой температурой.

ЗАВИСИМОСТИ
    Только стандартная библиотека + requests. Пакет `ollama` не требуется.
    Модуль не импортирует streamlit на верхнем уровне — тестируется автономно.

ПОДКЛЮЧЕНИЕ В app.py (пример, app.py НЕ правится этим модулем):
    # внутри уже посчитанного прохода, например новой вкладкой:
    try:
        import llm_consultant
        with st.expander("Консультант (локальная LLM)"):
            q = st.text_input("Вопрос", key="_llm_q")
            if st.button("Сформулировать", key="_llm_go"):
                st.write_stream(llm_consultant.consult_stream(globals(), q))
    except Exception as _e:
        st.caption(f"LLM-консультант недоступен: {_e}")

ПРОВЕРКА АВТОНОМНО:
    >>> import llm_consultant as L
    >>> L.health_check()           # доступна ли Ollama
    >>> print(L.consult({...}))    # с фиктивными globals
═══════════════════════════════════════════════════════════════════════════
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
_NARRATOR_TEMP = 0.30
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


# ──────────────────────────────────────────────────────────────────────────
#  СИСТЕМНЫЙ ПРОМПТ (Tier 0)
# ──────────────────────────────────────────────────────────────────────────
_SYSTEM_NARRATOR = """Ты — опытный клинический консультант системы IVF Digital Twin.
Твой читатель — ВРАЧ-репродуктолог. Твоя задача — не перечислять данные, а
создать КЛИНИЧЕСКИЙ НАРРАТИВ: объяснить, что стоит за каждой цифрой, описать
возможные пути развития событий при прохождении цикла, выделить то, на что
врачу следует обратить внимание на каждом этапе.

СТРОГИЕ ПРАВИЛА (нарушение недопустимо):
1. Все числа — ТОЛЬКО из предоставленного JSON. Не вычислять, не выдумывать,
   не округлять по-своему. Если значения нет — «не рассчитано».
2. Числа — опорные точки; важнее — их клинический смысл и возможные следствия.
3. Строй связный объяснительный текст: причина → следствие, «если → то»,
   «это означает», «вероятный сценарий», «следует ожидать».
4. Тон — коллеги-консультанта: осведомлённый, объяснительный, без директив.
   Аудитория — врач; клиническая терминология, профессиональный регистр.
5. Не давай конкретные дозы/протоколы как директиву; решение — за врачом.
6. Отвечай на русском. Термины BEFE, CSDI, MII, эуплоид, OHSS — свободно.

ФОРМАТ — развёрнутые разделы с объяснениями и сценариями
(раздел пропускай только при полном отсутствии данных):

### 1. Главный прогноз
Назови вероятность BEFE и 95% ДИ. Затем объясни: что означает этот уровень
вероятности в клинической практике ЭКО (сравни с типичным диапазоном для
возраста и фенотипа)? Какие факторы данного случая тянут прогноз вверх или вниз?
Что говорит ширина ДИ о надёжности числа — насколько широк диапазон реальных
возможных исходов?

### 2. Сценарии развития цикла
Используй p_overall_cycle, p_per_transfer, p_cancel_risk и p_viable для описания
ключевых развилок цикла:
  — Вероятностный «маршрут» от стимуляции до переноса: что ожидается на каждом
    этапе (пункция → оплодотворение → культивирование → перенос)?
  — При каком сценарии цикл будет отменён (что значит этот конкретный % отмены)?
  — Разница между p_per_transfer и p_overall_cycle: что означает «дойти до
    переноса»? Какова вероятность не дойти и что тогда происходит?
  — Как кумулятивная вероятность (при переносе всех эмбрионов) соотносится с
    одним переносом — что означает этот разрыв для планирования тактики?

### 3. Фенотип ответа и ожидания
Опиши, что означает данный тип ответа (кластер) для ЭТОГО пациента:
  — Чего ожидать от стимуляции: диапазон фолликулов, ооцитов, вероятная
    чувствительность к гонадотропинам?
  — Как этот фенотип соотносится с данными амг/афч/возраста — подтверждают
    ли они друг друга, или есть расхождение?
  — Какие нюансы протокола обычно актуальны для этого типа ответа?

### 4. Клиническая картина рисков
Для каждого присутствующего риска — не просто процент, а:
  — OHSS: что означает умеренный/тяжёлый уровень, когда обычно проявляется
    (после пункции или после переноса/ХГЧ), какие симптомы-маркеры, при каком
    пороге рассматривается freeze-all вместо свежего переноса?
  — Риск отсутствия бластоцист: на каком этапе цикла это станет ясно, что
    означает для текущей тактики и стратегии следующих попыток?
  — OHSS и риск отмены: есть ли связь для этого случая?

### 5. Стратегия банкинга MII ооцитов
КЛЮЧЕВОЕ РАЗГРАНИЧЕНИЕ: банкируются MII ООЦИТЫ (по нескольким циклам стимуляции),
а не эмбрионы. Из накопленных MII → оплодотворение → культивирование → ПГТ-А →
эуплоидные бластоцисты → перенос. Объясни логику цепочки:
  — Сколько MII ожидается в ЭТОМ цикле (поле «MII_ожидается_этот_цикл»)?
  — Сколько из них станут эуплоидными бластоцистами с учётом p_per_mii,
    фертилизации, бластуляции и эуплоидии по возрасту
    (поле «эуплоидных_бластоцист_из_этих_MII»)?
  — Сколько эуплоидных эмбрионов нужно для P50/P70 беременности? Почему?
  — Сколько ЦИКЛОВ СТИМУЛЯЦИИ нужно, чтобы накопить достаточно MII
    (поля «циклов_стимуляции_для_P50/70»)? Что означает «суммарно MII»?
  — Когда накопление MII оправдано против немедленного переноса?
  — Как возраст влияет на срочность: почему снижение эуплоидии по возрасту
    (Franasiak) делает стратегию накопления более или менее выгодной?

### 6. Исторические аналоги (GAT k-NN)
Если данные kNN_соседи есть, расскажи КЛИНИЧЕСКИ ЗНАЧИМУЮ ИСТОРИЮ:
  — Из какой «популяции» исторических случаев наибольшее сходство? Что общего
    у ближайших аналогов (возраст, ответ, эмбриологические параметры)?
  — Опиши 2–3 ближайших аналога (все числа — дословно из контекста): что
    отличает случай с наибольшим GNN-прогнозом от случая с наименьшим?
    Какие признаки «разделяют» лучший и худший прогноз среди соседей?
  — Как разброс GNN-вероятностей по соседям соотносится с диапазоном BEFE ДИ —
    подтверждает ли реальная база ту же степень неопределённости?

### 7. Природа и смысл неопределённости
Объясни неопределённость механистически, а не просто как ширину ДИ:
  — Откуда она: модели согласованы (узкий разброс ансамбля) или расходятся?
    Если расходятся — что могут видеть по-разному нейросетевой и механистический
    оценщики?
  — Что означает ширина ДИ BEFE для принятия врачебных решений: где стоит
    опираться на центральную оценку, а где лучше работать с диапазоном?
  — Надёжность BEFE и её полоса: что клинически означает надёжность 60 vs 85?
  — Если OOD активен — какой параметр вышел за пределы обучения? Как это
    конкретно влияет на интерпретацию прогноза?

### 8. Клинический итог
Синтез: не просто пересказ — найди 2–3 ключевых клинических соображения,
которые определяют ведение именно этого пациента. Что здесь нестандартно или
требует особого внимания? Каков главный вопрос, на который система отвечает
неопределённо — и что это значит для врача?"""


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
    ctx = build_clinical_context(g, include_trp=include_trp)
    try:
        msg = _chat(_build_messages(ctx, question), model=model,
                    stream=False, temperature=_NARRATOR_TEMP,
                    num_predict=num_predict, think=_NARRATOR_THINK)
        text = _strip_thinking(msg.get("content", "").strip())
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return _OFFLINE_MSG
    except (requests.RequestException, OllamaError) as exc:
        return f"Ошибка обращения к модели «{model}»: {exc}"
    if audit:
        _audit_log(ctx, question, text, model)
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
    ctx = build_clinical_context(g, include_trp=include_trp)
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
    _audit_log(ctx, question, "".join(collected), model)


# ──────────────────────────────────────────────────────────────────────────
#  АУДИТ (расширение существующего analytics-пайплайна)
# ──────────────────────────────────────────────────────────────────────────
def _audit_log(ctx: Dict[str, Any], question: str, answer: str,
               model: str) -> None:
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
        ctx    = build_clinical_context(g)
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
}]


def _execute_tool(name: str, args: Dict[str, Any], g: Dict[str, Any]) -> str:
    """Выполняет инструмент ВАШИМ кодом и возвращает результат строкой."""
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
    """
    Дымовой тест без app.py — фиктивные globals.

    Запуск:
        cd <папка проекта>
        python llm_consultant.py                    # оба теста
        python llm_consultant.py tier0              # только Tier 0 (нарратор)
        python llm_consultant.py tier1              # только Tier 1 (ансамбль)

    Предварительно:
        ollama serve                                # в отдельном терминале
        ollama pull medgemma1.5                    # первый раз
        ollama list                                 # проверить что модель есть
    """
    import sys, time

    # ── Фиктивные globals (имитируют состояние app.py после расчёта) ──────
    class _Befe:
        posterior, ci_low, ci_high     = 0.412, 0.331, 0.498
        ci_source                      = "clinical-beta-posterior"
        reliability, reliability_band  = 68, "moderate"
        prior_pull, evidence_pull      = 0.44, 0.56
        ood_clinical                   = False
        ood_embryology                 = False
        ood_final                      = False
        ood_note                       = "Within distribution"

    fake_globals = {
        "age": 34, "amh": 1.8, "afc": 11, "bmi": 23.5, "attempt_number": 2,
        "_befe_res": _Befe(),
        "_p_kat_raw": 0.40, "_p_nvsa": 0.43, "_w_gnn": 0.35,
        "_p_gnn_raw": 0.34, "_p_gnn_ens": 0.38,
        "_gnn_result": {"gnn_prob": 0.34, "ensemble_prob": 0.38, "w_gnn": 0.35},
        "res": {
            "p_per_transfer": 0.41, "p_overall_cycle": 0.38,
            "p_cum_if_viable": 0.55, "p_viable": 0.93,
            "p_cancel_risk": 0.06,
            "blasts_med": 3.0, "good_med": 2.0, "mii_med": 8.0,
            "rate_ci": (0.30, 0.53),
            "nn_prediction": {
                "base_prob_mean": 0.40,
                "base_prob_ci":   (0.31, 0.50),
            },
            "nn_nvsa": {
                "adjusted_mean": 0.43,
                "adjusted_ci":   (0.33, 0.53),
            },
            "cluster_analysis": {
                "dominant_cluster": 0,
                "cluster_probs": {0: 0.68, 1: 0.14, 2: 0.18},
            },
            "empty": {"p_no_blast": 0.09, "p_no_good_blast": 0.21},
            "ohss":  {"p_moderate_ohss": 0.10, "p_severe_ohss": 0.02},
        },
        "ca": {"dominant_cluster": 0,
               "cluster_probs": {0: 0.68, 1: 0.14, 2: 0.18}},
    }

    run_tier = sys.argv[1] if len(sys.argv) > 1 else "both"

    # ── Структура контекста без вызова LLM ────────────────────────────────
    print("=" * 60)
    print("  Ollama доступна:", health_check())
    print("=" * 60)

    if run_tier in ("tier1", "both"):
        print("\n[Tier 1] build_ensemble_context — структура:")
        print(json.dumps(build_ensemble_context(fake_globals),
                         ensure_ascii=False, indent=2))

    if run_tier in ("tier0", "both"):
        print("\n[Tier 0] build_clinical_context — структура:")
        print(json.dumps(build_clinical_context(fake_globals, include_trp=False),
                         ensure_ascii=False, indent=2))

    if not health_check():
        print("\nOllama не запущена — LLM-тест пропущен.")
        print("Запустите `ollama serve` и повторите.")
        sys.exit(0)

    # ── Прогрев ───────────────────────────────────────────────────────────
    print("\nПрогрев модели (первый холодный старт на CPU — несколько минут)...")
    t0 = time.time()
    ok, msg = warmup(MEDGEMMA)
    print(f"  {msg}  [{time.time() - t0:.0f} c]")
    if not ok:
        sys.exit(1)

    # ── Tier 1: анализ ансамбля ───────────────────────────────────────────
    if run_tier in ("tier1", "both"):
        print("\n" + "=" * 60)
        print("  TIER 1 — Анализ ансамбля (Ensemble Analyst)")
        print("=" * 60)
        t1 = time.time()
        print(analyse_ensemble(fake_globals, audit=False))
        print(f"\n[генерация: {time.time() - t1:.0f} c]")

    # ── Tier 0: клиническое резюме ────────────────────────────────────────
    # По умолчанию — narrative. Для быстрого теста: tier0 concise
    # Переменные окружения для тонкой настройки скорости:
    #   DT_LLM_NP_NARRATIVE=900   (меньше слов — быстрее)
    #   DT_LLM_TIMEOUT_READ=1200  (для очень медленных CPU)
    if run_tier in ("tier0", "both"):
        _t0_style = sys.argv[2] if len(sys.argv) > 2 else "narrative"
        _np = _STYLE_NUM_PREDICT.get(_t0_style, 1400)
        print("\n" + "=" * 60)
        print(f"  TIER 0 — Клиническое резюме (style={_t0_style}, "
              f"max_tokens={_np}, timeout={_TIMEOUT_READ}s)")
        print("=" * 60)
        t2 = time.time()
        print(consult(fake_globals, include_trp=False,
                      style=_t0_style, audit=False))
        elapsed = time.time() - t2
        print(f"\n[генерация: {elapsed:.0f} c  ·  "
              f"~{elapsed / max(_np, 1) * 1000:.0f} мс/токен]")
