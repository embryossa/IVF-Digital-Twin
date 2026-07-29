# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
score.py — детерминированная оценка нарративов (без вызова LLM, миллисекунды).

Метрики (все — проверяемые правила, не «мнение»):
  G  numeric_grounding    доля чисел текста, подтверждённых контекстом
  R  reclassification     нарушения «не переклассифицировать» (метка vs формулировка)
  D  directive_language   императивы, запрещённые правилом 3 системного промпта
  C  coverage             покрытие обязательных разделов
  L  leakage              сырые JSON-ключи / англицизмы / think-теги в тексте
  F  faithfulness         штатный faithfulness.check_faithfulness (цитаты + доза)
  N  null_safety          разговор о том, чего нет в контексте (доза без номограммы)
  P  perf                 tok/s, время, обрыв генерации

Запуск: python score.py [runs.jsonl]
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
DT_DIR = r"C:\Users\User\Desktop\IVF\AI\IVF Digital Twin Pro\IVF Digital Twin"
if DT_DIR not in sys.path:
    sys.path.insert(0, DT_DIR)
os.chdir(DT_DIR)

import faithfulness as FA  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
#  G — числовая заземлённость
# ──────────────────────────────────────────────────────────────────────────
# Числа, легитимные без прямого присутствия в JSON: уровень ДИ, целевые
# вероятности из имён полей (P50/P70/P80/P90), проценты-ориентиры из
# interpretation-подсказки («каждый 2–3-й перенос»), номера разделов.
_CONVENTIONAL = {95.0, 100.0, 50.0, 70.0, 80.0, 90.0, 0.0,
                 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
_NUM = re.compile(r"(?<![\w.,])(\d{1,4}(?:[.,]\d{1,2})?)")


def _collect_ctx_numbers(obj: Any, acc: set) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_ctx_numbers(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_ctx_numbers(v, acc)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        acc.add(round(float(obj), 3))
    elif isinstance(obj, str):
        # числа, зашитые в строковые поля («42.5–63.1», «100–150 МЕ»)
        for m in _NUM.finditer(obj):
            try:
                acc.add(round(float(m.group(1).replace(",", ".")), 3))
            except ValueError:
                pass


def numeric_grounding(text: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    ctx_nums: set = set()
    _collect_ctx_numbers(ctx, ctx_nums)
    # производные, которые код сам подаёт как проценты/пп
    derived = set()
    for n in list(ctx_nums):
        derived.add(round(n * 100, 1))
        derived.add(round(n, 0))
    allowed = ctx_nums | derived | _CONVENTIONAL

    # не трогаем номера markdown-заголовков «### 3.»
    body = re.sub(r"^#{1,6}\s*\d+\.?", "", text, flags=re.M)
    found, ungrounded = [], []
    for m in _NUM.finditer(body):
        raw = m.group(1).replace(",", ".")
        try:
            v = float(raw)
        except ValueError:
            continue
        found.append(v)
        ok = any(abs(v - a) <= 0.55 for a in allowed)
        if not ok:
            ctxt = body[max(0, m.start() - 45):m.end() + 25].replace("\n", " ")
            ungrounded.append({"value": v, "ctx": ctxt.strip()})
    n = len(found)
    return {"n_numbers": n,
            "n_ungrounded": len(ungrounded),
            "rate": round(1 - len(ungrounded) / n, 3) if n else 1.0,
            "examples": ungrounded[:12]}


# ──────────────────────────────────────────────────────────────────────────
#  R — переклассификация
# ──────────────────────────────────────────────────────────────────────────
# «высокая вероятность ОТМЕНЫ» — не оптимизм; исключаем негативные подлежащие,
# иначе метрика ловит ложные срабатывания.
_POS_PROG = re.compile(
    r"(высок\w+\s+(?:шанс|вероятност)(?!\w*\s+(?:отмен|риск|пуст|потер|неудач|"
    r"СГЯ|гиперстимул|отсутств|прерыван))|"
    r"благоприятн\w+\s+прогноз|хорош\w+\s+прогноз|прогноз\s+благоприятн)", re.I)
_NEG_PROG = re.compile(r"(низк\w+\s+(?:шанс|вероятност)|неблагоприятн\w+\s+прогноз|"
                       r"плох\w+\s+прогноз|мал\w+\s+шанс)", re.I)
_HIGH_OHSS = re.compile(r"(высок\w+|значительн\w+|существенн\w+)\s+риск\w*\s+"
                        r"(?:развития\s+)?(?:СГЯ|OHSS)", re.I)
_LOW_OHSS = re.compile(r"(низк\w+|минимальн\w+|незначительн\w+)\s+риск\w*\s+"
                       r"(?:развития\s+)?(?:СГЯ|OHSS)", re.I)
_HIGH_CONF = re.compile(r"(высок\w+\s+(?:надёжност|надежност|достоверност|уверенност)|"
                        r"прогноз\w*\s+надёжен|высоко\s*надёжн)", re.I)
_LOW_CONF = re.compile(r"(низк\w+\s+(?:надёжност|надежност|достоверност)|"
                       r"ненадёжн|низкая\s+уверенность)", re.I)
_DISAGREE = re.compile(r"(несоглас\w+|рассогласован\w+|расхожден\w+\s+модел|"
                       r"модели\s+расход\w+|значим\w+\s+расхожден)", re.I)
_AGREE = re.compile(r"(высок\w+\s+соглас\w+|модели\s+соглас\w+|"
                    r"ансамбль\s+единодуш|полн\w+\s+соглас\w+)", re.I)
_LOW_EMPTY = re.compile(r"(низк\w+|минимальн\w+)\s+риск\w*\s+"
                        r"(?:пуст\w+\s+цикл|отсутстви\w+\s+бластоцист)", re.I)
_HIGH_EMPTY = re.compile(r"(высок\w+|значительн\w+)\s+риск\w*\s+"
                         r"(?:пуст\w+\s+цикл|отсутстви\w+\s+бластоцист)", re.I)


def reclassification(text: str, flags: Dict[str, Any]) -> Dict[str, Any]:
    v: List[str] = []
    lvl = flags.get("prognosis_level")
    if lvl == "low" and _POS_PROG.search(text):
        v.append(f"level=low, но текст: «{_POS_PROG.search(text).group(0)}»")
    if lvl in ("favorable", "good") and _NEG_PROG.search(text):
        v.append(f"level={lvl}, но текст: «{_NEG_PROG.search(text).group(0)}»")

    oh = flags.get("OHSS_risk")
    if oh == "low" and _HIGH_OHSS.search(text):
        v.append(f"OHSS=low, но текст: «{_HIGH_OHSS.search(text).group(0)}»")
    if oh == "high" and _LOW_OHSS.search(text):
        v.append(f"OHSS=high, но текст: «{_LOW_OHSS.search(text).group(0)}»")

    ec = flags.get("empty_cycle_risk")
    if ec == "high" and _LOW_EMPTY.search(text):
        v.append(f"empty=high, но текст: «{_LOW_EMPTY.search(text).group(0)}»")
    if ec == "low" and _HIGH_EMPTY.search(text):
        v.append(f"empty=low, но текст: «{_HIGH_EMPTY.search(text).group(0)}»")

    # Категория согласия ансамбля — отдельная метка, её тоже нельзя пересказывать
    # в противоположную сторону (наблюдалось: agreement=moderate → «несогласие моделей»).
    agr = flags.get("agreement_category")
    if agr in ("high_model_agreement", "moderate_model_agreement") and _DISAGREE.search(text):
        v.append(f"agreement={agr}, но текст: «{_DISAGREE.search(text).group(0)}»")
    if agr == "high_model_disagreement" and _AGREE.search(text):
        v.append(f"agreement={agr}, но текст: «{_AGREE.search(text).group(0)}»")

    gr = flags.get("confidence_grade")
    if gr == "C" and _HIGH_CONF.search(text):
        v.append(f"grade=C, но текст: «{_HIGH_CONF.search(text).group(0)}»")
    if gr == "A" and _LOW_CONF.search(text):
        v.append(f"grade=A, но текст: «{_LOW_CONF.search(text).group(0)}»")
    return {"n": len(v), "violations": v}


# ──────────────────────────────────────────────────────────────────────────
#  D — директивный язык
# ──────────────────────────────────────────────────────────────────────────
_DIRECTIVE = [
    (re.compile(r"\bпоказан[оаы]?\b", re.I), "показан"),
    (re.compile(r"\bпротивопоказан", re.I), "противопоказан"),
    (re.compile(r"\bназнач(?:ить|аем|ается|ают|ение)\b", re.I), "назначить"),
    (re.compile(r"\bрекоменд(?:уется|уем|ованн?о)\b", re.I), "рекомендуется"),
    (re.compile(r"\bнеобходимо\s+(?:провести|выполнить|назначить)", re.I), "необходимо провести"),
    (re.compile(r"\bследует\s+(?:провести|назначить|выполнить)", re.I), "следует провести"),
    (re.compile(r"\bтребуется\s+(?:провести|назначить)", re.I), "требуется провести"),
    (re.compile(r"\bдолжн[аоы]\s+(?:быть\s+)?(?:назначен|проведен)", re.I), "должна быть назначена"),
    (re.compile(r"\bпроводим\b|\bвыполняем\b", re.I), "проводим"),
]
_SOFT = re.compile(r"(можно рассмотреть|имеет смысл (?:обсудить|рассмотреть)|"
                   r"стоит (?:учитывать|обсудить|держать|рассмотреть)|"
                   r"может быть основанием|целесообразно обсудить|"
                   r"разумно обсудить|можно обсудить)", re.I)


def directive_language(text: str) -> Dict[str, Any]:
    hits = []
    for pat, name in _DIRECTIVE:
        for m in pat.finditer(text):
            ctxt = text[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
            hits.append({"kind": name, "ctx": ctxt.strip()})
    return {"n": len(hits), "hits": hits[:10],
            "n_soft": len(_SOFT.findall(text))}


# ──────────────────────────────────────────────────────────────────────────
#  C — покрытие разделов
# ──────────────────────────────────────────────────────────────────────────
SECTIONS: List[Tuple[str, re.Pattern, str]] = [
    ("1_прогноз",     re.compile(r"(главн\w+ прогноз|основн\w+ прогноз|прогноз беременности|"
                                 r"^#+\s*\d*\.?\s*прогноз)", re.I | re.M), "main_forecast"),
    ("2_сценарии",    re.compile(r"(сценари|развилк|развити\w+ цикла|ход цикла)", re.I), "cycle"),
    ("3_фенотип",     re.compile(r"(фенотип|тип ответа|ответ на стимул)", re.I), "response_phenotype"),
    ("4_риски",       re.compile(r"(риск)", re.I), "risks"),
    ("5_банкинг",     re.compile(r"(банкинг|накоплени\w+ (?:MII|ооцит))", re.I), "banking_mii"),
    ("6_аналоги",     re.compile(r"(аналог|похож\w+ случа|сосед|исторически)", re.I), "historical_analogues"),
    ("7_надёжность",  re.compile(r"(надёжност|надежност|доверительн|уверенност|confidence)", re.I), "prediction_confidence"),
    # «Суть» — тот же слот, что «Итог»: v4 переносит вывод в начало текста.
    ("8_итог",        re.compile(r"(итог|заключени|резюме|в сумме|главное|^#+\s*суть)",
                                 re.I | re.M), None),
    ("9_протокол",    re.compile(r"(протокол стимул|номограмм|стартов\w+ доз|гайдлайн|рекомендаци\w+ ESHRE|ASRM)", re.I), "protocol_guidance"),
]


def coverage(text: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    present, missing, na = [], [], []
    for name, pat, key in SECTIONS:
        applicable = (key is None) or bool(ctx.get(key))
        # разделы с пустыми под-значениями считаем неприменимыми
        if applicable and key and isinstance(ctx.get(key), dict):
            applicable = any(v is not None for v in ctx[key].values())
        if not applicable:
            na.append(name)
            continue
        (present if pat.search(text) else missing).append(name)
    denom = len(present) + len(missing)
    return {"present": present, "missing": missing, "not_applicable": na,
            "rate": round(len(present) / denom, 3) if denom else 1.0}


# ──────────────────────────────────────────────────────────────────────────
#  L — утечки
# ──────────────────────────────────────────────────────────────────────────
# Ловим и `high_model_agreement`, и `P_pct` / `CI_low_pct` / `P_no_blast_pct`:
# модель печатает сырые ключи JSON прямо в клиническом тексте.
_SNAKE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+){1,4}\b")
_THINK = re.compile(r"<unused9[45]>|</?think>|/no_think", re.I)
_LATIN_WORD = re.compile(r"\b[A-Za-z]{4,}\b")
_ALLOWED_LATIN = {"BEFE", "OHSS", "MII", "CSDI", "GAT", "GNN", "ESHRE", "ASRM",
                  "BFS", "POSEIDON", "AMH", "AFC", "BMI", "ICSI", "IVF", "KPI",
                  "PGT", "OPTIMIST", "Hum", "Reprod", "Fertil", "Steril", "Open",
                  "Guideline", "Group", "Practice", "Committee", "Online",
                  "Biomed", "Med", "Assist", "Genet", "responder", "freeze",
                  "all", "high", "low", "moderate", "good", "favorable"}


def leakage(text: str) -> Dict[str, Any]:
    snake = sorted(set(_SNAKE.findall(text)))
    think = _THINK.findall(text)
    latin = [w for w in set(_LATIN_WORD.findall(text))
             if w not in _ALLOWED_LATIN and w.lower() not in
             {a.lower() for a in _ALLOWED_LATIN}]
    return {"snake_keys": snake, "n_snake": len(snake),
            "think_tags": think, "n_think": len(think),
            "latin_words": sorted(latin)[:15], "n_latin": len(latin)}


# ──────────────────────────────────────────────────────────────────────────
#  N — null-safety
# ──────────────────────────────────────────────────────────────────────────
_DOSE_TALK = re.compile(r"\d{2,4}\s*(?:МЕ|IU)\b|номограмм|стартов\w+ доз", re.I)
_CITATION_TALK = re.compile(r"\[(?:[^\]]{4,})\]|ESHRE|ASRM|POSEIDON|Hum Reprod|Fertil Steril")
_BANKING_TALK = re.compile(r"банкинг|накоплени\w+ MII|эуплоидн\w+ бластоцист", re.I)
_NEIGHBOR_TALK = re.compile(r"сосед|аналог\w+ случа|похож\w+ цикл\w+ из баз", re.I)


def null_safety(text: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    v = []
    if not ctx.get("protocol_guidance"):
        if _DOSE_TALK.search(text):
            v.append("говорит о дозе/номограмме без protocol_guidance: "
                     f"«{_DOSE_TALK.search(text).group(0)}»")
        if _CITATION_TALK.search(text):
            v.append("цитирует гайдлайны без блока protocol_guidance: "
                     f"«{_CITATION_TALK.search(text).group(0)}»")
    if not ctx.get("banking_mii") and _BANKING_TALK.search(text):
        v.append("раздел банкинга без данных banking_mii")
    if not ctx.get("historical_analogues") and _NEIGHBOR_TALK.search(text):
        v.append("говорит о GAT-соседях без historical_analogues")
    if not ctx.get("main_forecast") and re.search(r"BEFE", text):
        v.append("упоминает BEFE, которого нет в контексте")
    return {"n": len(v), "violations": v}


# ──────────────────────────────────────────────────────────────────────────
#  B — связка «метка ↔ число» (число есть в JSON, но приписано не тому полю)
#  Пилот показал: numeric_grounding это НЕ ловит (2PN=5 назван «5 зрелых ооцитов»).
# ──────────────────────────────────────────────────────────────────────────
def _path(ctx: Dict[str, Any], *keys):
    cur: Any = ctx
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


# (имя, regex с группой числа, функция извлечения ожидаемого значения, допуск)
_BINDINGS = [
    ("возраст", re.compile(r"(\d{2})\s*(?:лет|года|год)\b"),
     lambda c: _path(c, "patient", "age"), 0.6),
    ("AMH", re.compile(r"AMH[^\d\n]{0,25}?(\d{1,2}[.,]?\d?)", re.I),
     lambda c: _path(c, "patient", "amh_ng_ml"), 0.06),
    ("AFC", re.compile(r"(?:AFC|АФЧ|антральн\w+ фолликул\w+)[^\d\n]{0,25}?(\d{1,2})", re.I),
     lambda c: _path(c, "patient", "afc"), 0.6),
    ("P_беременности", re.compile(r"(?:шанс|вероятност\w+)\s+(?:наступления\s+)?беременност\w+"
                                  r"[^\d\n]{0,40}?(\d{1,2}[.,]\d)", re.I),
     lambda c: _path(c, "main_forecast", "P_pct"), 0.15),
    ("MII_цикла", re.compile(r"(\d{1,3})\s*(?:зрел\w+ ооцит|MII)", re.I),
     lambda c: _path(c, "banking_mii", "MII_this_cycle"), 0.6),
    ("эуплоиды", re.compile(r"(\d{1,2}[.,]\d)\s*эуплоидн", re.I),
     lambda c: _path(c, "banking_mii", "euploids_expected"), 0.15),
    ("P_без_бластоцист", re.compile(r"(?:отсутстви\w+ бластоцист|пуст\w+ цикл|без бластоцист)"
                                    r"[^\d\n]{0,40}?(\d{1,2}[.,]\d)", re.I),
     lambda c: _path(c, "risks", "empty_cycle", "P_no_blast_pct"), 0.15),
    ("OHSS_тяжёлый", re.compile(r"(?:тяжёл\w+|тяжел\w+)\s*(?:форм\w+|СГЯ)"
                                r"[^\d\n]{0,40}?(\d{1,2}[.,]\d)", re.I),
     lambda c: _path(c, "risks", "OHSS", "P_sev_pct"), 0.15),
    ("надёжность_BEFE", re.compile(r"(?:надёжност\w+|надежност\w+)\s*(?:оценки\s*)?BEFE"
                                   r"[^\d\n]{0,25}?(\d{1,3})", re.I),
     lambda c: _path(c, "prediction_confidence", "BEFE_reliability"), 0.6),
]

_UNSUPPORTED = re.compile(
    r"(это подтверждает|что подтверждает|это доказывает|что доказывает|"
    r"это гарантирует|можно утверждать, что|очевидно, что|"
    r"свидетельствует о том, что)", re.I)


_ANALOGUE_HDR = re.compile(r"(аналог|похож\w+ случа|сосед|исторически)", re.I)

# Внутри раздела «аналоги» числа принадлежат СОСЕДУ, а не пациентке —
# сверяем их с historical_analogues.closest.
_NB_BINDINGS = [
    ("сосед_возраст", re.compile(r"(\d{2})\s*(?:лет|года|год)\b"), "возраст", 0.6),
    ("сосед_AFC", re.compile(r"(?:AFC|АФЧ)[^\d\n]{0,20}?(\d{1,2})", re.I), "АФЧ", 0.6),
    ("сосед_MII", re.compile(r"(\d{1,3})\s*(?:зрел\w+ ооцит|MII)|"
                             r"(?:MII|зрел\w+ ооцит\w+)[^\d\n]{0,15}?(\d{1,3})", re.I),
     "MII", 0.6),
    ("сосед_бластоцисты", re.compile(r"(\d{1,2})\s*бластоцист", re.I), "бластоцисты", 0.6),
    ("сосед_хорошие_Bl", re.compile(r"(\d{1,2})\s*хорош\w+\s*(?:бластоцист|Bl)", re.I),
     "хорошие_Bl", 0.6),
]


def _split_sections(text: str):
    """→ (текст вне раздела аналогов, текст раздела аналогов)."""
    parts = re.split(r"(?m)^#{1,6}\s*", text)
    main, ana = [], []
    for p in parts:
        head = p.split("\n", 1)[0]
        (ana if _ANALOGUE_HDR.search(head) else main).append(p)
    return "\n".join(main), "\n".join(ana)


def label_binding(text: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Число рядом с меткой должно совпадать со значением ИМЕННО этого поля.

    Раздел «исторические аналоги» проверяется отдельно — против профиля
    ближайшего соседа, а не пациентки.
    """
    main_txt, ana_txt = _split_sections(text)
    bad = []
    for name, pat, getter, tol in _BINDINGS:
        exp = getter(ctx)
        if exp is None:
            continue
        try:
            exp = float(exp)
        except (TypeError, ValueError):
            continue
        for m in pat.finditer(main_txt):
            try:
                got = float(m.group(1).replace(",", "."))
            except (ValueError, TypeError):
                continue
            if abs(got - exp) > tol:
                ctxt = main_txt[max(0, m.start() - 50):m.end() + 30].replace("\n", " ")
                bad.append({"field": name, "expected": exp, "found": got,
                            "ctx": ctxt.strip()})

    closest = _path(ctx, "historical_analogues", "closest") or {}
    if ana_txt and closest:
        for name, pat, key, tol in _NB_BINDINGS:
            exp = closest.get(key)
            if exp is None:
                continue
            try:
                exp = float(exp)
            except (TypeError, ValueError):
                continue
            for m in pat.finditer(ana_txt):
                grp = next((x for x in m.groups() if x), None)
                if grp is None:
                    continue
                try:
                    got = float(grp.replace(",", "."))
                except ValueError:
                    continue
                if abs(got - exp) > tol:
                    ctxt = ana_txt[max(0, m.start() - 50):m.end() + 30].replace("\n", " ")
                    bad.append({"field": name, "expected": exp, "found": got,
                                "ctx": ctxt.strip()})
    return {"n": len(bad), "violations": bad[:8]}


# Поля-СЧЁТЧИКИ: их значение — штука, а не процент. Компактный контекст подаёт их
# голым числом без единицы («euploids_expected: 0.1»), и модель дописывает «%».
_COUNT_FIELDS = [
    ("banking_mii", "euploids_expected"), ("banking_mii", "MII_this_cycle"),
    ("banking_mii", "mii_target_80pct"), ("banking_mii", "cycles_for_P50"),
    ("banking_mii", "cycles_for_P70"), ("banking_mii", "euploids_for_P50"),
    ("historical_analogues", "n"),
]
_PCT_NUM = re.compile(r"(\d{1,3}(?:[.,]\d{1,2})?)\s*%")


def unit_confusion(text: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    counts, pcts = {}, set()

    def walk(o, keypath=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, keypath)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            if keypath.endswith(("_pct", "_проц", "_%")) or "pct" in keypath:
                pcts.add(round(float(o), 2))
    walk(ctx)
    for parent, key in _COUNT_FIELDS:
        v = _path(ctx, parent, key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            counts[round(float(v), 2)] = f"{parent}.{key}"

    bad = []
    for m in _PCT_NUM.finditer(text):
        try:
            v = round(float(m.group(1).replace(",", ".")), 2)
        except ValueError:
            continue
        # 50/70/80/90 — это метки целевых вероятностей (P50/P70/P80), а не счётчики,
        # даже если какое-то поле-счётчик случайно равно тому же числу.
        if v in _CONVENTIONAL:
            continue
        if v in counts and v not in pcts:
            bad.append({"value": v, "field": counts[v],
                        "ctx": text[max(0, m.start() - 70):m.end() + 15]
                        .replace("\n", " ").strip()})
    return {"n": len(bad), "violations": bad[:5]}


_LIVEBIRTH = re.compile(r"живорожд\w*", re.I)


def outcome_mislabel(text: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """P_pct/CI — вероятность клинической БЕРЕМЕННОСТИ на перенос.

    Называть её живорождением клинически неверно: это разные исходы.
    Слово допустимо только внутри цитаты гайдлайна (там оно встречается
    в исходном тексте рекомендаций), поэтому строки со ссылкой пропускаем.
    """
    # Собираем числа, которые ЕСТЬ вероятность беременности: главный прогноз,
    # границы ДИ и прогнозы GAT-аналогов.
    preg_nums = set()
    for key in ("P_pct", "CI_low_pct", "CI_high_pct"):
        v = _path(ctx, "main_forecast", key)
        if isinstance(v, (int, float)):
            preg_nums.add(round(float(v), 1))
    ha = ctx.get("historical_analogues") or {}
    for v in (ha.get("GNN_P_median_pct"), (ha.get("closest") or {}).get("GNN_P_pct")):
        if isinstance(v, (int, float)):
            preg_nums.add(round(float(v), 1))
    if not preg_nums:
        return {"n": 0, "hits": []}

    hits = []
    for sent in re.split(r"(?<=[.!?])\s+|\n", text):
        if not _LIVEBIRTH.search(sent):
            continue
        if "[" in sent and "]" in sent:
            continue          # внутри цитаты источника слово допустимо
        for m in _NUM.finditer(sent):
            try:
                v = round(float(m.group(1).replace(",", ".")), 1)
            except ValueError:
                continue
            if any(abs(v - p) <= 0.15 for p in preg_nums):
                hits.append(sent.strip()[:140])
                break
    return {"n": len(hits), "hits": hits[:5]}


def unsupported_inference(text: str) -> Dict[str, Any]:
    hits = []
    for m in _UNSUPPORTED.finditer(text):
        hits.append(text[max(0, m.start() - 60):m.end() + 60].replace("\n", " ").strip())
    return {"n": len(hits), "hits": hits[:6]}


# ──────────────────────────────────────────────────────────────────────────
def score_run(run: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    text, ctx, flags = run["text"], case["ctx"], case["flags"]
    words = len(re.findall(r"\b[\w–-]+\b", text))
    g = numeric_grounding(text, ctx)
    r = reclassification(text, flags)
    d = directive_language(text)
    c = coverage(text, ctx)
    lk = leakage(text)
    fa = FA.check_faithfulness(text, ctx)
    ns = null_safety(text, ctx)
    bd = label_binding(text, ctx)
    uc = unit_confusion(text, ctx)
    om = outcome_mislabel(text, ctx)
    ui = unsupported_inference(text)
    tps = round(run["out_tokens"] / run["eval_s"], 2) if run.get("eval_s") else None

    # композитный балл 0..100 — инженерный светофор, не клиническая оценка
    penalty = (12 * r["n"] + 8 * d["n"] + 10 * ns["n"]
               + 6 * g["n_ungrounded"] + 4 * lk["n_snake"] + 20 * lk["n_think"]
               + 10 * bd["n"] + 10 * uc["n"] + 10 * om["n"] + 5 * ui["n"]
               + 10 * (0 if fa.get("ok", True) else 1))
    score = max(0, round(100 * c["rate"] - penalty))

    return {
        "case": run["case"], "model": run["model"], "style": run["style"],
        "words": words, "out_tokens": run.get("out_tokens"),
        "wall_s": run.get("wall_s"), "tok_s": tps,
        "truncated": run.get("done_reason") == "length",
        "grounding_rate": g["rate"], "n_ungrounded": g["n_ungrounded"],
        "ungrounded_examples": g["examples"],
        "reclass_n": r["n"], "reclass": r["violations"],
        "directive_n": d["n"], "directive": d["hits"], "soft_n": d["n_soft"],
        "coverage_rate": c["rate"], "sections_missing": c["missing"],
        "sections_na": c["not_applicable"],
        "leak_snake": lk["snake_keys"], "leak_think": lk["n_think"],
        "leak_latin": lk["latin_words"],
        "faithfulness_ok": fa.get("ok"), "faithfulness": FA.summary(fa),
        "nullsafe_n": ns["n"], "nullsafe": ns["violations"],
        "binding_n": bd["n"], "binding": bd["violations"],
        "unit_n": uc["n"], "unit": uc["violations"],
        "outcome_n": om["n"], "outcome": om["hits"],
        "unsupported_n": ui["n"], "unsupported": ui["hits"],
        "score": score,
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    runs_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs.jsonl")
    with open(os.path.join(HERE, "contexts.json"), encoding="utf-8") as f:
        cases = {c["id"]: c for c in json.load(f)}
    rows = []
    with open(runs_path, encoding="utf-8") as f:
        for line in f:
            run = json.loads(line)
            if run.get("error") or not run.get("text"):
                continue
            case = cases[run["case"]]
            # Варианты с меткой «+compact» видели сжатый контекст — сверяем с ним,
            # иначе покрытие и null-safety считаются против блоков, которых
            # модель не получала.
            if "+compact" in (run.get("style") or ""):
                sys.path.insert(0, HERE)
                from compact_ctx import compact as _cc
                cctx, _ = _cc(case["ctx"])
                case = {**case, "ctx": cctx}
            rows.append(score_run(run, case))
    stem = os.path.splitext(os.path.basename(runs_path))[0]
    out = os.path.join(HERE, f"scores_{stem}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    hdr = (f"{'case':30s} {'model':18s} {'style':16s} {'w':>4s} {'min':>5s} "
           f"{'cov':>5s} {'grnd':>5s} {'RC':>3s} {'DIR':>3s} {'NS':>3s} "
           f"{'BND':>3s} {'UNI':>3s} {'OUT':>3s} {'INF':>3s} {'LEAK':>4s} {'F':>2s} {'SCORE':>5s}")
    print(hdr); print("─" * len(hdr))
    for r in rows:
        mins = f"{(r['wall_s'] or 0)/60:.1f}"
        print(f"{r['case']:30s} {r['model']:18s} {r['style']:16s} "
              f"{r['words']:4d} {mins:>5s} "
              f"{r['coverage_rate']:5.2f} {r['grounding_rate']:5.2f} "
              f"{r['reclass_n']:3d} {r['directive_n']:3d} {r['nullsafe_n']:3d} "
              f"{r['binding_n']:3d} {r['unit_n']:3d} {r['outcome_n']:3d} {r['unsupported_n']:3d} "
              f"{len(r['leak_snake']) + r['leak_think']:4d} "
              f"{'ok' if r['faithfulness_ok'] else 'X':>2s} {r['score']:5d}")
    print(f"\nsaved {out}  ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
