# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
faithfulness.py — IVF Digital Twin
Deterministic post-generation check: does the narrator text stay faithful to the
grounding it was given? Runs in milliseconds, no model call.

It verifies two high-risk hallucination surfaces in the protocol/guideline block:
  1. CITATIONS — no bracketed token equals an internal statement id (the leak we
     fixed), and bracketed citation-like tokens share a source/year with the
     citations actually supplied.
  2. DOSE      — every "N МЕ" / "N–M МЕ" figure in the text matches the dose band
     computed by the nomogram (no invented dose numbers).

Returns a small report dict suitable for the audit log. Never raises.

Dependencies: standard library only.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

_BRACKET = re.compile(r"\[([^\[\]]{2,120})\]")
# [FIX] Модель нередко ставит ссылку в КРУГЛЫХ скобках («(ESHRE ... 2025)»).
# Раньше проверялись только квадратные, и вся цитатная половина контроля молча
# не срабатывала, возвращая ok. Круглые разбираем только если внутри есть год
# или акроним — иначе поймали бы обычные пояснения в скобках.
_PAREN = re.compile(r"\(([^()]{2,120})\)")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_ACRONYM = re.compile(r"\b[A-ZЕА-Я]{3,}\b")           # ESHRE, ASRM, BFS, POSEIDON…
# [FIX] Единица дозы зависит от локали нарратора: русский пишет «МЕ»,
# английский рантайм (legacy_en_runtime) — «IU». До этой правки проверка дозы
# для английской сборки не срабатывала НИ РАЗУ.
_UNIT = r"(?:МЕ|IU)"
_DOSE_RANGE = re.compile(rf"(\d{{2,3}})\s*[–—-]\s*(\d{{2,3}})\s*{_UNIT}\b")
_DOSE_SINGLE = re.compile(rf"(?<![–—\-\d])(\d{{2,3}})\s*{_UNIT}\b")


def _statement_ids() -> List[str]:
    try:
        import guideline_rag
        pack = guideline_rag.load_pack()
        return [s.get("id", "") for s in pack.get("statements", [])]
    except Exception:
        return []


# Короткий тег корпуса склеен: «ESHRE2025», «Arce2014». Без разделения ни _YEAR,
# ни _ACRONYM его не видят (нет границы слова между буквой и цифрой), и множество
# разрешённых токенов оказывается пустым — тогда проверка неизвестных ссылок
# молча отключается. Разделяем буквы и цифры перед разбором.
_GLUED = re.compile(r"(?<=[A-Za-zА-Яа-я])(?=\d)|(?<=\d)(?=[A-Za-zА-Яа-я])")


def _split_glued(s: str) -> str:
    return _GLUED.sub(" ", s or "")


def _allowed_tokens(citations: List[str]) -> set:
    toks = set()
    for c in citations:
        c = _split_glued(c)
        toks.update(m.group(0) for m in _YEAR.finditer(c))
        toks.update(m.group(0) for m in _ACRONYM.finditer(c))
    return toks


def check_faithfulness(text: str,
                       ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Check narrator `text` against the protocol_guidance block in `ctx`."""
    report: Dict[str, Any] = {
        "checked": False, "ok": True,
        "leaked_ids": [], "unknown_citations": [], "dose_mismatches": [],
    }
    if not text or not isinstance(ctx, dict):
        return report
    pg = ctx.get("protocol_guidance")
    if not pg:
        return report
    report["checked"] = True

    guidelines = pg.get("гайдлайны", []) or []
    # `citation` — полная библиография (старый формат), `src` — короткий тег
    # корпуса (новый формат, «ESHRE2025»). Разрешаем оба, плюс легенду
    # `источники`, если она рядом: нарратор теперь печатает именно тег.
    citations = [g.get("citation", "") or g.get("src", "") for g in guidelines]
    citations += [f"{k} {v}" for k, v in (pg.get("источники") or {}).items()]
    allowed = _allowed_tokens(citations)
    known_ids = set(_statement_ids())

    brackets = [m.group(1).strip() for m in _BRACKET.finditer(text)]
    # Круглые скобки — только если внутри есть год или акроним (см. _PAREN).
    for m in _PAREN.finditer(text):
        cand = m.group(1).strip()
        if _YEAR.search(cand) or _ACRONYM.search(cand):
            brackets.append(cand)
    for b in brackets:
        # 1a. exact internal id leak (high precision)
        if b in known_ids:
            report["leaked_ids"].append(b)
            continue
        # 1b. citation-looking bracket that shares no source/year with corpus
        b_split = _split_glued(b)          # «[ESHRE2025]» → «ESHRE 2025»
        has_year = bool(_YEAR.search(b_split))
        has_acro = bool(_ACRONYM.search(b_split))
        if has_year or has_acro:
            b_tokens = set(m.group(0) for m in _YEAR.finditer(b_split))
            b_tokens |= set(m.group(0) for m in _ACRONYM.finditer(b_split))
            if allowed and b_tokens.isdisjoint(allowed):
                report["unknown_citations"].append(b)

    # 2. dose figures must match the nomogram band
    band = (pg.get("номограмма", {}) or {}).get("стартовая_доза_МЕ")
    if isinstance(band, (list, tuple)) and len(band) == 2:
        lo, hi = int(band[0]), int(band[1])
        # Единицу в сообщении не фиксируем: текст мог быть на русском (МЕ) или
        # английском (IU), и подставлять чужую было бы неверно.
        for m in _DOSE_RANGE.finditer(text):
            a, b2 = int(m.group(1)), int(m.group(2))
            if not (abs(a - lo) <= 1 and abs(b2 - hi) <= 1):
                report["dose_mismatches"].append(f"{a}–{b2} (ожидалось {lo}–{hi})")
        for m in _DOSE_SINGLE.finditer(text):
            v = int(m.group(1))
            if not (lo - 1 <= v <= hi + 1):
                report["dose_mismatches"].append(f"{v} (вне {lo}–{hi})")

    report["ok"] = not (report["leaked_ids"]
                        or report["unknown_citations"]
                        or report["dose_mismatches"])
    return report


def summary(report: Dict[str, Any]) -> str:
    if not report.get("checked"):
        return "faithfulness: n/a (нет блока протокола)"
    if report.get("ok"):
        return "faithfulness: ok"
    parts = []
    if report["leaked_ids"]:
        parts.append("утечка id: " + ", ".join(report["leaked_ids"]))
    if report["unknown_citations"]:
        parts.append("неизвестные ссылки: " + ", ".join(report["unknown_citations"]))
    if report["dose_mismatches"]:
        parts.append("доза вне контекста: " + "; ".join(report["dose_mismatches"]))
    return "faithfulness: FAIL — " + " | ".join(parts)


if __name__ == "__main__":
    import sys
    try:                       # консоль Windows по умолчанию cp1251
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ctx = {"protocol_guidance": {
        "номограмма": {"стартовая_доза_МЕ": [100, 150]},
        "гайдлайны": [
            {"citation": "ESHRE ... Hum Reprod Open, 2025"},
            {"citation": "Practice Committee ASRM. Fertil Steril 2024"},
        ],
    }}
    good = "Доза 100–150 МЕ [ESHRE ... 2025]; антагонист [ASRM ... 2024]."
    bad = "Доза 200–250 МЕ [ASRM_agonist_trigger]; см. [Foobar 1999]."
    print("GOOD →", summary(check_faithfulness(good, ctx)))
    print("BAD  →", summary(check_faithfulness(bad, ctx)))
