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
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_ACRONYM = re.compile(r"\b[A-ZЕА-Я]{3,}\b")           # ESHRE, ASRM, BFS, POSEIDON…
_DOSE_RANGE = re.compile(r"(\d{2,3})\s*[–—-]\s*(\d{2,3})\s*МЕ")
_DOSE_SINGLE = re.compile(r"(?<![–—\-\d])(\d{2,3})\s*МЕ")


def _statement_ids() -> List[str]:
    try:
        import guideline_rag
        pack = guideline_rag.load_pack()
        return [s.get("id", "") for s in pack.get("statements", [])]
    except Exception:
        return []


def _allowed_tokens(citations: List[str]) -> set:
    toks = set()
    for c in citations:
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
    citations = [g.get("citation", "") for g in guidelines]
    allowed = _allowed_tokens(citations)
    known_ids = set(_statement_ids())

    brackets = [m.group(1).strip() for m in _BRACKET.finditer(text)]
    for b in brackets:
        # 1a. exact internal id leak (high precision)
        if b in known_ids:
            report["leaked_ids"].append(b)
            continue
        # 1b. citation-looking bracket that shares no source/year with corpus
        has_year = bool(_YEAR.search(b))
        has_acro = bool(_ACRONYM.search(b))
        if has_year or has_acro:
            b_tokens = set(m.group(0) for m in _YEAR.finditer(b))
            b_tokens |= set(m.group(0) for m in _ACRONYM.finditer(b))
            if allowed and b_tokens.isdisjoint(allowed):
                report["unknown_citations"].append(b)

    # 2. dose figures must match the nomogram band
    band = (pg.get("номограмма", {}) or {}).get("стартовая_доза_МЕ")
    if isinstance(band, (list, tuple)) and len(band) == 2:
        lo, hi = int(band[0]), int(band[1])
        for m in _DOSE_RANGE.finditer(text):
            a, b2 = int(m.group(1)), int(m.group(2))
            if not (abs(a - lo) <= 1 and abs(b2 - hi) <= 1):
                report["dose_mismatches"].append(f"{a}–{b2} МЕ (ожидалось {lo}–{hi})")
        for m in _DOSE_SINGLE.finditer(text):
            v = int(m.group(1))
            if not (lo - 1 <= v <= hi + 1):
                report["dose_mismatches"].append(f"{v} МЕ (вне {lo}–{hi})")

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
