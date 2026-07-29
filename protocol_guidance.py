# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
protocol_guidance.py — IVF Digital Twin
Bridge between app globals (g) and the two new layers:
  stim_protocol.py  (deterministic dose/protocol calculator)
  guideline_rag.py  (offline guideline retrieval)

It produces ONE pre-classified context block, in the same spirit as
build_interpretation_flags(): everything is computed deterministically in Python
BEFORE the LLM is called. The narrator only explains and cites this block.

Why a separate module
----------------------
Keeping this here means llm_consultant.py needs only a 3-line import+call to
gain protocol/guideline grounding, and the whole feature degrades gracefully:
if stim_protocol/guideline_rag are absent, build_protocol_guidance() returns
None and the narrator behaves exactly as before.

Dependencies: standard library (+ the two local modules).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _f(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _retrieval_query(stim_dict: Dict[str, Any]) -> str:
    """Short natural-language query for the optional semantic rerank."""
    return (f"{stim_dict.get('response_phenotype', '')} "
            f"риск СГЯ {stim_dict.get('ohss_risk', '')} "
            f"стартовая доза гонадотропина протокол стимуляции")


def _probabilistic_ohss(g: Dict[str, Any]) -> Optional[str]:
    """Probabilistic OHSS tier from the MC pipeline (res['ohss']), using the SAME
    thresholds as build_interpretation_flags. Reads res only — BEFE untouched."""
    res = g.get("res") or {}
    ohss = res.get("ohss") or {}
    p_sev = _f(ohss.get("p_severe_ohss"))
    p_mod = _f(ohss.get("p_moderate_ohss"))
    if p_sev is None and p_mod is None:
        return None
    if p_sev is not None and p_sev >= 0.10:
        return "high"
    if (p_sev is not None and p_sev >= 0.05) or (p_mod is not None and p_mod >= 0.20):
        return "moderate"
    return "low"


def _reconcile_ohss(prob_level: Optional[str], ort_level: str) -> Dict[str, Any]:
    """Compare the two OHSS lenses. 'elevated' (ORT) ~ 'moderate' (probabilistic)."""
    if prob_level is None:
        return {"вероятностный": None, "по_резерву": ort_level,
                "согласуются": None,
                "комментарий": "вероятностная оценка СГЯ недоступна для этого расчёта"}
    norm = {"elevated": "moderate"}
    agree = norm.get(prob_level, prob_level) == norm.get(ort_level, ort_level)
    if agree:
        comment = f"обе оценки риска СГЯ совпадают ({prob_level})"
    else:
        comment = (
            f"вероятностная модель (по симуляции цикла) — {prob_level}; "
            f"оценка по овариальному резерву (AMH/AFC) — {ort_level}. "
            "Расхождение ожидаемо: это две линзы — апостериорная по исходам цикла "
            "и априорная по резерву; при расхождении ориентируйся на более "
            "осторожную из двух."
        )
    return {"вероятностный": prob_level, "по_резерву": ort_level,
            "согласуются": agree, "комментарий": comment}


def build_protocol_guidance(g: Dict[str, Any],
                            use_semantic: bool = True,
                            max_guidelines: int = 4) -> Optional[Dict[str, Any]]:
    """Compute the deterministic protocol block + matching guideline statements.

    Returns a dict for insertion into the narrator context, or None when the
    required inputs/modules are unavailable (graceful fallback).
    """
    age = _f(g.get("age"))
    amh = _f(g.get("amh"))
    afc = g.get("afc")
    if age is None or amh is None or afc is None:
        return None  # not enough to dose-stratify → stay silent

    try:
        from stim_protocol import compute_stim, StimInput
    except Exception:
        return None

    try:
        inp = StimInput(
            age=age,
            amh_ng_ml=amh,
            afc=int(afc),
            bmi=_f(g.get("bmi")) or 24.0,
            weight_kg=_f(g.get("weight_kg")),          # optional; None is fine
            protocol_pref=str(g.get("protocol_pref", "auto") or "auto"),
        )
        stim = compute_stim(inp)
        stim_dict = stim.as_dict()
    except Exception:
        return None

    # Guideline grounding (rule-keyed primary; semantic rerank if available).
    guidelines: List[Dict[str, Any]] = []
    try:
        from guideline_rag import retrieve
        query = _retrieval_query(stim_dict) if use_semantic else None
        guidelines = retrieve(stim.situation_keys, query=query, max_n=max_guidelines)
    except Exception:
        guidelines = []

    # [FIX] Do NOT expose the internal `id` to the narrator: the model sometimes
    # cited it verbatim (e.g. "[ASRM_agonist_trigger]"). The narrator only needs
    # the human citation. `id` stays available upstream for retrieval/audit.
    #
    # [PERF] The full bibliography was repeated for every statement — on a CPU
    # box that is ~200 characters of prompt per statement, re-read on every
    # patient (measured: prompt eval ≈ 0.17 s/token). We now hand the narrator
    # the short corpus key as `src` ("ESHRE2025") and return the full
    # bibliography ONCE in `источники`, for the UI/report to render.
    guidelines_for_llm = []
    sources_legend: Dict[str, str] = {}
    for s in guidelines:
        tag = s.get("source") or ""
        if tag and s.get("citation"):
            sources_legend.setdefault(tag, s["citation"])
        guidelines_for_llm.append({
            "text": s.get("text"),
            "evidence_level": s.get("evidence_level"),
            "src": tag or s.get("citation", ""),
        })

    # [IMP] Reconcile the two OHSS signals without touching BEFE: probabilistic
    # (res['ohss'], MC pipeline) vs reserve-based (nomogram).
    ohss_reconciliation = _reconcile_ohss(_probabilistic_ohss(g), stim_dict["ohss_risk"])

    # [PERF] `_role` и `инструкция_нарратору` убраны из per-patient блока:
    # это статический текст, он не меняется от пациента к пациенту и его место
    # в системном промпте (llm_consultant._SYSTEM_NARRATOR, раздел про протокол).
    # Здесь остаются только данные.
    return {
        "номограмма": {
            "фенотип_ответа":        stim_dict["response_phenotype"],
            "обоснование_фенотипа":  stim_dict["phenotype_reasons"],
            "риск_СГЯ":              stim_dict["ohss_risk"],
            "протокол":              stim_dict["protocol_type"],
            "целевой_выход_ооцитов": stim_dict["target_oocyte_yield"],
            "стартовая_доза_МЕ":     stim_dict["suggested_start_dose_iu"],
            "оговорки_по_дозе":      stim_dict["dose_caveats"],
            "рычаги_профилактики":   stim_dict["mitigation_levers"],
            "фоллитропин_дельта_мкг": stim_dict["follitropin_delta_ug"],
            "версия_параметров":     stim_dict["param_version"],
        },
        "согласование_СГЯ": ohss_reconciliation,
        "гайдлайны": guidelines_for_llm,
        # Полная библиография — ОДИН раз на весь блок, а не в каждом утверждении.
        # Нарратору в тексте нужен короткий тег из `src`; это поле существует для
        # UI/PDF-отчёта, где ссылку раскрывают целиком.
        "источники": sources_legend,
    }


if __name__ == "__main__":
    import json
    demo_g = {"age": 31, "amh": 5.6, "afc": 26, "bmi": 23}
    block = build_protocol_guidance(demo_g, use_semantic=False)
    print(json.dumps(block, ensure_ascii=False, indent=2))
