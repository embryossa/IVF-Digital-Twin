# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
compact_ctx.py — прототип компактного контекста нарратора.

Мотивация (измерено на этом ПК): обработка промпта идёт со скоростью 6.2 ток/с,
то есть КАЖДАЯ 1000 токенов контекста стоит ~2.7 минуты ожидания врача.
В контексте S03 (3066 токенов) на блок protocol_guidance приходится 1531 токен,
из них 1049 — на `гайдлайны`, где 5 уникальных источников повторяются полной
библиографией в каждом утверждении.

Что делает компактор (ничего не выдумывает, только сжимает представление):
  1. citation → короткий тег («ESHRE 2025»), полная библиография выносится
     ОДИН раз в легенду `источники`;
  2. число утверждений ограничивается (по умолчанию 4 — retrieval уже
     отранжировал их по силе совпадения);
  3. `_role` и `инструкция_нарратору` убираются из per-patient контекста —
     их место в системном промпте (он стабилен и кэшируется);
  4. пустые/None-поля отбрасываются.

Функция обратима по смыслу: врач видит те же утверждения и те же источники.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Tuple

_SRC_PAT = [
    (re.compile(r"ESHRE.*?(\d{4})", re.S), "ESHRE {}"),
    (re.compile(r"ASRM.*?(\d{4})", re.S), "ASRM {}"),
    (re.compile(r"British Fertility Society.*?(\d{4})", re.S), "BFS {}"),
    (re.compile(r"POSEIDON", re.I), "POSEIDON"),
    (re.compile(r"OPTIMIST", re.I), "OPTIMIST"),
    (re.compile(r"^([A-Z][A-Za-z-]+)[^.]*?(\d{4})"), "{} {}"),
]


_JOURNAL = re.compile(r"\b(J Assist Reprod Genet|Hum Reprod Open|Hum Reprod|"
                      r"Fertil Steril|Reprod Biomed Online|Hum Fertil)\b")
_ANY_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def short_tag(citation: str) -> str:
    c = (citation or "").strip()
    for pat, tmpl in _SRC_PAT:
        m = pat.search(c)
        if m:
            try:
                return tmpl.format(*m.groups())
            except (IndexError, KeyError):
                return tmpl
    # Нет узнаваемой организации — берём журнал + год («J Assist Reprod Genet 2021»).
    ym = _ANY_YEAR.search(c)
    jm = _JOURNAL.search(c)
    if jm and ym:
        return f"{jm.group(1)} {ym.group(0)}"
    if ym:
        first = next((w for w in c.split() if w[:1].isupper()), "источник")
        return f"{first.strip('.,')} {ym.group(0)}"
    return (c[:28] + "…") if len(c) > 28 else c


def _prune(obj: Any) -> Any:
    """Рекурсивно убирает None и пустые контейнеры."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            pv = _prune(v)
            if pv is None or pv == {} or pv == []:
                continue
            out[k] = pv
        return out
    if isinstance(obj, list):
        return [_prune(v) for v in obj if _prune(v) is not None]
    return obj


def compact(ctx: Dict[str, Any], max_guidelines: int = 4,
            prune_nulls: bool = True) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """→ (компактный контекст, легенда {тег: полная библиография})."""
    c = copy.deepcopy(ctx)
    legend: Dict[str, str] = {}

    pg = c.get("protocol_guidance")
    if isinstance(pg, dict):
        pg.pop("_role", None)
        pg.pop("инструкция_нарратору", None)
        gl: List[Dict[str, Any]] = pg.get("гайдлайны") or []
        new_gl = []
        for st in gl[:max_guidelines]:
            tag = short_tag(st.get("citation", ""))
            legend.setdefault(tag, st.get("citation", ""))
            new_gl.append({"text": st.get("text"),
                           "level": st.get("evidence_level"),
                           "src": tag})
        pg["гайдлайны"] = new_gl
        # согласование_СГЯ: длинный комментарий нужен только при расхождении
        rec = pg.get("согласование_СГЯ")
        if isinstance(rec, dict) and rec.get("согласуются") is True:
            pg["согласование_СГЯ"] = {"обе_линзы": rec.get("вероятностный"),
                                      "согласуются": True}
        nom = pg.get("номограмма")
        if isinstance(nom, dict):
            nom.pop("версия_параметров", None)

    if prune_nulls:
        c = _prune(c)
    return c, legend


if __name__ == "__main__":
    import os
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    HERE = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(HERE, "contexts.json"), encoding="utf-8") as f:
        cases = json.load(f)

    CPT = 2.97      # символов на токен (замер на этом стеке)
    PROMPT_TPS = 6.22
    print(f"{'case':30s} {'полн.симв':>9s} {'комп.симв':>9s} {'-%':>5s} "
          f"{'~ток сэконом':>12s} {'~сек':>6s}")
    print("─" * 78)
    tot_saved = 0
    for c in cases:
        full = json.dumps(c["ctx"], ensure_ascii=False, indent=2)
        comp, legend = compact(c["ctx"])
        cj = json.dumps(comp, ensure_ascii=False, indent=2)
        saved_tok = (len(full) - len(cj)) / CPT
        tot_saved += saved_tok
        print(f"{c['id']:30s} {len(full):9d} {len(cj):9d} "
              f"{100 * (1 - len(cj) / len(full)):4.0f}% {saved_tok:12.0f} "
              f"{saved_tok / PROMPT_TPS:6.0f}")
    print("─" * 78)
    print(f"средняя экономия: {tot_saved / len(cases):.0f} токенов ≈ "
          f"{tot_saved / len(cases) / PROMPT_TPS / 60:.1f} мин на пациента")

    comp, legend = compact(cases[0]["ctx"])
    print("\nЛегенда источников (выносится в системный промпт один раз):")
    for k, v in legend.items():
        print(f"  {k:16s} → {v[:90]}")
