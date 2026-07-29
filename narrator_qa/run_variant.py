# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
run_variant.py — A/B прогон альтернативных системных промптов на тех же контекстах.

Отличие от run_eval.py: системный промпт берётся из файла prompts/<name>.md,
всё остальное (контекст, вопрос, параметры) идентично продакшену.

Запуск:
    python run_variant.py --prompt v4_core --cases S01... --model gemma4:12b-it-qat
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
DT_DIR = r"C:\Users\User\Desktop\IVF\AI\IVF Digital Twin Pro\IVF Digital Twin"
if DT_DIR not in sys.path:
    sys.path.insert(0, DT_DIR)

import requests  # noqa: E402
import llm_consultant as LC  # noqa: E402

OUT = os.path.join(HERE, "runs_variant.jsonl")
_S = requests.Session()
_S.trust_env = False


def build_messages(ctx: Dict[str, Any], question: str, system: str):
    ctx_json = json.dumps(ctx, ensure_ascii=False, indent=2)
    user = ("/no_think\n"
            "Контекст пациента (использовать ТОЛЬКО эти числа):\n"
            f"```json\n{ctx_json}\n```\n\nЗапрос: {question}")
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="имя файла в prompts/ без .md")
    ap.add_argument("--model", default="gemma4:12b-it-qat")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--num-predict", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=0.35)
    ap.add_argument("--question", default=None,
                    help="переопределить запрос пользователя (по умолчанию — концизный из проекта)")
    ap.add_argument("--compact", action="store_true",
                    help="сжать контекст (compact_ctx): короткие теги источников, ≤4 гайдлайна")
    ap.add_argument("--max-guidelines", type=int, default=4)
    ap.add_argument("--tag", default=None, help="суффикс метки style для различения прогонов")
    a = ap.parse_args()

    with open(os.path.join(HERE, "prompts", a.prompt + ".md"), encoding="utf-8") as f:
        system = f.read().strip()
    with open(os.path.join(HERE, "contexts.json"), encoding="utf-8") as f:
        cases = json.load(f)
    if a.cases:
        cases = [c for c in cases if c["id"] in a.cases]

    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["case"], r["model"], r["style"]))
                except Exception:
                    pass

    question = a.question or (LC._QUESTION_CONCISE + LC._STYLE_HINT["concise"])
    style = f"variant:{a.prompt}" + ("+compact" if a.compact else "") + \
            (f"+{a.tag}" if a.tag else "")

    if a.compact:
        from compact_ctx import compact
        legend_all = {}
        for c in cases:
            c["ctx"], lg = compact(c["ctx"], max_guidelines=a.max_guidelines)
            legend_all.update(lg)
        if legend_all:
            system += ("\n\nЛЕГЕНДА ИСТОЧНИКОВ (используй короткий тег в тексте):\n"
                       + "\n".join(f"  {k} — {v}" for k, v in legend_all.items()))

    todo = [c for c in cases if (c["id"], a.model, style) not in done]
    print(f"[plan] {len(todo)} runs · prompt={a.prompt} · model={a.model}", flush=True)

    for i, c in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {c['id']} …", flush=True)
        t0 = time.time()
        try:
            r = _S.post(f"{LC.OLLAMA_HOST}/api/chat",
                        json={"model": a.model,
                              "messages": build_messages(c["ctx"], question, system),
                              "stream": False, "think": False,
                              "options": {"temperature": a.temperature,
                                          "num_predict": a.num_predict},
                              "keep_alive": "1h"},
                        timeout=(5, 3600), proxies={"http": None, "https": None})
            r.raise_for_status()
            d = r.json()
            rec = {"case": c["id"], "title": c["title"], "probes": c["probes"],
                   "model": a.model, "style": style,
                   "prompt_name": a.prompt,
                   "temperature": a.temperature, "num_predict": a.num_predict,
                   "text": LC._strip_thinking(d.get("message", {}).get("content", "") or ""),
                   "wall_s": round(time.time() - t0, 1),
                   "prompt_tokens": d.get("prompt_eval_count"),
                   "prompt_eval_s": round((d.get("prompt_eval_duration") or 0) / 1e9, 1),
                   "out_tokens": d.get("eval_count"),
                   "eval_s": round((d.get("eval_duration") or 0) / 1e9, 1),
                   "done_reason": d.get("done_reason"),
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            tps = rec["out_tokens"] / rec["eval_s"] if rec["eval_s"] else 0
            print(f"     ok {rec['wall_s']}s in={rec['prompt_tokens']} "
                  f"out={rec['out_tokens']} {tps:.2f} tok/s", flush=True)
        except Exception as e:
            rec = {"case": c["id"], "model": a.model, "style": style,
                   "prompt_name": a.prompt, "error": repr(e)}
            print(f"     ERROR {e!r}", flush=True)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
