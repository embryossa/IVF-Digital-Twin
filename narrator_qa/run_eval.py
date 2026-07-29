# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
run_eval.py — прогон сценариев через локальную Ollama с ПРОДАКШН-промптом DT.

Использует llm_consultant._SYSTEM_NARRATOR и _build_messages без изменений,
чтобы измерять именно то, что видит врач в приложении.

Резюмируемый: уже выполненные (case, model, style) пропускаются.
Пишет JSONL инкрементально — прогон можно прервать и продолжить.

Запуск:
    python run_eval.py --models gemma4:12b-it-qat --styles narrative
    python run_eval.py --plan            # только показать матрицу и оценку времени
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
DT_DIR = r"C:\Users\User\Desktop\IVF\AI\IVF Digital Twin Pro\IVF Digital Twin"
if DT_DIR not in sys.path:
    sys.path.insert(0, DT_DIR)

import requests  # noqa: E402
import llm_consultant as LC  # noqa: E402

OUT = os.path.join(HERE, "runs.jsonl")
CTX_FILE = os.path.join(HERE, "contexts.json")

_S = requests.Session()
_S.trust_env = False


def load_cases() -> List[Dict[str, Any]]:
    with open(CTX_FILE, encoding="utf-8") as f:
        return json.load(f)


def done_keys() -> set:
    if not os.path.exists(OUT):
        return set()
    ks = set()
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                ks.add((r["case"], r["model"], r["style"]))
            except Exception:
                pass
    return ks


def run_one(case: Dict[str, Any], model: str, style: str,
            temperature: float, num_predict: int) -> Dict[str, Any]:
    question = (LC._QUESTION_CONCISE if style == "concise" else LC._DEFAULT_QUESTION)
    question += LC._STYLE_HINT.get(style, "")
    messages = LC._build_messages(case["ctx"], question)

    # Потоковый режим: это продакшн-путь UI (consult_stream) и, в отличие от
    # блокирующего вызова, соединение не «засыпает» на многоминутной генерации.
    # Модели без поддержки thinking (medgemma1.5) на явный think=False могут
    # ответить 4xx — тогда повторяем запрос без этого поля.
    def _post(with_think: bool):
        payload = {"model": model, "messages": messages, "stream": True,
                   "options": {"temperature": temperature,
                               "num_predict": num_predict},
                   "keep_alive": "1h"}
        if with_think:
            payload["think"] = False
        return _S.post(f"{LC.OLLAMA_HOST}/api/chat", json=payload, stream=True,
                       timeout=(5, 900), proxies={"http": None, "https": None})

    t0 = time.time()
    buf, d, ttft = [], {}, None
    resp = _post(True)
    if resp.status_code >= 400:
        resp.close()
        t0 = time.time()
        resp = _post(False)
    with resp as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            piece = (chunk.get("message") or {}).get("content", "")
            if piece:
                if ttft is None:
                    ttft = time.time() - t0
                buf.append(piece)
            if chunk.get("done"):
                d = chunk
    wall = time.time() - t0
    text = LC._strip_thinking("".join(buf))

    return {
        "case": case["id"], "title": case["title"], "probes": case["probes"],
        "model": model, "style": style,
        "temperature": temperature, "num_predict": num_predict,
        "text": text,
        "wall_s": round(wall, 1),
        "ttft_s": round(ttft, 1) if ttft else None,
        "prompt_tokens": d.get("prompt_eval_count"),
        "prompt_eval_s": round((d.get("prompt_eval_duration") or 0) / 1e9, 1),
        "out_tokens": d.get("eval_count"),
        "eval_s": round((d.get("eval_duration") or 0) / 1e9, 1),
        "load_s": round((d.get("load_duration") or 0) / 1e9, 1),
        "done_reason": d.get("done_reason"),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma4:12b-it-qat"])
    ap.add_argument("--styles", nargs="+", default=["narrative"])
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--temperature", type=float, default=LC._NARRATOR_TEMP)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()

    cases = load_cases()
    if a.cases:
        cases = [c for c in cases if c["id"] in a.cases]
    already = done_keys()

    todo = [(c, m, s) for m in a.models for s in a.styles for c in cases
            if (c["id"], m, s) not in already]

    if a.plan:
        print(f"cases={len(cases)} models={a.models} styles={a.styles}")
        print(f"already done: {len(already)}  |  to run: {len(todo)}")
        for c, m, s in todo:
            print(f"  {c['id']:32s} {m:22s} {s}")
        return 0

    print(f"[plan] {len(todo)} runs (skipping {len(already)} done)", flush=True)
    for i, (c, m, s) in enumerate(todo, 1):
        np_ = LC._STYLE_NUM_PREDICT.get(s, 2500)
        print(f"[{i}/{len(todo)}] {c['id']} · {m} · {s} …", flush=True)
        try:
            rec = run_one(c, m, s, a.temperature, np_)
        except Exception as e:
            rec = {"case": c["id"], "model": m, "style": s, "error": repr(e),
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            print(f"     ERROR {e!r}", flush=True)
        else:
            tps = (rec["out_tokens"] / rec["eval_s"]) if rec.get("eval_s") else 0
            print(f"     ok {rec['wall_s']}s  in={rec['prompt_tokens']}tok"
                  f"({rec['prompt_eval_s']}s) out={rec['out_tokens']}tok "
                  f"{tps:.2f} tok/s  reason={rec['done_reason']}", flush=True)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
