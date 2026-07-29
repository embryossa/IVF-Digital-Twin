# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""probe.py — потоковый замер: видно, как реально идёт генерация (токенов/с во времени)."""
from __future__ import annotations
import json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DT_DIR = r"C:\Users\User\Desktop\IVF\AI\IVF Digital Twin Pro\IVF Digital Twin"
sys.path.insert(0, DT_DIR)
import requests, llm_consultant as LC  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
model = sys.argv[1] if len(sys.argv) > 1 else "gemma4:12b-it-qat"
case = sys.argv[2] if len(sys.argv) > 2 else "S01_high_responder_ohss"
npred = int(sys.argv[3]) if len(sys.argv) > 3 else 700

cases = {c["id"]: c for c in json.load(open(os.path.join(HERE, "contexts.json"), encoding="utf-8"))}
ctx = cases[case]["ctx"]
q = LC._QUESTION_CONCISE + LC._STYLE_HINT["concise"]
msgs = LC._build_messages(ctx, q)

S = requests.Session(); S.trust_env = False
t0 = time.time(); first = None; n = 0; buf = []
print(f"model={model} case={case} num_predict={npred}", flush=True)
with S.post(f"{LC.OLLAMA_HOST}/api/chat",
            json={"model": model, "messages": msgs, "stream": True, "think": False,
                  "options": {"temperature": 0.35, "num_predict": npred},
                  "keep_alive": "1h"},
            stream=True, timeout=(5, 3600), proxies={"http": None, "https": None}) as r:
    r.raise_for_status()
    for line in r.iter_lines():
        if not line:
            continue
        d = json.loads(line)
        ch = (d.get("message") or {}).get("content", "")
        if ch:
            if first is None:
                first = time.time() - t0
                print(f"  TTFT (обработка промпта) = {first:.1f}s", flush=True)
            buf.append(ch); n += 1
            if n % 50 == 0:
                el = time.time() - t0 - first
                print(f"  {n:5d} чанков  {el:6.1f}s  {n/max(el,.01):5.2f} tok/s", flush=True)
        if d.get("done"):
            print(f"\ndone_reason={d.get('done_reason')} "
                  f"prompt={d.get('prompt_eval_count')} out={d.get('eval_count')} "
                  f"pe={(d.get('prompt_eval_duration') or 0)/1e9:.1f}s "
                  f"gen={(d.get('eval_duration') or 0)/1e9:.1f}s "
                  f"wall={time.time()-t0:.1f}s", flush=True)
text = LC._strip_thinking("".join(buf))
open(os.path.join(HERE, f"probe_{case}_{model.replace(':','_')}.txt"), "w",
     encoding="utf-8").write(text)
print(f"\nсимволов в ответе: {len(text)}")
print(text[:600])
