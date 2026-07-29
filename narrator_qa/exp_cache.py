# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
exp_cache.py — проверка гипотезы о переиспользовании KV-кэша префикса.

Гипотеза: llama.cpp/Ollama сохраняет KV-кэш последнего запроса на слоте и
переиспользует ОБЩИЙ ПРЕФИКС следующего. Сейчас в DT переменная часть
(patient, main_forecast…) идёт В НАЧАЛЕ user-сообщения, а стабильная
(protocol_guidance с корпусом гайдлайнов) — В КОНЦЕ. Значит между двумя
пациентами совпадает только системный промпт, и ~1500 токенов корпуса
пересчитываются каждый раз.

Эксперимент (num_predict=8 — измеряем только обработку промпта):
  A1  пациент A, текущий порядок            → полная стоимость промпта
  A2  пациент A ПОВТОРНО                    → есть ли кэш вообще
  B1  пациент B, текущий порядок            → сколько переиспользовано
  C1  пациент A, СТАБИЛЬНОЕ-ВПЕРЁД          → базовая линия нового порядка
  C2  пациент B, СТАБИЛЬНОЕ-ВПЕРЁД          → сколько переиспользовано теперь

Запуск: python exp_cache.py [--model medgemma1.5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DT_DIR = r"C:\Users\User\Desktop\IVF\AI\IVF Digital Twin Pro\IVF Digital Twin"
if DT_DIR not in sys.path:
    sys.path.insert(0, DT_DIR)

import requests  # noqa: E402
import llm_consultant as LC  # noqa: E402

_S = requests.Session()
_S.trust_env = False


def msgs_current(ctx):
    """Текущий продакшн-порядок: весь контекст одним JSON, пациент первым."""
    return LC._build_messages(ctx, "Кратко: главный прогноз одним предложением.")


def msgs_stable_first(ctx):
    """Предлагаемый порядок: инвариантный блок гайдлайнов ДО переменного пациента."""
    c = dict(ctx)
    pg = c.pop("protocol_guidance", None)
    stable = ""
    if pg:
        stable = ("Справочный блок (инвариантен для этого клинического профиля):\n"
                  f"```json\n{json.dumps(pg, ensure_ascii=False, indent=2)}\n```\n\n")
    user = ("/no_think\n" + stable
            + "Контекст пациента (использовать ТОЛЬКО эти числа):\n"
            f"```json\n{json.dumps(c, ensure_ascii=False, indent=2)}\n```\n\n"
            "Запрос: Кратко: главный прогноз одним предложением.")
    return [{"role": "system", "content": LC._SYSTEM_NARRATOR},
            {"role": "user", "content": user}]


def call(model, messages, label):
    t0 = time.time()
    r = _S.post(f"{LC.OLLAMA_HOST}/api/chat",
                json={"model": model, "messages": messages, "stream": False,
                      "think": False,
                      "options": {"temperature": 0.1, "num_predict": 8},
                      "keep_alive": "1h"},
                timeout=(5, 3600), proxies={"http": None, "https": None})
    r.raise_for_status()
    d = r.json()
    pe = (d.get("prompt_eval_duration") or 0) / 1e9
    pt = d.get("prompt_eval_count") or 0
    print(f"{label:34s} prompt_tokens={pt:5d}  prompt_eval={pe:7.1f}s  "
          f"({pt / max(pe, .01):6.1f} t/s)  wall={time.time() - t0:6.1f}s", flush=True)
    return {"label": label, "prompt_tokens": pt, "prompt_eval_s": round(pe, 1),
            "wall_s": round(time.time() - t0, 1)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medgemma1.5")
    ap.add_argument("--a", default="S01_high_responder_ohss")
    ap.add_argument("--b", default="S12_obese_high_reserve_dose")
    args = ap.parse_args()

    with open(os.path.join(HERE, "contexts.json"), encoding="utf-8") as f:
        cases = {c["id"]: c for c in json.load(f)}
    A, B = cases[args.a]["ctx"], cases[args.b]["ctx"]

    print(f"model={args.model}  A={args.a}  B={args.b}\n" + "─" * 92)
    out = []
    out.append(call(args.model, msgs_current(A), "A1 текущий порядок, пациент A"))
    out.append(call(args.model, msgs_current(A), "A2 тот же запрос повторно"))
    out.append(call(args.model, msgs_current(B), "B1 текущий порядок, пациент B"))
    out.append(call(args.model, msgs_stable_first(A), "C1 стабильное-вперёд, пациент A"))
    out.append(call(args.model, msgs_stable_first(B), "C2 стабильное-вперёд, пациент B"))

    print("─" * 92)
    a1, a2, b1, c1, c2 = out
    print(f"кэш точного повтора:      {a1['prompt_eval_s']:.1f}s → {a2['prompt_eval_s']:.1f}s")
    print(f"смена пациента, СЕЙЧАС:   {b1['prompt_eval_s']:.1f}s")
    print(f"смена пациента, ПРЕДЛОЖ.: {c2['prompt_eval_s']:.1f}s "
          f"(база нового порядка {c1['prompt_eval_s']:.1f}s)")
    if b1["prompt_eval_s"] > 0:
        print(f"выигрыш на пациента:      "
              f"{b1['prompt_eval_s'] - c2['prompt_eval_s']:.1f}s "
              f"({100 * (1 - c2['prompt_eval_s'] / b1['prompt_eval_s']):.0f}%)")
    with open(os.path.join(HERE, "exp_cache.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "runs": out}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
