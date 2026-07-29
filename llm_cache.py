# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
llm_cache.py — IVF Digital Twin
Tiny disk cache for narrator outputs, keyed by a hash of the exact context.

Motivation: on CPU the narrator is the latency bottleneck. Identical patient
context (same numbers, same grounding, same model/style) should not be
re-generated — re-opening a patient becomes instant.

Reproducibility note: the narrator runs at temperature > 0, so generation is not
deterministic. Caching pins the FIRST generated narrative for a given context,
which is actually desirable clinically (same input → same displayed text). Clear
the cache dir or set DT_LLM_CACHE=0 to bypass.

Cache auto-invalidates when anything in the context changes — including the
retrieved guideline statements — because the whole context is hashed.

Dependencies: standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional

_ENABLED = os.environ.get("DT_LLM_CACHE", "1") not in ("0", "false", "False", "")
_DIR = os.environ.get(
    "DT_LLM_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dt_llm_cache"),
)


def enabled() -> bool:
    return _ENABLED


def make_key(*parts: Any) -> str:
    """Stable hash over arbitrary parts (dicts are JSON-normalised, sorted)."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (dict, list)):
            p = json.dumps(p, ensure_ascii=False, sort_keys=True)
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def get(key: str) -> Optional[str]:
    if not _ENABLED:
        return None
    path = os.path.join(_DIR, key + ".txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def set(key: str, text: str) -> None:
    if not _ENABLED or not text:
        return
    try:
        os.makedirs(_DIR, exist_ok=True)
        with open(os.path.join(_DIR, key + ".txt"), "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass


def clear() -> int:
    """Delete all cached narratives. Returns count removed."""
    n = 0
    try:
        for fn in os.listdir(_DIR):
            if fn.endswith(".txt"):
                os.remove(os.path.join(_DIR, fn))
                n += 1
    except Exception:
        pass
    return n
