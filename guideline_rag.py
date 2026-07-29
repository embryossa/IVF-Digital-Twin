# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
guideline_rag.py — IVF Digital Twin
Offline retrieval over the published-guideline corpus (Layer B).

Design
------
Two retrieval mechanisms, in order of trust:

1. RULE-KEYED (primary, deterministic, always available).
   stim_protocol emits `situation_keys`; we pull every statement tagged with
   any of those keys. No embeddings, nothing to hallucinate, fully reproducible.
   This is the backbone and is what ships enabled.

2. SEMANTIC RERANK (optional, opt-in).
   When a free-text query is supplied AND an Ollama embedding model is
   reachable, we re-order the rule-keyed candidates by cosine similarity to the
   query. The corpus is tiny (tens of statements) so brute-force cosine over a
   plain list is more than enough — no vector DB, no FAISS, no LangChain.
   If Ollama is unavailable, we silently fall back to rule-keyed order.

The retrieved statements are returned verbatim from the corpus (already short
paraphrases with citations) and handed to the narrator as grounding facts. The
LLM cites them; it does not invent guideline content.

Dependencies: standard library + requests (only for the optional embed call).
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

_PACK_PATH = os.environ.get(
    "DT_GUIDELINES_PACK",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "guidelines_pack.json"),
)

# Reuse the same Ollama host convention as llm_consultant.py.
_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
_EMBED_MODEL = os.environ.get("DT_EMBED_MODEL", "nomic-embed-text")
_EMBED_TIMEOUT = (3, int(os.environ.get("DT_EMBED_TIMEOUT", 30)))

_PACK_CACHE: Optional[Dict[str, Any]] = None


# ──────────────────────────────────────────────────────────────────────────
#  CORPUS LOADING
# ──────────────────────────────────────────────────────────────────────────
def load_pack(path: str = _PACK_PATH, use_cache: bool = True) -> Dict[str, Any]:
    """Load the guideline corpus. Returns {'_meta':..., 'statements':[...]}.

    On any failure returns an empty but well-formed pack so callers never crash.
    """
    global _PACK_CACHE
    if use_cache and _PACK_CACHE is not None:
        return _PACK_CACHE
    try:
        with open(path, "r", encoding="utf-8") as f:
            pack = json.load(f)
        if "statements" not in pack:
            pack = {"_meta": {}, "statements": []}
    except Exception:
        pack = {"_meta": {}, "statements": []}
    _PACK_CACHE = pack
    return pack


def _format_statement(st: Dict[str, Any], pack: Dict[str, Any]) -> Dict[str, Any]:
    """Attach a resolved citation string to a statement for the narrator.

    `source` (the short corpus key, e.g. "ESHRE2025") is kept alongside the full
    `citation`: it is the tag the narrator is asked to print, so the prompt no
    longer has to carry ~200 characters of bibliography per statement. `id` stays
    as-is — eval_retrieval.py checks retrieval coverage by statement id.
    """
    src_key = st.get("source", "")
    citation = (pack.get("_meta", {}).get("sources", {}) or {}).get(src_key, src_key)
    return {
        "id": st.get("id"),
        "source": src_key,
        "text": st.get("text"),
        "evidence_level": st.get("evidence_level"),
        "citation": citation,
    }


# ──────────────────────────────────────────────────────────────────────────
#  1) RULE-KEYED RETRIEVAL (primary)
# ──────────────────────────────────────────────────────────────────────────
def retrieve_by_keys(situation_keys: List[str],
                     pack: Optional[Dict[str, Any]] = None,
                     max_n: int = 8) -> List[Dict[str, Any]]:
    """Return statements whose `keys` intersect `situation_keys`.

    Two-pass selection:
      1. COVERAGE — for each situation_key (in order), guarantee its single
         best-overlap statement is included. This ensures rarer-but-important
         keys (e.g. freeze_all_candidate) are never crowded out by keys that
         happen to appear in many statements.
      2. FILL — remaining slots are filled by overall overlap score.
    Result is finally ordered by overlap (strongest grounding first).
    Fully deterministic.
    """
    pack = pack or load_pack()
    keys = list(dict.fromkeys(situation_keys or []))   # dedupe, keep order
    statements = pack.get("statements", [])

    # Score every matching statement by overlap size (ties broken by corpus order).
    scored = []
    for idx, st in enumerate(statements):
        overlap = len(set(keys).intersection(st.get("keys", [])))
        if overlap:
            scored.append({"overlap": overlap, "idx": idx, "st": st})
    scored.sort(key=lambda d: (d["overlap"], -d["idx"]), reverse=True)

    chosen: List[Dict[str, Any]] = []
    chosen_ids = set()

    # Pass 1 — coverage: best statement per key.
    for k in keys:
        if len(chosen) >= max_n:
            break
        for d in scored:
            st = d["st"]
            if k in st.get("keys", []) and st["id"] not in chosen_ids:
                chosen.append(d)
                chosen_ids.add(st["id"])
                break

    # Pass 2 — fill remaining slots by overlap rank.
    for d in scored:
        if len(chosen) >= max_n:
            break
        if d["st"]["id"] not in chosen_ids:
            chosen.append(d)
            chosen_ids.add(d["st"]["id"])

    # Final ordering: strongest grounding first.
    chosen.sort(key=lambda d: (d["overlap"], -d["idx"]), reverse=True)
    return [_format_statement(d["st"], pack) for d in chosen[:max_n]]


# ──────────────────────────────────────────────────────────────────────────
#  2) SEMANTIC RERANK (optional)
# ──────────────────────────────────────────────────────────────────────────
def _embed_ollama(texts: List[str]) -> Optional[List[List[float]]]:
    """Embed texts via Ollama /api/embed. Returns None on any failure."""
    try:
        import requests
        r = requests.post(
            f"{_OLLAMA_HOST}/api/embed",
            json={"model": _EMBED_MODEL, "input": texts},
            timeout=_EMBED_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        embs = data.get("embeddings")
        if isinstance(embs, list) and len(embs) == len(texts):
            return embs
    except Exception:
        pass
    return None


def _cosine_np(q, m):
    """Cosine of query vector q (D,) against matrix m (N,D). Returns (N,)."""
    import numpy as np
    qn = q / (np.linalg.norm(q) + 1e-12)
    mn = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    return mn @ qn


# ──────────────────────────────────────────────────────────────────────────
#  PRECOMPUTED CORPUS EMBEDDINGS (built once, cached to .npz)
#  On a query we embed ONLY the query and cosine it against the cached matrix.
#  Cache auto-invalidates when the corpus text or the embed model changes.
# ──────────────────────────────────────────────────────────────────────────
_EMB_CACHE_PATH = os.environ.get(
    "DT_GUIDELINES_EMB_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "guideline_emb_cache.npz"),
)
_EMB_MEM: Optional[Dict[str, Any]] = None   # in-process cache: {"sig","ids","mat"}


def _corpus_signature(pack: Dict[str, Any]) -> str:
    import hashlib
    parts = [_EMBED_MODEL]
    for st in pack.get("statements", []):
        parts.append(f"{st.get('id')}\t{st.get('text')}")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def build_embedding_cache(pack: Optional[Dict[str, Any]] = None,
                          force: bool = False) -> bool:
    """Embed every corpus statement ONCE and persist to .npz. Returns success.

    Safe to call at startup. No-op (returns True) if an up-to-date cache exists
    and force is False. Returns False if the embedder is unavailable.
    """
    pack = pack or load_pack()
    sig = _corpus_signature(pack)
    if not force and os.path.exists(_EMB_CACHE_PATH):
        try:
            import numpy as np
            z = np.load(_EMB_CACHE_PATH, allow_pickle=True)
            if str(z["sig"]) == sig:
                return True
        except Exception:
            pass
    statements = pack.get("statements", [])
    texts = [st.get("text", "") for st in statements]
    ids = [st.get("id", "") for st in statements]
    embs = _embed_ollama(texts)
    if embs is None:
        return False
    try:
        import numpy as np
        np.savez(_EMB_CACHE_PATH,
                 sig=sig, model=_EMBED_MODEL,
                 ids=np.asarray(ids, dtype=object),
                 mat=np.asarray(embs, dtype=np.float32))
        return True
    except Exception:
        return False


def _ensure_corpus_embeddings(pack: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return {'ids':[...], 'mat':NxD np.array} for the corpus, building/loading
    the cache as needed. None if embeddings are unavailable (graceful)."""
    global _EMB_MEM
    sig = _corpus_signature(pack)
    if _EMB_MEM is not None and _EMB_MEM.get("sig") == sig:
        return _EMB_MEM
    try:
        import numpy as np
    except Exception:
        return None
    # Try disk cache; rebuild if stale/missing.
    if not (os.path.exists(_EMB_CACHE_PATH)
            and _try_sig_match(_EMB_CACHE_PATH, sig)):
        if not build_embedding_cache(pack, force=True):
            return None
    try:
        z = np.load(_EMB_CACHE_PATH, allow_pickle=True)
        _EMB_MEM = {"sig": str(z["sig"]),
                    "ids": list(z["ids"]),
                    "mat": np.asarray(z["mat"], dtype=np.float32)}
        return _EMB_MEM
    except Exception:
        return None


def _try_sig_match(path: str, sig: str) -> bool:
    try:
        import numpy as np
        z = np.load(path, allow_pickle=True)
        return str(z["sig"]) == sig
    except Exception:
        return False


def semantic_rerank(query: str,
                    candidates: List[Dict[str, Any]],
                    pack: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Reorder candidates by cosine(query, cached corpus vector). Graceful.

    Only the QUERY is embedded at call time; candidate vectors come from the
    precomputed corpus cache (keyed by statement id). Falls back to the input
    order if embeddings are unavailable.
    """
    if not query or len(candidates) < 2:
        return candidates
    pack = pack or load_pack()
    corpus = _ensure_corpus_embeddings(pack)
    if corpus is None:
        return candidates
    q = _embed_ollama([query])
    if not q:
        return candidates
    try:
        import numpy as np
        qv = np.asarray(q[0], dtype=np.float32)
        id_to_row = {sid: i for i, sid in enumerate(corpus["ids"])}
        mat = corpus["mat"]
        scored = []
        for rank, c in enumerate(candidates):
            row = id_to_row.get(c.get("id"))
            sim = float(_cosine_np(qv, mat[row:row + 1])[0]) if row is not None else -1.0
            scored.append((sim, -rank, c))   # stable: keep input order on ties
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [c for _, _, c in scored]
    except Exception:
        return candidates


# ──────────────────────────────────────────────────────────────────────────
#  PUBLIC ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────
def retrieve(situation_keys: List[str],
             query: Optional[str] = None,
             max_n: int = 6,
             pack: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Primary retrieval call used by the narrator integration.

    1. Rule-keyed candidate set (deterministic).
    2. Optional semantic rerank if `query` given and Ollama embedder is up.
    """
    pack = pack or load_pack()
    candidates = retrieve_by_keys(situation_keys, pack=pack, max_n=max(max_n, 8))
    if query:
        candidates = semantic_rerank(query, candidates, pack=pack)
    return candidates[:max_n]


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if "--build" in sys.argv:
        ok = build_embedding_cache(force=True)
        print("embedding cache built:" if ok else "embedder unavailable:",
              _EMB_CACHE_PATH if ok else "(rule-keyed retrieval still works)")
        sys.exit(0)
    keys = ["high_responder", "ohss_risk_elevated", "freeze_all_candidate", "amh_high"]
    print("Rule-keyed retrieval for:", keys)
    for s in retrieve(keys):
        print(f"  • [{s['evidence_level']}] {s['text']}  ({s['citation']})")
