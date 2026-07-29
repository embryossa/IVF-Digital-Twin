# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
eval_retrieval.py — IVF Digital Twin
Deterministic regression test for the guideline retriever (Layer B).

For a set of patient profiles it checks that:
  - the nomogram produces the expected response phenotype, and
  - the rule-keyed retriever surfaces the guideline statements that MUST be
    present for that clinical situation (coverage), and
  - reports precision@k against the full set of key-relevant statements.

Rule-keyed only (no embeddings) so the result is fully reproducible and usable
as a CI gate. Exits non-zero if any required statement is missing.

Run:  python eval_retrieval.py
"""

from __future__ import annotations

import sys
from typing import Dict, List, Set

from stim_protocol import compute_stim, StimInput
import guideline_rag as gr

K = 8

# profile → (expected phenotype, statement ids that MUST be retrieved)
CASES = [
    {
        "name": "high responder / OHSS",
        "input": StimInput(age=31, amh_ng_ml=5.6, afc=26, bmi=23),
        "phenotype": "high_responder",
        "must_include": {"ESHRE_ort_dose_reduces_ohss",
                         "ASRM_antagonist_over_agonist",
                         "ASRM_avoid_fresh_transfer"},
    },
    {
        "name": "poor responder / low reserve",
        "input": StimInput(age=41, amh_ng_ml=0.6, afc=4, bmi=31),
        "phenotype": "poor_responder",
        "must_include": {"poor_responder_dose_ceiling",
                         "ESHRE_poor_no_superior_protocol",
                         "POSEIDON_low_prognosis"},
    },
    {
        "name": "normal responder",
        "input": StimInput(age=34, amh_ng_ml=2.1, afc=12, bmi=26),
        "phenotype": "normal_responder",
        "must_include": {"ESHRE_conventional_dose",
                         "ESHRE_antagonist_first_line"},
    },
]


def _relevant_ids(keys: List[str], pack) -> Set[str]:
    want = set(keys)
    return {s["id"] for s in pack.get("statements", [])
            if want.intersection(s.get("keys", []))}


def main() -> int:
    gr._PACK_CACHE = None
    pack = gr.load_pack()
    all_ok = True
    print(f"Retrieval eval (rule-keyed, k={K})\n" + "─" * 48)

    for case in CASES:
        out = compute_stim(case["input"])
        keys = out.situation_keys
        retrieved = gr.retrieve_by_keys(keys, pack=pack, max_n=K)
        got_ids = {r.get("id") for r in retrieved}

        pheno_ok = (out.response_phenotype == case["phenotype"])
        missing = case["must_include"] - got_ids
        cover_ok = not missing

        relevant = _relevant_ids(keys, pack)
        prec = len(got_ids & relevant) / max(len(got_ids), 1)

        ok = pheno_ok and cover_ok
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {case['name']}")
        print(f"        phenotype: {out.response_phenotype} "
              f"({'ok' if pheno_ok else 'EXPECTED ' + case['phenotype']})")
        print(f"        precision@{K}: {prec:.2f}  | coverage: "
              f"{'ok' if cover_ok else 'MISSING ' + ', '.join(sorted(missing))}")

    print("─" * 48)
    print("RESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
