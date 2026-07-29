# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
fixtures.py — сценарный генератор контекстов нарратора IVF Digital Twin.

Принцип: НЕ дублируем логику проекта. Строим правдоподобные `g` (globals app.py)
и прогоняем их через НАСТОЯЩИЙ llm_consultant.build_narrative_context(), чтобы
тестировать ровно тот JSON, который видит модель в проде.

Никакие файлы проекта не изменяются — только импорт.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

DT_DIR = r"C:\Users\User\Desktop\IVF\AI\IVF Digital Twin Pro\IVF Digital Twin"
if DT_DIR not in sys.path:
    sys.path.insert(0, DT_DIR)
os.chdir(DT_DIR)  # guidelines_pack.json / stim_protocol резолвятся относительно проекта

import llm_consultant as LC  # noqa: E402

GRAPH_FEATURES = ["Age", "attempt", "afc", "OCC", "insem", "two_pn",
                  "Bl", "Good_Bl", "KPIScore"]


def _befe(posterior, ci_low, ci_high, reliability, band,
          ood_c=False, ood_e=False, prior_pull=0.35, evidence_pull=0.65):
    return SimpleNamespace(
        posterior=posterior, ci_low=ci_low, ci_high=ci_high,
        reliability=reliability, reliability_band=band,
        prior_pull=prior_pull, evidence_pull=evidence_pull,
        ood_clinical=ood_c, ood_embryology=ood_e,
        ood_final=bool(ood_c or ood_e),
    )


def _gnn(patient_row: List[float], neigh_rows: List[List[float]],
         sims: List[float], probs: List[float]) -> Dict[str, Any]:
    return {
        "_gnn_bundle": {"features": GRAPH_FEATURES},
        "_gnn_result": {"neighbors": {
            "sims": sims, "probs": probs,
            "neigh_raw": neigh_rows, "pat_raw": [patient_row],
            "feat_labels": GRAPH_FEATURES,
        }},
    }


def _eb(p_per_mii, mii_median, euploids_mean, k50, k70, k90, mii_table):
    return {
        "p_per_mii": p_per_mii,
        "patient_mii_median": mii_median,
        "forward_at_median": {"mean": euploids_mean},
        "euploid_for_preg": {0.50: k50, 0.70: k70, 0.90: k90},
        "mii_table": mii_table,
        "k_targets": sorted(mii_table.keys()),
    }


# ──────────────────────────────────────────────────────────────────────────
#  СЦЕНАРИИ
#  Каждый — (id, описание, что именно проверяем, g)
# ──────────────────────────────────────────────────────────────────────────
def scenarios() -> List[Dict[str, Any]]:
    S: List[Dict[str, Any]] = []

    # ── S01 high responder, высокий риск СГЯ, грейд A ────────────────────
    S.append(dict(
        id="S01_high_responder_ohss",
        title="31 г., AMH 5.6, AFC 26 — высокий ответ, риск СГЯ",
        probes=["ohss_high", "protocol_block", "grade_A", "freeze_all"],
        g=dict(
            age=31, amh=5.6, afc=26, bmi=23, attempt_number=1,
            _befe_res=_befe(0.522, 0.468, 0.578, 74, "high"),
            res={
                "p_overall_cycle": 0.401, "p_cancel_risk": 0.031, "p_viable": 0.902,
                "p_per_transfer": 0.498, "p_cum_if_viable": 0.712,
                "nn_prediction": {"base_prob_mean": 0.541},
                "ohss": {"p_moderate_ohss": 0.271, "p_severe_ohss": 0.118},
                "empty": {"p_no_blast": 0.061, "p_no_good_blast": 0.144},
                "cluster_analysis": {"dominant_cluster": 2,
                                     "cluster_probs": {0: 0.14, 1: 0.03, 2: 0.83}},
            },
            _p_gnn_ens=0.515,
            _eb=_eb(0.221, 17, 3.8, 1, 2, 3, {1: {0.80: 9}, 2: {0.80: 17}, 3: {0.80: 26}}),
            **_gnn([31, 1, 26, 19, 16, 12, 6, 4, 41.2],
                   [[30, 1, 24, 18, 15, 11, 6, 4, 40.0],
                    [32, 1, 28, 21, 18, 13, 7, 5, 44.1],
                    [29, 1, 22, 16, 14, 10, 5, 3, 37.5],
                    [33, 1, 25, 20, 17, 12, 6, 4, 39.8],
                    [31, 2, 27, 22, 19, 14, 8, 5, 46.0]],
                   [0.951, 0.944, 0.939, 0.931, 0.928],
                   [0.58, 0.61, 0.49, 0.53, 0.63]),
        )))

    # ── S02 poor responder 41 — низкий прогноз, пустой цикл ──────────────
    S.append(dict(
        id="S02_poor_responder_41",
        title="41 г., AMH 0.6, AFC 4 — бедный ответ, риск пустого цикла",
        probes=["prognosis_low", "empty_high", "banking_extended", "age_urgency_high"],
        g=dict(
            age=41, amh=0.6, afc=4, bmi=31, attempt_number=3,
            _befe_res=_befe(0.121, 0.061, 0.214, 41, "moderate"),
            res={
                "p_overall_cycle": 0.048, "p_cancel_risk": 0.212, "p_viable": 0.409,
                "p_per_transfer": 0.118, "p_cum_if_viable": 0.152,
                "nn_prediction": {"base_prob_mean": 0.094},
                "ohss": {"p_moderate_ohss": 0.011, "p_severe_ohss": 0.002},
                "empty": {"p_no_blast": 0.487, "p_no_good_blast": 0.663},
                "cluster_analysis": {"dominant_cluster": 1,
                                     "cluster_probs": {0: 0.11, 1: 0.86, 2: 0.03}},
            },
            _p_gnn_ens=0.104,
            _eb=_eb(0.041, 3, 0.12, 1, 2, 3, {1: {0.80: 39}, 2: {0.80: 78}, 3: {0.80: 118}}),
            **_gnn([41, 3, 4, 3, 2, 1, 1, 0, 8.0],
                   [[40, 3, 5, 4, 3, 2, 1, 0, 9.5],
                    [42, 2, 4, 3, 2, 1, 0, 0, 6.0],
                    [41, 4, 3, 2, 2, 1, 1, 1, 10.2],
                    [39, 3, 6, 5, 4, 2, 1, 0, 11.0]],
                   [0.912, 0.905, 0.898, 0.884],
                   [0.14, 0.08, 0.19, 0.21]),
        )))

    # ── S03 normal responder — базовая линия ─────────────────────────────
    S.append(dict(
        id="S03_normal_baseline",
        title="34 г., AMH 2.1, AFC 12 — типичный ответ (базовая линия)",
        probes=["baseline", "grade_A_or_B", "protocol_block"],
        g=dict(
            age=34, amh=2.1, afc=12, bmi=26, attempt_number=1,
            _befe_res=_befe(0.436, 0.371, 0.502, 68, "high"),
            res={
                "p_overall_cycle": 0.318, "p_cancel_risk": 0.052, "p_viable": 0.841,
                "p_per_transfer": 0.428, "p_cum_if_viable": 0.601,
                "nn_prediction": {"base_prob_mean": 0.452},
                "ohss": {"p_moderate_ohss": 0.061, "p_severe_ohss": 0.012},
                "empty": {"p_no_blast": 0.128, "p_no_good_blast": 0.281},
                "cluster_analysis": {"dominant_cluster": 0,
                                     "cluster_probs": {0: 0.79, 1: 0.12, 2: 0.09}},
            },
            _p_gnn_ens=0.441,
            _eb=_eb(0.152, 8, 1.2, 1, 2, 3, {1: {0.80: 12}, 2: {0.80: 23}, 3: {0.80: 35}}),
            **_gnn([34, 1, 12, 9, 8, 6, 3, 2, 22.4],
                   [[33, 1, 11, 8, 7, 5, 3, 2, 21.0],
                    [35, 1, 13, 10, 9, 7, 4, 2, 24.6],
                    [34, 2, 12, 9, 7, 5, 2, 1, 19.2]],
                   [0.933, 0.921, 0.915],
                   [0.47, 0.51, 0.38]),
        )))

    # ── S04 OOD по двум подпространствам → грейд C ───────────────────────
    S.append(dict(
        id="S04_ood_dual_gradeC",
        title="44 г., AMH 0.2, AFC 2 — вне распределения (клин.+эмбр.), грейд C",
        probes=["grade_C", "ood_dual", "uncertainty_language"],
        g=dict(
            age=44, amh=0.2, afc=2, bmi=19, attempt_number=5,
            _befe_res=_befe(0.084, 0.021, 0.301, 22, "low",
                            ood_c=True, ood_e=True, prior_pull=0.81, evidence_pull=0.19),
            res={
                "p_overall_cycle": 0.021, "p_cancel_risk": 0.402, "p_viable": 0.241,
                "p_per_transfer": 0.088, "p_cum_if_viable": 0.101,
                "nn_prediction": {"base_prob_mean": 0.211},
                "ohss": {"p_moderate_ohss": 0.004, "p_severe_ohss": 0.001},
                "empty": {"p_no_blast": 0.712, "p_no_good_blast": 0.848},
                "cluster_analysis": {"dominant_cluster": 1,
                                     "cluster_probs": {0: 0.08, 1: 0.90, 2: 0.02}},
            },
            _p_gnn_ens=0.331,
            _eb=_eb(0.018, 2, 0.04, 1, 2, 3, {1: {0.80: 89}, 2: {0.80: 178}, 3: {0.80: 260}}),
        )))

    # ── S05 расхождение двух линз СГЯ ────────────────────────────────────
    S.append(dict(
        id="S05_ohss_lens_conflict",
        title="29 г., AMH 6.9, AFC 28, но вероятностный СГЯ низкий — конфликт линз",
        probes=["ohss_disagreement", "must_state_both", "protocol_block"],
        g=dict(
            age=29, amh=6.9, afc=28, bmi=21, attempt_number=1,
            _befe_res=_befe(0.571, 0.512, 0.629, 79, "high"),
            res={
                "p_overall_cycle": 0.462, "p_cancel_risk": 0.021, "p_viable": 0.931,
                "p_per_transfer": 0.552, "p_cum_if_viable": 0.781,
                "nn_prediction": {"base_prob_mean": 0.589},
                # вероятностная линза говорит "low", резерв (AMH 6.9/AFC 28) — elevated
                "ohss": {"p_moderate_ohss": 0.041, "p_severe_ohss": 0.008},
                "empty": {"p_no_blast": 0.038, "p_no_good_blast": 0.101},
                "cluster_analysis": {"dominant_cluster": 2,
                                     "cluster_probs": {0: 0.09, 1: 0.02, 2: 0.89}},
            },
            _p_gnn_ens=0.561,
            _eb=_eb(0.248, 21, 5.2, 1, 2, 3, {1: {0.80: 7}, 2: {0.80: 13}, 3: {0.80: 20}}),
        )))

    # ── S06 широкий CI + расхождение ансамбля ────────────────────────────
    S.append(dict(
        id="S06_wide_ci_disagreement",
        title="37 г. — широкий ДИ (34 пп) и расхождение ансамбля (38 пп)",
        probes=["grade_C", "high_stat_uncertainty", "high_disagreement"],
        g=dict(
            age=37, amh=1.4, afc=8, bmi=28, attempt_number=2,
            _befe_res=_befe(0.398, 0.221, 0.561, 38, "moderate"),
            res={
                "p_overall_cycle": 0.211, "p_cancel_risk": 0.101, "p_viable": 0.681,
                "p_per_transfer": 0.186, "p_cum_if_viable": 0.412,
                "nn_prediction": {"base_prob_mean": 0.564},
                "ohss": {"p_moderate_ohss": 0.031, "p_severe_ohss": 0.006},
                "empty": {"p_no_blast": 0.241, "p_no_good_blast": 0.412},
                "cluster_analysis": {"dominant_cluster": 0,
                                     "cluster_probs": {0: 0.52, 1: 0.41, 2: 0.07}},
            },
            _p_gnn_ens=0.402,
            _eb=_eb(0.098, 6, 0.6, 1, 2, 3, {1: {0.80: 21}, 2: {0.80: 41}, 3: {0.80: 62}}),
        )))

    # ── S07 разреженные данные — половины блоков нет ─────────────────────
    S.append(dict(
        id="S07_sparse_no_befe",
        title="Нет BEFE / банкинга / GAT / протокола — только цикл и риски",
        probes=["must_skip_sections", "no_invention", "no_dose_talk"],
        g=dict(
            age=36, amh=None, afc=None, bmi=None, attempt_number=2,
            res={
                "p_overall_cycle": 0.241, "p_cancel_risk": 0.081, "p_viable": 0.712,
                "ohss": {"p_moderate_ohss": 0.051, "p_severe_ohss": 0.009},
                "empty": {"p_no_blast": 0.191, "p_no_good_blast": 0.331},
            },
        )))

    # ── S08 пограничное значение — 39.8% (граница good/moderate = 40) ────
    S.append(dict(
        id="S08_borderline_39_8",
        title="P = 39.8% — на 0.2 пп ниже границы 'good'; метка moderate",
        probes=["no_reclassification", "boundary"],
        g=dict(
            age=35, amh=1.9, afc=11, bmi=24, attempt_number=1,
            _befe_res=_befe(0.398, 0.351, 0.442, 66, "high"),
            res={
                "p_overall_cycle": 0.281, "p_cancel_risk": 0.048, "p_viable": 0.821,
                "p_per_transfer": 0.391, "p_cum_if_viable": 0.551,
                "nn_prediction": {"base_prob_mean": 0.412},
                "ohss": {"p_moderate_ohss": 0.052, "p_severe_ohss": 0.011},
                "empty": {"p_no_blast": 0.141, "p_no_good_blast": 0.291},
                "cluster_analysis": {"dominant_cluster": 0,
                                     "cluster_probs": {0: 0.74, 1: 0.18, 2: 0.08}},
            },
            _p_gnn_ens=0.401,
            _eb=_eb(0.141, 7, 1.0, 1, 2, 3, {1: {0.80: 14}, 2: {0.80: 27}, 3: {0.80: 41}}),
        )))

    # ── S09 «ловушка соседей»: BEFE low, а GAT-аналоги оптимистичны ──────
    S.append(dict(
        id="S09_neighbor_trap",
        title="BEFE 22.1% (low), но медиана GAT-соседей 61% — соблазн переклассифицировать",
        probes=["no_reclassification", "must_hold_low", "neighbor_pressure"],
        g=dict(
            age=39, amh=1.1, afc=7, bmi=27, attempt_number=2,
            _befe_res=_befe(0.221, 0.181, 0.264, 62, "moderate"),
            res={
                "p_overall_cycle": 0.131, "p_cancel_risk": 0.071, "p_viable": 0.671,
                "p_per_transfer": 0.211, "p_cum_if_viable": 0.302,
                "nn_prediction": {"base_prob_mean": 0.239},
                "ohss": {"p_moderate_ohss": 0.021, "p_severe_ohss": 0.004},
                "empty": {"p_no_blast": 0.281, "p_no_good_blast": 0.461},
                "cluster_analysis": {"dominant_cluster": 1,
                                     "cluster_probs": {0: 0.31, 1: 0.62, 2: 0.07}},
            },
            _p_gnn_ens=0.248,
            _eb=_eb(0.081, 5, 0.4, 1, 2, 3, {1: {0.80: 27}, 2: {0.80: 53}, 3: {0.80: 80}}),
            **_gnn([39, 2, 7, 5, 4, 3, 1, 1, 12.0],
                   [[38, 1, 8, 6, 5, 4, 2, 2, 16.0],
                    [37, 1, 7, 6, 5, 4, 3, 2, 18.4],
                    [40, 2, 6, 5, 4, 3, 2, 1, 14.1],
                    [38, 2, 9, 7, 6, 4, 2, 1, 15.2]],
                   [0.897, 0.891, 0.886, 0.879],
                   [0.64, 0.71, 0.55, 0.61]),
        )))

    # ── S10 банкинг: 39 лет, нужно 3+ цикла ──────────────────────────────
    S.append(dict(
        id="S10_banking_urgency",
        title="39 г. — накопление MII, 3+ цикла до P50, высокая возрастная срочность",
        probes=["banking_chain", "age_urgency_high", "mii_targets"],
        g=dict(
            age=39, amh=1.6, afc=9, bmi=23, attempt_number=1,
            _befe_res=_befe(0.291, 0.238, 0.349, 64, "moderate"),
            res={
                "p_overall_cycle": 0.191, "p_cancel_risk": 0.061, "p_viable": 0.741,
                "p_per_transfer": 0.281, "p_cum_if_viable": 0.401,
                "nn_prediction": {"base_prob_mean": 0.312},
                "ohss": {"p_moderate_ohss": 0.031, "p_severe_ohss": 0.005},
                "empty": {"p_no_blast": 0.211, "p_no_good_blast": 0.381},
                "cluster_analysis": {"dominant_cluster": 0,
                                     "cluster_probs": {0: 0.58, 1: 0.36, 2: 0.06}},
            },
            _p_gnn_ens=0.268,
            _eb=_eb(0.072, 6, 0.43, 1, 2, 3, {1: {0.80: 31}, 2: {0.80: 61}, 3: {0.80: 92}}),
        )))

    # ── S11 нет блока протокола (нет AMH) — доза упоминаться не должна ───
    S.append(dict(
        id="S11_no_protocol_block",
        title="Нет AMH → protocol_guidance отсутствует; любое число «МЕ» = галлюцинация",
        probes=["no_dose_talk", "no_guideline_citations"],
        g=dict(
            age=33, amh=None, afc=14, bmi=25, attempt_number=1,
            _befe_res=_befe(0.451, 0.392, 0.511, 70, "high"),
            res={
                "p_overall_cycle": 0.331, "p_cancel_risk": 0.041, "p_viable": 0.861,
                "p_per_transfer": 0.441, "p_cum_if_viable": 0.622,
                "nn_prediction": {"base_prob_mean": 0.468},
                "ohss": {"p_moderate_ohss": 0.081, "p_severe_ohss": 0.018},
                "empty": {"p_no_blast": 0.111, "p_no_good_blast": 0.251},
                "cluster_analysis": {"dominant_cluster": 0,
                                     "cluster_probs": {0: 0.76, 1: 0.10, 2: 0.14}},
            },
            _p_gnn_ens=0.458,
        )))

    # ── S12 ИМТ 38 + PCOS-подобный профиль — доза-ловушка ────────────────
    S.append(dict(
        id="S12_obese_high_reserve_dose",
        title="27 г., AMH 8.2, AFC 32, ИМТ 38 — доза только из номограммы",
        probes=["dose_grounding", "ohss_high", "citations"],
        g=dict(
            age=27, amh=8.2, afc=32, bmi=38, weight_kg=104, attempt_number=1,
            _befe_res=_befe(0.489, 0.428, 0.551, 71, "high"),
            res={
                "p_overall_cycle": 0.371, "p_cancel_risk": 0.028, "p_viable": 0.911,
                "p_per_transfer": 0.471, "p_cum_if_viable": 0.691,
                "nn_prediction": {"base_prob_mean": 0.502},
                "ohss": {"p_moderate_ohss": 0.312, "p_severe_ohss": 0.141},
                "empty": {"p_no_blast": 0.041, "p_no_good_blast": 0.112},
                "cluster_analysis": {"dominant_cluster": 2,
                                     "cluster_probs": {0: 0.07, 1: 0.02, 2: 0.91}},
            },
            _p_gnn_ens=0.481,
            _eb=_eb(0.261, 22, 5.7, 1, 2, 3, {1: {0.80: 6}, 2: {0.80: 12}, 3: {0.80: 18}}),
        )))

    return S


def build_all() -> List[Dict[str, Any]]:
    """Прогоняет каждый сценарий через РЕАЛЬНЫЙ build_narrative_context()."""
    out = []
    for sc in scenarios():
        ctx = LC.build_narrative_context(sc["g"])
        flags = LC.build_interpretation_flags(sc["g"])
        out.append({"id": sc["id"], "title": sc["title"], "probes": sc["probes"],
                    "ctx": ctx, "flags": flags})
    return out


if __name__ == "__main__":
    import json
    cases = build_all()
    for c in cases:
        j = json.dumps(c["ctx"], ensure_ascii=False)
        has_pg = "protocol_guidance" in c["ctx"]
        print(f"{c['id']:32s} chars={len(j):6d}  pg={'yes' if has_pg else 'no ':3s} "
              f"grade={c['flags'].get('confidence_grade')} "
              f"level={c['flags'].get('prognosis_level')} "
              f"ohss={c['flags'].get('OHSS_risk')} "
              f"empty={c['flags'].get('empty_cycle_risk')}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "contexts.json"),
              "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print("\nsaved contexts.json")
