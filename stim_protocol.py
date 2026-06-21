"""
stim_protocol.py — IVF Digital Twin
Deterministic stimulation-protocol module (decision SUPPORT, not prescription).

Role in the architecture
-------------------------
This is a *calculator*, not a RAG component and not an LLM call. It maps the
patient features the system already has (age, AMH, AFC, BMI) onto:
  - a response phenotype (ORT-based, independent of the ML cluster layer),
  - an OHSS risk level and the corresponding mitigation levers,
  - a bounded target oocyte yield,
  - a suggested STARTING-DOSE BAND (IU/day, conventional gonadotrophin),
  - optional exact follitropin-delta (ug/day) when body weight is supplied,
  - a set of `situation_keys` used by guideline_rag.py to pull the matching
    published recommendations.

Every number is deterministic and traceable to stim_params.json. The LLM never
sees this code; it only narrates the resulting JSON. The physician sets the
final dose. All thresholds live in stim_params.json (versioned + cited).

Important honesty note
----------------------
The IU dose bands encode published *principles* (ORT-based dosing, conventional
150-225 IU band, reduce in high responders, ceiling in poor responders because
higher FSH does not improve live birth). They are a transparent synthesis, not a
single validated black-box formula. The follitropin-delta branch is the only
exact published drug nomogram here and ships DISABLED until its table is verified
against the current SmPC (see stim_params.json).

Dependencies: standard library only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

_PARAMS_PATH = os.environ.get(
    "DT_STIM_PARAMS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "stim_params.json"),
)

# Embedded fallback so the module never hard-fails if the JSON is missing.
_FALLBACK_PARAMS: Dict[str, Any] = {
    "_meta": {"version": "fallback", "amh_pmol_per_ng_ml": 7.14},
    "phenotype_thresholds": {
        "high_responder": {"amh_ng_ml_min": 3.4, "afc_min": 18, "logic": "OR"},
        "poor_responder": {"amh_ng_ml_max": 1.1, "afc_max": 5, "logic": "OR"},
    },
    "ohss_thresholds": {"amh_ng_ml_elevated": 3.4, "afc_elevated": 18,
                        "amh_ng_ml_high": 5.0, "afc_high": 24},
    "target_yield": {"normal_responder": [8, 14], "high_responder": [8, 12],
                     "poor_responder": [4, 8]},
    "dose_bands_iu": {
        "normal_responder": {"low": 150, "high": 225},
        "high_responder": {"low": 100, "high": 150},
        "poor_responder": {"low": 225, "high": 300,
                           "ceiling_note": "higher_dose_no_LBR_gain"},
    },
    "dose_modifiers": {
        "bmi": {"apply_above": 30.0, "delta_iu": 25},
        "age": {"apply_at_or_above": 40, "delta_iu": 25},
    },
    "follitropin_delta": {"enabled": False},
}


def _load_params() -> Dict[str, Any]:
    try:
        with open(_PARAMS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return _FALLBACK_PARAMS


# ──────────────────────────────────────────────────────────────────────────
#  INPUT / OUTPUT
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class StimInput:
    age: float
    amh_ng_ml: float
    afc: int
    bmi: float = 24.0
    weight_kg: Optional[float] = None          # only needed for follitropin-delta
    protocol_pref: str = "auto"                # "auto" | "antagonist" | "agonist"


@dataclass
class StimOutput:
    response_phenotype: str
    phenotype_reasons: List[str]
    ohss_risk: str                             # "low" | "elevated" | "high"
    protocol_type: str                         # "GnRH-antagonist" | "GnRH-agonist"
    target_oocyte_yield: Tuple[int, int]
    suggested_start_dose_iu: Tuple[int, int]
    dose_caveats: List[str]
    mitigation_levers: List[str]
    follitropin_delta_ug: Optional[float]
    situation_keys: List[str]
    param_version: str
    disclaimer: str = ("Поддержка решения, не назначение. Итоговую дозу и протокол "
                       "определяет лечащий врач.")

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["target_oocyte_yield"] = list(self.target_oocyte_yield)
        d["suggested_start_dose_iu"] = list(self.suggested_start_dose_iu)
        return d


# ──────────────────────────────────────────────────────────────────────────
#  DETERMINISTIC LOGIC
# ──────────────────────────────────────────────────────────────────────────
def _classify_phenotype(inp: StimInput, p: Dict[str, Any]) -> Tuple[str, List[str]]:
    th = p["phenotype_thresholds"]
    hi, lo = th["high_responder"], th["poor_responder"]
    reasons: List[str] = []

    high = (inp.amh_ng_ml >= hi["amh_ng_ml_min"]) or (inp.afc >= hi["afc_min"])
    poor = (inp.amh_ng_ml <= lo["amh_ng_ml_max"]) or (inp.afc <= lo["afc_max"])

    # High takes precedence over poor if both somehow fire (discordant markers).
    if high:
        if inp.amh_ng_ml >= hi["amh_ng_ml_min"]:
            reasons.append(f"AMH {inp.amh_ng_ml} ≥ {hi['amh_ng_ml_min']} нг/мл")
        if inp.afc >= hi["afc_min"]:
            reasons.append(f"AFC {inp.afc} ≥ {hi['afc_min']}")
        return "high_responder", reasons
    if poor:
        if inp.amh_ng_ml <= lo["amh_ng_ml_max"]:
            reasons.append(f"AMH {inp.amh_ng_ml} ≤ {lo['amh_ng_ml_max']} нг/мл")
        if inp.afc <= lo["afc_max"]:
            reasons.append(f"AFC {inp.afc} ≤ {lo['afc_max']}")
        return "poor_responder", reasons
    return "normal_responder", ["AMH/AFC в среднем диапазоне"]


def _ohss_risk(inp: StimInput, phenotype: str, p: Dict[str, Any]) -> str:
    th = p["ohss_thresholds"]
    if inp.amh_ng_ml >= th["amh_ng_ml_high"] or inp.afc >= th["afc_high"]:
        return "high"
    if (phenotype == "high_responder"
            or inp.amh_ng_ml >= th["amh_ng_ml_elevated"]
            or inp.afc >= th["afc_elevated"]):
        return "elevated"
    return "low"


def _dose_band(inp: StimInput, phenotype: str,
               p: Dict[str, Any]) -> Tuple[Tuple[int, int], List[str]]:
    band = p["dose_bands_iu"][phenotype]
    low, high = int(band["low"]), int(band["high"])
    caveats: List[str] = []

    mods = p.get("dose_modifiers", {})
    # BMI / age nudges apply ONLY to normal responders:
    #  - high responders get a deliberately reduced dose (OHSS prophylaxis) — no upward nudge;
    #  - poor responders are already at the ceiling (higher FSH gives no LBR gain) — no nudge.
    if phenotype == "normal_responder":
        bmi_m = mods.get("bmi", {})
        if bmi_m and inp.bmi >= float(bmi_m.get("apply_above", 1e9)):
            d = int(bmi_m["delta_iu"])
            low, high = low + d, high + d
            caveats.append(f"ИМТ {inp.bmi} → +{d} МЕ (слабая доказательность)")
        age_m = mods.get("age", {})
        if age_m and inp.age >= float(age_m.get("apply_at_or_above", 1e9)):
            d = int(age_m["delta_iu"])
            low, high = low + d, high + d
            caveats.append(f"Возраст {int(inp.age)} → +{d} МЕ (слабая доказательность)")

    if phenotype == "poor_responder" and band.get("ceiling_note"):
        caveats.append("потолок дозы: выше не повышает живорождение")
    if phenotype == "high_responder":
        caveats.append("сниженная доза как мера профилактики СГЯ")

    return (low, high), caveats


def _follitropin_delta(inp: StimInput, p: Dict[str, Any]) -> Optional[float]:
    fd = p.get("follitropin_delta", {})
    if not fd.get("enabled", False):
        return None
    if inp.weight_kg is None or inp.weight_kg <= 0:
        return None
    conv = float(p.get("_meta", {}).get("amh_pmol_per_ng_ml", 7.14))
    amh_pmol = inp.amh_ng_ml * conv
    if amh_pmol < float(fd["amh_pmol_fixed_dose_below"]):
        return round(float(fd["fixed_dose_ug"]), 1)
    ug_kg = None
    for row in fd.get("ug_per_kg_table_placeholder", []):
        if row["amh_pmol_min"] <= amh_pmol <= row["amh_pmol_max"]:
            ug_kg = float(row["ug_per_kg"])
            break
    if ug_kg is None:
        return None
    dose = ug_kg * float(inp.weight_kg)
    dose = max(float(fd["min_ug"]), min(float(fd["max_ug"]), dose))
    return round(dose, 1)


def _protocol_type(inp: StimInput, ohss_risk: str) -> str:
    if inp.protocol_pref == "agonist":
        return "GnRH-agonist"
    if inp.protocol_pref == "antagonist":
        return "GnRH-antagonist"
    # auto: antagonist whenever OHSS is a concern (ASRM 2023 preference).
    return "GnRH-antagonist" if ohss_risk in ("elevated", "high") else "GnRH-antagonist"


def _levers_and_keys(phenotype: str, ohss_risk: str,
                     inp: StimInput) -> Tuple[List[str], List[str]]:
    levers: List[str] = []
    keys: List[str] = [phenotype]

    if ohss_risk in ("elevated", "high"):
        keys.append("ohss_risk_elevated")
        levers.append("предпочесть GnRH-антагонист")
        levers.append("снизить стартовую дозу гонадотропина")
        keys.append("antagonist_preferred")
    if ohss_risk == "high":
        keys += ["agonist_trigger_candidate", "freeze_all_candidate"]
        levers.append("рассмотреть агонист-триггер")
        levers.append("рассмотреть freeze-all для устранения позднего СГЯ")
    if phenotype == "high_responder":
        keys.append("amh_high")
    if phenotype == "poor_responder":
        keys += ["amh_low", "poor_response_dose_ceiling"]
        levers.append("без эскалации дозы выше потолка (нет выигрыша в LBR)")

    # de-duplicate, preserve order
    seen = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]
    return levers, keys


def compute_stim(inp: StimInput,
                 params: Optional[Dict[str, Any]] = None) -> StimOutput:
    """Main entry point: patient features -> deterministic protocol guidance."""
    p = params or _load_params()

    phenotype, reasons = _classify_phenotype(inp, p)
    ohss = _ohss_risk(inp, phenotype, p)
    band, caveats = _dose_band(inp, phenotype, p)
    yld = tuple(int(x) for x in p["target_yield"][phenotype])  # type: ignore
    fd_ug = _follitropin_delta(inp, p)
    proto = _protocol_type(inp, ohss)
    levers, keys = _levers_and_keys(phenotype, ohss, inp)

    return StimOutput(
        response_phenotype=phenotype,
        phenotype_reasons=reasons,
        ohss_risk=ohss,
        protocol_type=proto,
        target_oocyte_yield=yld,            # type: ignore
        suggested_start_dose_iu=band,
        dose_caveats=caveats,
        mitigation_levers=levers,
        follitropin_delta_ug=fd_ug,
        situation_keys=keys,
        param_version=str(p.get("_meta", {}).get("version", "unknown")),
    )


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick self-test across the three phenotypes.
    for label, inp in [
        ("high",   StimInput(age=31, amh_ng_ml=5.6, afc=26, bmi=23)),
        ("normal", StimInput(age=34, amh_ng_ml=2.1, afc=12, bmi=26)),
        ("poor",   StimInput(age=41, amh_ng_ml=0.6, afc=4,  bmi=31)),
    ]:
        out = compute_stim(inp)
        print(f"\n=== {label} ===")
        print(json.dumps(out.as_dict(), ensure_ascii=False, indent=2))
