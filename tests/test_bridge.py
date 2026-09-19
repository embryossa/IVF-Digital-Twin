# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""The 7.1 data flow used by app.py (dt_bridge): L1-L6 -> L7 -> headline -> TRP anchor.

Runs without model weights, as in CI: the neural layers are unavailable and
BEFE fuses the L1 prior alone.
"""
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import dt_bridge  # noqa: E402

BASE = dict(age=35.0, amh=2.5, afc=15, bmi=23.0, attempt=1, follicles=None,
            sperm_source="ejaculate", known_okk=None, known_mii=None, known_pn2=None,
            known_blasts=None, known_good=None, known_euploid=None, n_sim=500, seed=42)


@pytest.fixture(scope="module")
def open_cycle():
    return dt_bridge.compute(BASE)


def test_headline_is_per_transfer(open_cycle):
    summary = open_cycle["clinical_summary"]
    assert summary["kind"] == "per_transfer"
    assert not summary["no_transfer_confirmed"]
    assert summary["probability"] == open_cycle["headline"]
    assert 0 < summary["probability"] < 1


def test_cycle_probability_in_unit_interval(open_cycle):
    assert 0 < open_cycle["cycle_probability"] < 1


def test_same_seed_same_result(open_cycle):
    again = dt_bridge.compute(BASE)
    assert again["headline"] == open_cycle["headline"]
    assert again["cycle_probability"] == open_cycle["cycle_probability"]


def test_trp_anchor_is_current_cycle(open_cycle):
    value, source = dt_bridge.trp_anchor(open_cycle)
    assert source == "current-L1-L7"
    assert value == pytest.approx(open_cycle["cycle_probability"])


def test_closed_cycle_headline_zero_and_prospective_anchor(open_cycle):
    closed = dt_bridge.compute(dict(BASE, known_okk=5, known_mii=4, known_pn2=0))
    assert closed["clinical_summary"] == {"probability": 0.0, "kind": "current_cycle",
                                          "no_transfer_confirmed": True}
    assert closed["cycle_probability"] == 0.0
    value, source = dt_bridge.trp_anchor(closed)
    # A fresh cycle without the current observations, not the zero of this one.
    assert source == "prospective-L1-L7"
    assert value == pytest.approx(open_cycle["cycle_probability"])


def test_analytics_row_carries_l7_kat(open_cycle, tmp_path):
    import csv
    from ivf_core import save_analytics_record, _ANALYTICS_COLUMNS

    path = tmp_path / "dt_predictions.csv"
    # A file written with an older schema is archived, not appended to.
    path.write_text("record_id,p_kat_raw\nold,0.5\n", encoding="utf-8")
    p = open_cycle["patient"]
    rid = save_analytics_record(
        result=open_cycle, age=p["age"], amh=p["amh"], afc=p["afc"], bmi=p["bmi"],
        attempt=p["attempt"], sperm_source=p["sperm_source"], follicles=p["follicles"],
        analytics_csv=str(path))
    assert rid is not None
    assert len(list(tmp_path.glob("dt_predictions_schema_v2_*.csv"))) == 1
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == _ANALYTICS_COLUMNS
    row = rows[0]
    kat = open_cycle["p_kat_raw"]   # the KAT value that enters L7
    if kat is None:
        assert row["kat_transfer_mean"] == ""
    else:
        assert float(row["kat_transfer_mean"]) == pytest.approx(kat, abs=1e-4)
    # p_kat_raw keeps the 7.0 definition: nn_prediction over all scenarios.
    full = open_cycle["res"]["nn_prediction"]["base_prob_mean"]
    assert float(row["p_kat_raw"]) == pytest.approx(full, abs=1e-4)


def test_reliability_band_thresholds():
    assert dt_bridge.reliability_band(80, 70, 45) == "High"
    assert dt_bridge.reliability_band(50, 70, 45) == "Moderate"
    assert dt_bridge.reliability_band(10, 70, 45) == "Low"
