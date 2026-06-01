"""
fit_befe_ood.py — fit the two OOD subspaces for BEFE (run ONCE, offline)
========================================================================

Computes mean + inverse covariance for:
    clinical    = [age, AMH, AFC, BMI]
    embryology  = [OCC, MII, 2PN, Blast, KPI]
on your training cohort, and saves them to  models/befe_ood_stats.npz.

The app loads that file at startup into st.session_state["_befe_ood_stats"],
which switches the BEFE OOD detector ON automatically (no other change).

USAGE:
    python fit_befe_ood.py path/to/clinical_protocols.xlsx
    python fit_befe_ood.py path/to/cohort.csv

IMPORTANT: the FEATURE ORDER here must match what befe_app uses at inference:
    clinical  = [age, amh, afc, bmi]
    embryo    = [okk(OCC), mii, pn2, blasts, kpi]
Edit COLUMN_MAP below to match your column names (Russian or English).
"""

import os
import sys
import numpy as np
import pandas as pd

from befe import fit_gaussian   # same dir as befe.py

# --------------------------------------------------------------------------- #
#  EDIT THIS to match your training file's column names.
#  Left = logical feature (do NOT change keys); right = your column header.
# --------------------------------------------------------------------------- #
COLUMN_MAP = {
    # clinical subspace
    "age":   "Возраст",
    "amh":   "АМГ",
    "afc":   "АФС",
    "bmi":   "ИМТ",
    # embryology subspace
    "okk":   "Число ОКК",
    "mii":   "MII",
    "pn2":   "2 pN",
    "blast": "Число Bl",
    "kpi":   "KPIScore",
}

CLINICAL = ["age", "amh", "afc", "bmi"]
EMBRYO   = ["okk", "mii", "pn2", "blast", "kpi"]

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "models", "befe_ood_stats.npz")


def _load(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _subspace_matrix(df: pd.DataFrame, logical_cols):
    cols = []
    for lc in logical_cols:
        src = COLUMN_MAP[lc]
        if src not in df.columns:
            raise KeyError(
                f"Column '{src}' (for '{lc}') not found. "
                f"Available: {list(df.columns)[:20]}... "
                f"Edit COLUMN_MAP in fit_befe_ood.py.")
        cols.append(src)
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    return sub.values.astype(float), cols


def main(path: str):
    df = _load(path)
    print(f"Loaded {len(df)} rows from {path}")

    clin_X, clin_cols = _subspace_matrix(df, CLINICAL)
    emb_X,  emb_cols  = _subspace_matrix(df, EMBRYO)
    print(f"Clinical  : {clin_X.shape[0]} complete rows over {clin_cols}")
    print(f"Embryology: {emb_X.shape[0]} complete rows over {emb_cols}")

    if clin_X.shape[0] < 30 or emb_X.shape[0] < 30:
        print("WARNING: <30 complete rows in a subspace — covariance may be "
              "unstable. Proceeding with regularisation anyway.")

    clin_mu, clin_ci = fit_gaussian(clin_X)
    emb_mu,  emb_ci  = fit_gaussian(emb_X)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(
        OUT_PATH,
        clinical_mu=clin_mu, clinical_cov_inv=clin_ci,
        embryo_mu=emb_mu,   embryo_cov_inv=emb_ci,
        clinical_order=np.array(CLINICAL),
        embryo_order=np.array(EMBRYO),
    )
    print(f"\nSaved OOD stats -> {OUT_PATH}")
    print("clinical mean:", np.round(clin_mu, 2))
    print("embryo   mean:", np.round(emb_mu, 2))
    print("\nRestart the app — the BEFE OOD detector is now ON.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fit_befe_ood.py <training_file.csv|.xlsx>")
        sys.exit(1)
    main(sys.argv[1])
