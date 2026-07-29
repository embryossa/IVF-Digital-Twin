#!/usr/bin/env python3
# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
batch_predict.py — Run IVF Digital Twin predictions on a CSV file.

Usage:
    python scripts/batch_predict.py --input data/sample/sample_patients.csv \
                                    --output results.csv

CSV columns required:
    female_age, amh, afc, bmi

Optional CSV columns for mid-cycle Bayesian updating:
    okk, mii, pn2, blasts, good, euploid

Optional CSV columns for NN layer:
    attempt_number, follicles
"""

import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
exec(open(os.path.join(os.path.dirname(__file__), '..', 'src', 'ivf_digital_twin.py')).read()
     .replace("if __name__ ==", "if False and __name__ =="))


def predict_row(row, nn_model=None):
    patient = PatientInput(
        female_age = float(row['female_age']),
        amh        = float(row['amh']),
        afc        = int(row['afc']),
        bmi        = float(row['bmi']),
    )
    known = KnownValues(
        okk     = int(row['okk'])     if 'okk'     in row and pd.notna(row['okk'])     else None,
        mii     = int(row['mii'])     if 'mii'     in row and pd.notna(row['mii'])     else None,
        pn2     = int(row['pn2'])     if 'pn2'     in row and pd.notna(row['pn2'])     else None,
        blasts  = int(row['blasts'])  if 'blasts'  in row and pd.notna(row['blasts'])  else None,
        good    = int(row['good'])    if 'good'    in row and pd.notna(row['good'])    else None,
        euploid = int(row['euploid']) if 'euploid' in row and pd.notna(row['euploid']) else None,
    )
    attempt = int(row['attempt_number']) if 'attempt_number' in row and pd.notna(row.get('attempt_number', None)) else 1
    follicles = int(row['follicles']) if 'follicles' in row and pd.notna(row.get('follicles', None)) else None

    res = run_pipeline_extended(
        patient, known=known,
        attempt_number=attempt,
        follicles=follicles,
        nn_model=nn_model,
        max_attempts_curve=6,
    )

    ca = res['cluster_analysis']
    post = res['posterior']

    return {
        'okk_median':       int(res['okk_med']),
        'mii_median':       int(res['mii_med']),
        'pn2_median':       int(res['pn2_med']),
        'blasts_median':    int(res['blasts_med']),
        'good_median':      int(res['good_med']),
        'euploid_median':   int(res['euploid_med']),
        'warmed_median':    int(res['warmed_med']),
        'kpi_score_median': res['kpi_score_median'],
        'p_per_transfer':   round(res['p_per_transfer'], 4),
        'p_cum_if_viable':  round(res['p_cum_if_viable'], 4),
        'p_overall':        round(res['p_overall_cycle'], 4),
        'p_viable':         round(res['p_viable'], 4),
        'p_fortune':        round(res['p_per_transfer_fortune'], 4),
        'p_kpi':            round(res['p_per_transfer_kpi'], 4),
        'nn_base_prob':     round(res['nn_prediction']['base_prob_mean'], 4),
        'nn_nvsa_prob':     round(res['nn_nvsa']['adjusted_mean'], 4),
        'posterior_mean':   round(post['mean'], 4),
        'posterior_ci_lo':  round(post['ci_low'], 4),
        'posterior_ci_hi':  round(post['ci_high'], 4),
        'dominant_cluster': ca['dominant_cluster'],
        'cluster_name':     CLUSTER_INTERPRETATIONS[ca['dominant_cluster']]['name'],
        'p_cluster_0':      round(ca['cluster_probs'][0], 4),
        'p_cluster_1':      round(ca['cluster_probs'][1], 4),
        'p_cluster_2':      round(ca['cluster_probs'][2], 4),
        'ohss_severe':      round(res['ohss']['p_severe_ohss'], 4),
        'ohss_moderate':    round(res['ohss']['p_moderate_ohss'], 4),
        'empty_cycle':      round(res['empty']['p_no_blast'], 4),
    }


def main():
    parser = argparse.ArgumentParser(description="IVF Digital Twin — batch prediction")
    parser.add_argument("--input",  required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--n-sim",  type=int, default=1000,
                        help="Monte Carlo iterations per patient (default 1000 for batch speed)")
    args = parser.parse_args()

    # Override global N_SIM for batch speed
    global N_SIM
    N_SIM = args.n_sim

    # Try loading NN
    nn_model = load_nn_ensemble()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} patients from {args.input}")

    results = []
    for i, row in df.iterrows():
        np.random.seed(i)     # reproducible per-row
        try:
            out = predict_row(row, nn_model)
            results.append({**row.to_dict(), **out})
        except Exception as e:
            print(f"  Row {i}: ERROR — {e}")
            results.append(row.to_dict())

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
