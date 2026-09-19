# Coefficient Attribution — IVF Digital Twin 7.1

Every coefficient is traceable to its source: peer-reviewed literature, or a
documented local refit. The L1 values in use are the constants in
`src/embryology.py`; this page records where they come from.

## Categories

| Category | Meaning |
|---|---|
| **Imported** | Coefficients used verbatim from a single source, no modification |
| **Adapted** | Structure imported from a source; coefficients locally re-calibrated |
| **Fitted** | Estimated in 7.1 on local cohorts (data named in the Notes column) |
| **Novel** | Developed de novo for this platform |

## Full attribution table

| Component | Category | Source | Notes |
|---|---|---|---|
| S1: Oocyte yield — negative binomial, log link | **Fitted** | Stimulated cycles, n = 687 with OCC ≥ 1; zero-truncated NB, 5-fold patient CV (2026-09-11) | μ = exp(0.05976 + 0.26843·ln(AMH+0.1) + 0.76892·ln(AFC+1) − 0.00393·(age−35)), θ = 5.772. Replaces the ART-ONE linear predictor, which under-estimated low responders (AFC < 5: 1.4 predicted vs 2.7 observed). Blank OCC cells are explained by NB zeros plus an unrecorded rate, so no structural-zero component is fitted; BMI is not recorded and has no term |
| S2: Maturity logistic — all coefficients | **Imported** | Herasight (Craig et al. 2025), Table A1 column 3, n = 90,479 HFEA cycles | Used verbatim: 2.4665, 0.005, −0.782, 0.24, −0.069 |
| S3: Fertilisation logistic — all coefficients | **Imported** | Herasight (Craig et al. 2025), Table A1 column 5, n = 90,088 HFEA cycles | Used verbatim: 1.1678, 0.004, −0.303, −0.051 |
| S4: Blastocyst yield — zero-inflated beta-binomial | **Fitted** | protocols_15k, culture-observed cycles, n = 7,919 (day-5/6/7 transfers recorded with Bl = 0 excluded) | logit m = 0.61431 − 0.016956·max(0, age−35) + 0.019593·ln(min(2PN, 5)); κ = 7.823; π₀ = 0.0095 |
| S5: Good-quality fraction — beta-binomial | **Fitted** | protocols_15k | logit m = 0.08825 − 0.063516·max(0, age−35) + 0.35722·ln(Blast); κ = 4.131 |
| S6: Euploidy age table | **Imported** | Franasiak et al. 2014 (n = 15,169) + Armstrong et al. 2023 (n = 86,208) | 6-band lookup; Beta dispersion 6 |
| Warming survival 95% | **Imported** | Coello et al. 2021 | Fixed 95% per blastocyst |
| Fair- vs good-quality transfer | **Fitted** | Day-5 single blastocyst transfers, n = 2,975 (399 fair only), adjusted for age and ln(2PN) | log OR = −0.6664 (OR 0.51, 95% CI 0.35–0.75) |
| Transfer frailty | **Fitted** | Retrievals recording the outcome of every transfer: 1,703 retrievals, 2,406 transfers, 461 with ≥ 2 | Shared logit-normal effect σ = 0.746 (95% profile CI 0.29–1.13; latent ICC 0.145); 0.732 after adjusting for age and blastocyst count. A never-implants share was not identifiable and is not modelled |
| Follicles at puncture from OCC | **Fitted** | protocols_15k, median OCC / punctured follicles | 0.846 |
| L2: FORTUNE per-transfer logit structure | Adapted | FORTUNE (Carrasquillo et al. 2025, PMID 40889782) | 0.40 − 0.55·z_age + 0.15·z_lnAMH − 0.20·z_BMI |
| L2: KPIScore component thresholds | Novel | Original KPIScore system (Sergeev et al.) | Five-component ordinal score, 1/3/5 per component. L1 uses the AMH form; KAT/GAT/CSDI/NVSA use the follicle form recorded in their training data |
| L2: KPI-to-Beta probability mapping | Novel | Original | Moment-matching 95% CIs from the KPIScore table to Beta(α, β) |
| L2: Logit-scale ensemble | Novel | Standard stacking practice (Steyerberg 2019) | Logit-weighted average, w = 0.5 |
| L2: Three-level decomposition | Novel | Original contribution | per transfer / if transferable / whole cycle |
| L3: KAT architecture (KAN + FT-Transformer) | Novel | Based on Liu et al. 2024 (KAN) + Gorishniy et al. 2021 (FTT) | Novel combination for IVF |
| L3: Interpolated isotonic calibration | Novel (application) | Isotonic regression (standard) | Linear interpolation through step centres; same fit as the step calibrator on protocols_15k (Brier within 0.0001) |
| L3: NVSA correction | Novel | Original contribution | KPI-anchored correction with ±50% cap |
| L3: Beta-Binomial Bayesian posterior | Novel | Conjugate family (standard) | Covariate-dependent prior (κ = 20) + clinic batches + NN pseudo-observations (100) |
| L4: Cluster centroids | **Imported** | Sergeev et al. (under review), 1,556 cycles | 18-feature centroids from k-means k = 3 |
| L4: Nearest-centroid assignment | Novel (application) | Standard k-means predict step | Per scenario, z-score standardised 18-D space |
| L7: Evidence fusion | Novel | Conjugate Gaussian approximation to Bayesian model averaging | Precision-weighted logit pooling |
| L7: Uncertainty range — model term | Novel (application) | DerSimonian & Laird 1986 | Random-effects variance of the fused L1/KAT/GAT estimate |
| L7: Reliability weights | Novel | Original | 0.40 consensus / 0.30 diffusion / 0.20 graph / 0.10 cluster; bands 70 / 45 |
| Banking: euploid blastocyst per MII | **Imported** | Esteves et al. 2019, Front Endocrinol 10:99 (POSEIDON ART calculator), Table 2 | Logistic in age by sperm-source stratum |
| TRP: selection decay per attempt | **Imported** | Malizia et al. 2009 | 0.08 on the logit scale |
| TRP: AMH decline | Adapted | Dölleman et al. 2013; Tehrani et al. 2011 | AMH(t) = AMH₀·exp(−k·t), ln k ~ N(ln 0.07, 0.45) |

## OHSS risk thresholds

| Risk class | Threshold | Source |
|---|---|---|
| Moderate OHSS | 15–19 retrieved oocytes | ESHRE OHSS Guideline 2023 |
| Severe OHSS | ≥ 20 retrieved oocytes | ESHRE OHSS Guideline 2023 + Humaidan 2010 |

## What 7.1 no longer uses

| 7.0 component | Replaced by |
|---|---|
| S1 ZINB with ART-ONE linear mean (θ = 5.0) and logistic zero-inflation | S1 NB with log link (θ = 5.772), no structural zero |
| S1 truncated-normal variance σ = 0.22·μ + 1.2 (pre-6.2) | NB dispersion |
| S4 clip(0.70 − 0.012·max(0, age−40), 0.30, 0.75) | S4 zero-inflated beta-binomial |
| S5 clip(0.78 − 0.008·max(0, age−35), 0.40, 0.85) | S5 beta-binomial conditional on the blastocyst count |
| Transfers limited to euploid embryos, independent transfers | Transfer scenario (PGT-A or all blastocysts) with shared frailty |
| Venn-Abers / step isotonic KAT calibration | Interpolated isotonic calibration |
| Clinic temperature and dynamic τ (calibrate_for_clinic.py) in the clinical path | Not applied; research diagnostics only |

## References

[1] CDC IVF Success Estimator. https://www.cdc.gov/art/ivf-success-estimator/

[6] Craig A et al. Stage-Structured, Distributional Prediction of IVF Outcomes
    with Conditional Updating. medRxiv 2025.09.27.25336680.

[18] Franasiak JM et al. The nature of aneuploidy with increasing age.
     Fertil Steril. 2014;101(3):656–663.

[19] Armstrong A et al. F&S Reports. 2023;4(3):256–261.

[20] Coello A et al. Reprod Biomed Online. 2021;42(5):881–891.

[21] ESHRE Guideline: ovarian stimulation for IVF/ICSI.
     Hum Reprod Open. 2023;2023(1):hoad006.

[23] Carrasquillo R et al. FORTUNE (IVIRMA). Hum Reprod. 2025. PMID:40889782.

[25] SART National Summary Report 2023.
     https://www.sartcorsonline.com/rptCSR_PublicMultYear.aspx?reportingYear=2023

[27] Liu Z et al. KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756. 2024.

[29] Gorishniy Y et al. Revisiting Deep Learning Models for Tabular Data.
     NeurIPS. 2021.

[33] Sergeev S et al. Decoding IVF Laboratory Performance through
     Dimensionality Reduction and Cluster Analysis. (under review).

[34] Esteves SC et al. Front Endocrinol. 2019;10:99. doi:10.3389/fendo.2019.00099.

[35] DerSimonian R, Laird N. Meta-analysis in clinical trials.
     Control Clin Trials. 1986;7(3):177–188.

[36] Malizia BA, Hacker MR, Penzias AS. Cumulative live-birth rates after in
     vitro fertilization. N Engl J Med. 2009;360(3):236–243.

Earlier references [11] ART-ONE (Merck KGaA), [14] Romanski et al. 2022 and
[15] Sainte-Rose et al. 2021 described the 7.0 S1 and S4 formulas.
