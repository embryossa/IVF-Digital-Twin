# Coefficient Attribution

All coefficients in the IVF Digital Twin are publicly documented and traceable
to peer-reviewed literature. This page provides the complete attribution record.

## Categories

| Category | Meaning |
|---|---|
| **Imported** | Coefficients used verbatim from a single source, no modification |
| **Adapted** | Structure imported from a source; coefficients locally re-calibrated |
| **Novel** | Developed de novo for this platform |

## Full attribution table

| Component | Category | Source | Notes |
|---|---|---|---|
| S1: Oocyte yield mean formula | Adapted | ART-ONE, Merck KGaA [CONSORT/ENGAGE/ESTHER trials] | Intercept and AFC/AMH coefficients locally calibrated to clinic population |
| S1: Oocyte yield variance formula σ = 0.22·μ + 1.2 | Novel | Original estimation | Heteroscedastic noise model |
| S2: Maturity logistic — all coefficients | **Imported** | Herasight (Craig et al. 2025), Table A1 column 3, n=90,479 HFEA cycles | Used verbatim: 2.4665, 0.005, −0.782, 0.24, −0.069 |
| S3: Fertilisation logistic — all coefficients | **Imported** | Herasight (Craig et al. 2025), Table A1 column 5, n=90,088 HFEA cycles | Used verbatim: 1.1678, 0.004, −0.303, −0.051 |
| S4: Blastulation age formula — plateau and slope | Adapted | Romanski et al. 2022 (n=3,362); Sainte-Rose et al. 2021 (n=4,952) | Re-formulated as clip(0.70 − 0.012·max(0, age−40), 0.30, 0.75) |
| S5: Good-blastocyst fraction | Adapted | Internal clinic data + Herasight Serdj report cross-check | clip(0.78 − 0.008·max(0, age−35), 0.40, 0.85) |
| S6: Euploidy age table | **Imported** | Franasiak et al. 2014 (n=15,169) + Armstrong et al. 2023 (n=86,208) | 6-age-band lookup synthesised from both sources |
| S6b: Post-thaw survival 95% | **Imported** | Coello et al. 2021 | Fixed 95% per-blastocyst survival |
| L2: FORTUNE per-transfer logit structure | Adapted | FORTUNE (Carrasquillo et al. 2025, PMID 40889782) | Intercept recalibrated from 0.40 to −0.25 for local population |
| L2: KPIScore component thresholds | Novel | Original KPIScore system (Sergeev et al.) | Five-component ordinal score, 1/3/5 per component |
| L2: KPI-to-Beta probability mapping | Novel | Original | Moment-matching 95% CIs from KPIScore table to Beta(α, β) |
| L2: Logit-scale ensemble | Novel | Standard stacking practice (Steyerberg 2019) | Logit-weighted average with tunable w |
| L2: Three-level decomposition | Novel | Original contribution | per-transfer / cum-if-viable / overall |
| L3: KAT architecture (KAN + FT-Transformer) | Novel | Based on Liu et al. 2024 (KAN) + Gorishniy et al. 2021 (FTT) | Novel combination for IVF |
| L3: Venn-Abers conformal calibration | Novel (application) | Vovk & Petej 2012, venn-abers library | Standard conformal method applied to IVF NN |
| L3: NVSA correction | Novel | Original contribution | KPI-anchored correction with ±50% cap |
| L3: Beta-Binomial Bayesian posterior | Novel | Conjugate family (standard) | Integration of prior + real batches + NN pseudo-obs |
| L4: Cluster centroids | **Imported** | Sergeev et al. (under review), 1,556 cycles | 18-feature centroids from k-means k=3 |
| L4: Nearest-centroid assignment | Novel (application) | Standard k-means predict step | Per-MC-iteration, z-score standardised 18-D space |

## OHSS risk thresholds

| Risk class | Threshold | Source |
|---|---|---|
| Moderate OHSS | 15–19 retrieved oocytes | ESHRE OHSS Guideline 2023 |
| Severe OHSS | ≥ 20 retrieved oocytes | ESHRE OHSS Guideline 2023 + Humaidan 2010 |

## References

[1] CDC IVF Success Estimator. https://www.cdc.gov/art/ivf-success-estimator/

[6] Craig A et al. Stage-Structured, Distributional Prediction of IVF Outcomes
    with Conditional Updating. medRxiv 2025.09.27.25336680.

[11] Merck KGaA. ART-ONE. https://art-one.merckgroup.com/art

[14] Romanski PA et al. Age-specific blastocyst conversion rates.
     Reprod Biomed Online. 2022;45(3):432–439.

[15] Sainte-Rose R et al. Extended embryo culture is effective for patients of
     an advanced maternal age. Sci Rep. 2021;11(1):13499.

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

[30] Vovk V, Petej I. Venn-Abers predictors. arXiv:1211.0025. 2012.

[33] Sergeev S et al. Decoding IVF Laboratory Performance through
     Dimensionality Reduction and Cluster Analysis. (under review).
