# IVF Digital Twin v6.2

**An Integrated Multi-Source Ensemble Platform for Stage-Stratified IVF Outcome Prediction**

> Stochastic Simulation · Neural Network Ensembling · Bayesian Evidence Synthesis · Unsupervised Phenotype Classification · Diffusion-Based Generative Module · Graph Attention Network

*Sergeev et al., 2025 · embryossa@gmail.com*  
*Research prototype — not for standalone clinical use*

---

## Overview

Current clinical decision-support tools for in vitro fertilization (IVF) typically address isolated endpoints and return point estimates without quantifying the substantial biological variability inherent to reproductive medicine. They cannot incorporate stage-by-stage information as the cycle unfolds, do not benchmark a given patient against documented IVF protocol phenotypes, and do not integrate independent generative verification of laboratory outcome distributions.

**IVF Digital Twin v6.2** is a five-layer (+ GAT) probabilistic prediction system that models the entire IVF treatment trajectory as a sequential probabilistic pipeline. It is the first IVF prediction system to integrate:

- Stochastic stage-wise Monte Carlo simulation
- Multi-source per-transfer ensemble prediction
- Calibrated neural-network prediction (KAT: KAN + FT-Transformer)
- Bayesian evidence synthesis with mid-cycle conditional updating
- Unsupervised phenotype classification
- Diffusion-based generative laboratory module (CSDI Hybrid v3)
- Graph Attention Transformer for patient-similarity reasoning (L6)

…within a single transparent, auditable framework. All coefficients are traceable to peer-reviewed sources.

---

## Key Validation Results

| Component | Metric | Value |
|-----------|--------|-------|
| Stochastic pipeline | Spearman ρ (oocyte count) | 0.73 |
| Stochastic pipeline | Spearman ρ (blastocyst count) | 0.41 |
| Overall pregnancy prediction | AUC | 0.63 |
| Overall pregnancy prediction | Brier Score | 0.22 |
| CSDI Hybrid v3 (L5) | AUC | 0.661 |
| CSDI Hybrid v3 (L5) | Brier Score | 0.209 |
| CSDI Hybrid v3 (L5) | ECE | 0.029 |
| CSDI Hybrid v3 (L5) | Prevalence bias | −0.1 pp |
| GAT Graph Transformer (L6) | AUC-ROC | 0.632 |
| GAT + KAT Ensemble | AUC-ROC | ~0.658–0.665 |

---

## Architecture

The system is organized into six layers that operate in series:

```
┌──────────────────────────────────────────────────────────────┐
│                    IVF Digital Twin v6.2                     │
│                                                              │
│  Patient inputs: age, AMH, AFC, BMI, attempt number         │
│                         │                                    │
│  L1 ─ Stochastic Monte Carlo pipeline (N=5,000 iterations)  │
│       ZINB oocyte model → 7 sequential biological stages    │
│                         │                                    │
│  L2 ─ Per-transfer ensemble (FORTUNE + KPIScore)            │
│       Three-level pregnancy decomposition                    │
│                         │                                    │
│  L3 ─ KAT Neural Network Ensemble                           │
│       KAN (B-spline) + FT-Transformer + Venn-Abers calib.  │
│       Beta-Binomial Bayesian posterior synthesis             │
│                         │                                    │
│  L4 ─ Unsupervised Cluster Classifier                        │
│       Nearest-centroid in 18-dim space (k=3 phenotypes)     │
│                         │                                    │
│  L5 ─ CSDI Hybrid v3 Diffusion Module                       │
│       CSDI Transformer + LightGBM + Conformal Prediction    │
│                         │                                    │
│  L6 ─ GAT Graph Attention Transformer                       │
│       Patient-similarity graph · 1,172 clinical protocols   │
│                                                              │
│  OUTPUT: Full probability distributions, uncertainty        │
│          intervals, phenotype label, PDF clinical report    │
└──────────────────────────────────────────────────────────────┘
```

### Layer 1 — Stochastic Monte Carlo Pipeline

Models the IVF cycle as a sequence of seven probabilistic stages. Patient inputs are female age, AMH (ng/mL), AFC, and BMI.

- **Stage 1** — Retrieved oocytes: Zero-Inflated Negative Binomial (ZINB) distribution with a logistic zero-inflation component. Calibrated against published cancellation rates (Craig et al., HFEA registry); dispersion θ = 5.0.
- **Stage 2** — Mature (MII) oocytes: logistic-binomial filter
- **Stage 3** — Two-pronuclear zygotes (2PN): binomial filter
- **Stage 4** — Blastocysts: age-stratified rates (Romanski et al., 2022)
- **Stage 5** — Good-quality blastocysts: Beta-binomial
- **Stage 6** — Euploid embryos: age-stratified aneuploidy table (Franasiak et al., 2014)
- **Stage 6b** — Post-thaw survival (Coello et al., 2021)

Explicit risk quantification: OHSS probability and empty-cycle risk are computed per iteration.

### Layer 2 — Per-Transfer Ensemble

Combines two independent pregnancy probability sources via logit-scale weighting:
- **FORTUNE-based** logistic regression on clinical predictors (Carrasquillo et al., 2025)
- **KPIScore-based** laboratory performance metric (score 5–25, Beta-distributed 95% CIs per integer)

Produces three-level pregnancy decomposition: per-transfer, cumulative-if-viable, and overall cycle.

### Layer 3 — KAT Neural Network + Bayesian Posterior

- **KAN** (3-layer, B-spline activations, Liu et al. 2024) + **FT-Transformer** (Gorishniy et al., NeurIPS 2021) ensemble
- Venn-Abers conformal calibration
- NVSA correction for attempt-number-dependent probability decay
- **Beta-Binomial conjugate Bayesian posterior** fusing: NN output, covariate-dependent Beta regression prior, and retrospective clinic data

Mid-cycle conditional updating: posterior is re-conditioned as each new stage observation becomes available (known OCC, 2PN, blastocyst counts).

### Layer 4 — Unsupervised Cluster Classifier

Nearest-centroid assignment to three published IVF protocol phenotypes (Poor / Standard / High responder):
- 18-dimensional z-score-standardized feature space
- Centroids from 1,556 cycles, k-means k=3 (Sergeev et al., manuscript under review)
- PCA visualization with synthetic cohort clouds

### Layer 5 — CSDI Hybrid v3 Diffusion Module

A two-stage generative model trained on ~15,000 IVF cycles that independently reconstructs embryological outcome distributions without parametric rate assumptions.

**Architecture:**

```
CONDITIONING INPUTS (7):
follicle count, OCC, inseminated oocytes, 2PN,
OCC retrieval rate, fertilization rate, KPIScore

STAGE 1: CSDI TRANSFORMER (Generative)
  QuantileNormalizer → CSDIDenoiser (Transformer)
  • 4 layers × 4 heads × hidden=128
  • DDIM sampling (50 steps, T=1000 cosine schedule)
  • Generates: Число Bl, Число Bl хор.кач-ва
  • Derives analytically: blast_rate, good_rate ∈ [0,1]

STAGE 2: LIGHTGBM + PLATT SCALING
  Input: 7 conditioning + 2 count medians → P(pregnancy)
  DART boosting + Platt calibration → ECE = 0.029

STAGE 3: SPLIT CONFORMAL PREDICTION
  Distribution-free coverage guarantees
  90% PI → actual ~91–93% coverage
```

**Why this architecture?** The previous TabDDPM v3 (FiLM-ResNet) had a +15.8 pp prevalence bias and AUROC 0.578. CSDI Hybrid v3 resolves this by separating count generation (diffusion) from binary prediction (discriminative classifier), reducing ECE from 0.158 to 0.029 and removing the bias entirely.

**Training:** 15,193 IVF cycles, 3-way split (85% diffusion / 7.5% LGB / 7.5% conformal calibration). AdamW, cosine annealing, 200 epochs.

**Saved model structure** (not included — see [Model Availability](#model-availability)):
```
models/embryo_v3_model/
├── config.json
├── csdi_weights.pt       # CSDIDenoiser state dict
├── normalizer.pt         # QuantileNormalizer (numpy arrays)
├── lgb_state.pt          # LGBMClassifier state
├── platt_calibrator.pt   # Platt scaling weights
├── conformal.pt          # Conformal radii
└── training_history.json
```

### Layer 6 — Graph Attention Transformer (GAT)

A patient-similarity graph model that formalizes the clinical intuition of "I've seen patients like this before."

**Clinical intuition:** Rather than treating each patient as an isolated data point, GAT constructs a clinical similarity network across the entire cohort. Each patient is a node in an 18-dimensional space; edges are weighted by cosine similarity of clinical and embryological features. The graph topology is determined exclusively by clinical parameters — KAT scores are deliberately excluded from edge construction to ensure the graph reflects biological resemblance, not prior model beliefs.

**Architecture:**
- **Message passing with learned attention**: each node aggregates information from neighbours, weighting by learned attention scores
- **3 GAT layers** → node embedding encodes both individual characteristics and neighbourhood context
- **Two output heads**: P(pregnancy) classification + continuous PRAI score (multi-task)
- **18 node features**: age, AFC, OCC, fertilization outcomes, blastocyst counts, quality rates, KPI Score, KAT ensemble probability

**Training:** 1,172 real IVF clinical protocols from a single centre. Transductive cross-validation (5-fold): validation patients remain visible in the graph but outcome labels are masked. Label smoothing ε = 0.08, L2 regularisation, attention dropout.

**Ensemble:** The final GAT Ensemble score combines graph and KAT predictions:
```
P(pregnancy) = w_GNN × P_GNN + (1 − w_GNN) × P_KAT
```
where w_GNN ∈ [0.25, 0.40] is determined automatically during cross-validation to maximise AUC. The two models agree with r = 0.79, and their structured partial disagreement is precisely what makes the ensemble valuable.

**Clinical interpretability:** The model can display the ten training cohort patients most similar to the current patient (ranked by cosine similarity), with their documented outcomes — a direct, intuitive contextualisation not available from any scalar probability score.

| Metric | GNN alone | KAT Ensemble | GAT+KAT |
|--------|-----------|--------------|---------|
| AUC-ROC | 0.632 | 0.652 | ~0.658–0.665 |
| Brier Score | 0.241 | 0.233 | ~0.229 |
| F1 Score | 0.623 | — | — |

**Position in pipeline:** L6 operates downstream of KAT (L3) and receives the KAT probability as a node-level attribute — allowing the attention mechanism to weight it alongside clinical parameters while forming its own structural judgement.

---

## Repository Structure

```
ivf-digital-twin/
├── app.py                          # Streamlit clinical application (~2,300 lines)
├── requirements.txt                # Python dependencies
├── INSTALL.bat                     # Windows one-click installer
├── Start_IVF_Twin.bat              # Windows launcher
├── license_DEMO.lic                # Demo license file
├── logo22.png                      # Application logo
├── IVF_Digital_Twin_Overview.html  # Offline overview page
│
├── src/
│   ├── ivf_digital_twin.py         # Core L1–L4 pipeline (~138 KB)
│   ├── embryo_csdi_v3.py           # CSDI Hybrid v3 diffusion module (L5)
│   ├── embryo_tabddpm.py           # TabDDPM v3 (previous architecture, L5)
│   ├── gnn_predictor.py            # GAT Graph Attention Transformer (L6)
│   ├── pdf_report.py               # Clinical PDF report generator
│   └── crypt_engine.py             # RSA+AES-256 offline license engine
│
├── models/
│   ├── config.json                 # CSDI model hyperparameters + thresholds
│   ├── training_history.json       # Train/val loss history
│   └── CSDI_Hybrid_v3_Technical_Description.md  # Detailed model documentation
│       [neural network weights not included — see Model Availability]
│
├── data/
│   └── sample/
│       └── sample_patients.csv     # 5 reference patients for testing
│
├── dt_analytics_data/
│   └── dt_predictions.csv          # Prediction log (auto-generated at runtime)
│
├── fonts/                          # DejaVu fonts for PDF export
│
├── docs/                           # Additional documentation
├── scripts/                        # Utility scripts
│
├── IVF_Digital_Twin_v6_2.pdf       # Full technical documentation (37 pages)
└── GAT.pdf                         # GAT layer description
```

---

## Installation

### Windows (recommended)

1. Install **Python 3.11** from [python.org](https://www.python.org/downloads/) — check "Add Python to PATH"
2. Place neural network model files in `models/` (see [Model Availability](#model-availability))
3. Double-click **`INSTALL.bat`** — installs all dependencies including PyTorch and PyTorch Geometric
4. Double-click **`Start_IVF_Twin.bat`** — opens the application at `http://localhost:8501`

The installer performs 9 steps:
1. Python version check
2. Virtual environment creation (`.venv/`)
3. pip upgrade
4. Core packages (numpy, scipy, pandas, plotly, streamlit, matplotlib, cryptography, reportlab, kaleido)
5. PyTorch 2.5.1+cpu (~200 MB)
6. ML packages (lightgbm, scikit-learn==1.5.0, joblib)
7. NN extras (pykan, mambular, crepes, cloudpickle)
8. PyTorch Geometric (torch-scatter, torch-sparse, torch-cluster, torch-geometric via pyg.org wheels)
9. Launcher script

### Manual installation

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
pip install torch-geometric

streamlit run app.py
```

### Key dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.32.0 | Web application UI |
| torch | 2.5.1+cpu | CSDI diffusion model, KAT neural network |
| lightgbm | ≥4.0.0 | CSDI pregnancy classifier (L5) |
| torch-geometric | latest | GAT graph neural network (L6) |
| scikit-learn | 1.5.0 | QuantileTransformer, Platt scaling |
| pykan | ≥0.2.8 | Kolmogorov-Arnold Network (L3) |
| mambular | 0.2.2 | FT-Transformer component (L3) |
| crepes | 0.8.0 | Conformal prediction (L5) |
| reportlab | ≥4.0.0 | Clinical PDF report generation |
| cryptography | ≥41.0.0 | RSA+AES-256 license engine |

> **Note:** PyTorch Geometric wheels must match the exact PyTorch version (2.5.1+cpu). INSTALL.bat handles this automatically.

---

## Usage

After launching, the Streamlit application presents a sidebar for patient input and the following tabs:

| Tab | Content |
|-----|---------|
| 📊 Overview | Summary dashboard with key predictions |
| 🎯 Monte Carlo | Full L1 stochastic pipeline distributions |
| 🧠 Neural Network | KAT ensemble + Bayesian posterior (L3) |
| 🔬 Clusters | Phenotype classification visualization (L4) |
| 🧬 Diffusion | CSDI Hybrid v3 outputs + conformal intervals (L5) |
| 🕸️ Graph | GAT patient-similarity graph + neighbour display (L6) |
| 📋 Report | PDF clinical report export |

### Patient inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `female_age` | years | Patient age at cycle start |
| `amh` | ng/mL | Anti-Müllerian hormone |
| `afc` | integer | Antral follicle count |
| `bmi` | kg/m² | Body mass index |
| `attempt_number` | integer | IVF attempt number (for decay curve) |

Mid-cycle updates (optional, entered as they become available):
`follicles_tvp`, `known_okk`, `known_mii`, `known_pn2`, `known_blasts`, `known_good`

### CSDI module (programmatic)

```python
from src.embryo_csdi_v3 import EmbryoHybridV3

model = EmbryoHybridV3.load('models/embryo_v3_model')

patient = {
    "Количество фолликулов":  12.0,
    "Число ОКК":               9.0,
    "Число инсеминированных":  8.0,
    "2 pN":                    6.0,
    "Частота получения ОКК":   0.75,
    "Частота оплодотворения":  0.75,
    "KPIScore":               18.0,
}

result = model.mc_sample(patient, n_samples=2000)

print(result['P_pregnancy'])        # calibrated P(pregnancy)
print(result['CI_95'])              # (lo, hi) Wilson 95% CI
print(result['blast_total_median']) # median blastocyst count
print(result['PI_90_counts'])       # {'Число Bl': (lo, hi), ...}
print(result['samples'])            # pd.DataFrame, 2000 × 4
```

### Sample data

`data/sample/sample_patients.csv` contains five reference patients:

| Age | AMH | AFC | Phenotype |
|-----|-----|-----|-----------|
| 32 | 2.5 | 18 | Standard responder |
| 42 | 0.5 | 6 | Poor responder |
| 28 | 4.5 | 28 | High responder |
| 35 | 2.5 | 15 | Typical patient |
| 38 | 1.5 | 10 | Mid-range borderline |

---

## Licensing

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.

The application includes an offline RSA+AES-256 license engine (`src/crypt_engine.py`). A demo license (`license_DEMO.lic`) is included for evaluation. For clinic deployment licenses, contact embryossa@gmail.com.

---

## Model Availability

**Neural network weights are not included in this repository** to protect proprietary training data. The repository contains all source code, architecture definitions, configuration files, and training history metadata.

The following model files are required for full functionality but must be obtained separately:

| File | Size | Purpose |
|------|------|---------|
| `models/embryo_v3_model/csdi_weights.pt` | ~25 MB | CSDI Transformer denoiser |
| `models/embryo_v3_model/normalizer.pt` | ~1 MB | QuantileNormalizer arrays |
| `models/embryo_v3_model/lgb_state.pt` | ~5 MB | LightGBM classifier |
| `models/embryo_v3_model/platt_calibrator.pt` | ~1 KB | Platt scaling weights |
| `models/embryo_v3_model/conformal.pt` | ~1 KB | Conformal radii |
| `models/kat_ensemble/` | ~50 MB | KAN + FT-Transformer (L3) |
| `models/gnn_model/` | ~10 MB | GAT Graph Transformer (L6) |

Contact embryossa@gmail.com to request model weights for research collaboration.

The application will run in degraded mode (L1, L2, L4 functional) without the neural network weights. CSDI (L5) and GAT (L6) tabs will show a "model not loaded" message.

---

## Graceful Fallback

The architecture includes explicit fallback handling. If any neural network component fails to load (missing weights, incompatible PyTorch version, insufficient memory), the system falls back to the Monte Carlo + ensemble prediction (L1+L2), which remains fully functional. The application clearly indicates which layers are active.

---

## Clinical Interpretation Guide

> **This system is a research prototype and decision-support tool. It does not replace clinical judgment and should not be used as the sole basis for clinical decisions.**

### Pregnancy probability interpretation

| Source | Clinical role |
|--------|--------------|
| L2 ensemble (FORTUNE + KPI) | Population-calibrated per-transfer probability |
| L3 KAT neural network | Complex non-linear interaction capture |
| L3 Bayesian posterior | Re-anchored to the clinic's own historical data |
| L5 CSDI P(pregnancy) | Independent data-driven verification (threshold: 0.343) |
| L6 GAT ensemble | Patient-similarity contextualisation |

**When L1 and L5 converge** (KS p > 0.05 between MC and CSDI blastocyst distributions), this constitutes equifinality verification — two genuinely independent epistemic sources agree.

**When they diverge**, the divergence itself is diagnostically valuable for quality management.

### CSDI pregnancy probability thresholds

| P(pregnancy) | Interpretation |
|-------------|----------------|
| ≥ 0.343 | Favourable — expected laboratory outcomes support transfer |
| 0.25–0.343 | Moderate — consider additional cycles or PGT-A |
| < 0.25 | Cautious — low blastocyst yield warrants counselling |

### Conformal predictive intervals (L5)

The 90% conformal PI for blastocyst counts has guaranteed finite-sample coverage (≥90%) without distributional assumptions. Example output for a typical patient (age 35, AFC=12, KPI=18):
```
P(pregnancy) = 46.7%   95% CI: [43.6%, 49.8%]
Blastocysts total  (median):  2   PI_90: [0, 5]
Blastocysts good   (median):  1   PI_90: [0, 3]
```

### Quality management application

The platform functions as a continuous laboratory surveillance instrument. Because it generates full predicted distributions at every cycle stage, the predicted–observed difference at each stage can be charted as a Shewhart control chart with patient-adjusted centre line and Monte Carlo-derived control limits — enabling stage-resolved discrepancy attribution and early detection of process variability.

---

## Comparison with Existing Tools

| Feature | CDC IVF Estimator | Orchid Calculator | Herasight | **IVF Digital Twin v6.2** |
|---------|------------------|-------------------|-----------|--------------------------|
| Full probability distributions | ✗ | ✗ | ✓ | ✓ |
| Mid-cycle conditional updating | ✗ | ✗ | ✓ | ✓ |
| OHSS risk quantification | ✗ | ✗ | ✗ | ✓ |
| Phenotype benchmarking | ✗ | ✗ | ✗ | ✓ |
| Independent generative verification | ✗ | ✗ | ✗ | ✓ |
| Patient-similarity graph reasoning | ✗ | ✗ | ✗ | ✓ |
| Bayesian clinic-specific updating | ✗ | ✗ | ✗ | ✓ |
| Uncertainty intervals (all stages) | ✗ | Partial | ✗ | ✓ |
| PDF clinical report | ✗ | ✗ | ✗ | ✓ |

---

## References

1. Craig A et al. Stage-Structured, Distributional Prediction of IVF Outcomes with Conditional Updating. *medRxiv* 2025.09.27.25336680.
2. Carrasquillo R et al. FORTUNE: A clinically validated prediction model for IVF live birth rate. *Hum Reprod.* 2025. PMID:40889782.
3. Liu Z et al. KAN: Kolmogorov-Arnold Networks. *arXiv:2404.19756.* 2024.
4. Gorishniy Y et al. Revisiting Deep Learning Models for Tabular Data. *NeurIPS.* 2021.
5. Vovk V, Petej I. Venn-Abers predictors. *arXiv:1211.0025.* 2012.
6. Tashiro Y et al. CSDI: Conditional Score-based Diffusion for Imputation. *NeurIPS 2021.*
7. Nichol A, Dhariwal P. Improved Denoising Diffusion Probabilistic Models. *ICML 2021.*
8. Song J et al. Denoising Diffusion Implicit Models. *ICLR 2021.*
9. Ke G et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017.*
10. Angelopoulos A, Bates S. A Gentle Introduction to Conformal Prediction. *arXiv:2107.07511.* 2022.
11. Franasiak JM et al. The nature of aneuploidy with increasing age of the female partner: 15,169 biopsies. *Fertil Steril.* 2014;101(3):656–663.
12. Romanski PA et al. Age-specific blastocyst conversion rates in embryo cryopreservation cycles. *Reprod Biomed Online.* 2022;45(3):432–439.
13. Coello A et al. Prediction of embryo survival and live birth rates after cryotransfers. *Reprod Biomed Online.* 2021;42(5):881–891.
14. Sergeev S et al. Decoding IVF Laboratory Performance through Dimensionality Reduction and Cluster Analysis. *(manuscript under review).*

---

## Citation

If you use this software in research, please cite:

```bibtex
@software{sergeev2025ivf,
  author    = {Sergeev, Sergei},
  title     = {IVF Digital Twin v6.2: An Integrated Multi-Source Ensemble 
               Platform for Stage-Stratified IVF Outcome Prediction},
  year      = {2025},
  url       = {https://github.com/embryossa/IVF-Digital-Twin},
  note      = {Research prototype}
}
```

---

## Contact

**Sergei Sergeev** · embryossa@gmail.com

For research collaboration, model weight access, clinic deployment licensing, or external validation partnerships.
