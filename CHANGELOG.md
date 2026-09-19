# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [7.1.0] — 2026-09-20

Computation logic of the 7.1 clinic build. The Streamlit interface keeps its
7.0 layout; it now reads every number from the 7.1 data flow. **Predictions
change** — see below.

### Changed — model layers
- **L1 stage 1** refitted: negative binomial with a log link on
  ln(AMH), ln(AFC) and age (θ = 5.77); the ART-ONE linear predictor
  under-estimated yield. No separate structural-zero component (not
  identifiable in the data).
- **L1 stages 4–5** refitted: zero-inflated beta-binomial blastocyst yield and
  beta-binomial quality, conditional on the count at the previous stage.
- Entered laboratory counts now condition the whole chain (forward filtering /
  backward sampling) instead of overwriting one stage.
- **Transfer scenario:** without PGT-A every blastocyst is transferable (good
  quality first, fair quality with a data-derived odds ratio); with PGT-A only
  euploid embryos. Transfers of one cycle share a random effect (frailty), so
  cumulative probabilities no longer assume independent transfers.
- **KAT:** isotonic calibration steps are interpolated (no plateaus or jumps);
  model inputs follow the training definitions (follicles at puncture,
  day-5 embryos, frozen embryos). FT-Transformer encoding keeps the fitted
  PLE thresholds (`src/mambular_ple_fix.py`, as in the clinic build's
  mambular): with the pip package the output of one patient depended on the
  other Monte Carlo rows, and KAT differed from the clinic build by 6–7 pp.
- **CSDI (L5):** runs on the transfer profile with the training KPIScore,
  enters fusion only inside its training domain (fail-closed without
  `models/csdi_ood_stats.npz`); the fabricated Wilson interval is replaced by
  Monte Carlo prediction quantiles.
- **GAT (L6):** feature contract checked against training; exact, cached k-NN
  graph.
- **BEFE (L7):** fuses the view restricted to scenarios with a transfer; the
  range shown is now the model uncertainty (scenario spread + random-effects
  model term), which narrows as observations arrive — not the clinic
  historical corridor. OOD statistics schema 2/3 with fitted thresholds;
  collapsed OOD dimensions are rejected.
- **TRP:** outcomes integrated out (cycle 1 equals the anchor exactly); the
  anchor is the L1–L7 cycle probability; time to the first pregnancy only.
- **Esteves banking** comes from the pipeline's logistic model; transfers for
  a target account for the shared cycle effect; sperm sources are resolved
  explicitly.
- Research temperature / dynamic-τ clinic adaptation no longer enters the
  clinical calculation.

### Changed — data flow and interface wiring
- New `dt_bridge.py`: one order of computation for the interface —
  `ivf_core.predict_single_patient` (L1–L6) → `compute_l7_posterior` (L7) →
  `presentation.clinical_summary` → cycle probability → TRP anchor. The screen,
  PDF, BEFE tab and analytics row read the same result.
- Headline contract (`presentation.py`): per-transfer probability, or 0 for
  the current cycle when the entered results exclude a transfer.
- `app.py` and `scripts/batch_predict.py` import the pipeline as a module
  instead of `exec` into globals (CSDI used to overwrite core names).
- Labels: the L7 range is shown as "model uncertainty range"; banking and PDF
  tables follow the 7.1 Esteves output.

### Added
- `src/embryology.py` (shared L1 parameters and exact conditioning),
  `src/modelio.py` (one model loader), `src/local_network.py` (Ollama host
  must be loopback), `src/mambular_ple_fix.py`, `tests/test_bridge.py`,
  `tests/test_mambular_ple_fix.py`.

## [7.0.1] — 2026-07-30

Repository and licensing release. No change to model behaviour or predictions.

### Changed
- **Licence: Apache-2.0 → PolyForm Noncommercial 1.0.0.** The project is now
  source-available. Noncommercial use stays free; commercial use requires a
  separate licence. Everything up to tag `v6.2-apache` remains Apache-2.0
  forever — see [LICENSE-HISTORY.md](LICENSE-HISTORY.md).
- Code copyright corrected to the sole copyright holder. Scientific
  co-authorship is credited in [AUTHORS.md](AUTHORS.md) and
  [CITATION.cff](CITATION.cff), where it belongs.
- `app.py` runs in **Research Mode** when the licence engine is absent, instead
  of refusing to start. The public repository is now runnable end to end.

### Security
- Removed the offline licence engine from the public repository and purged it
  from git history. It embedded a hardcoded AES-256 key that would have let
  anyone decrypt distributed model weights. The key is treated as compromised
  and is being rotated. No patient data was involved — see
  [SECURITY.md](SECURITY.md).
- Real clinic outcome data replaced by `clinic_config.template.json`.
- Hardened `.gitignore`: spreadsheets, weights, keys and pipeline outputs are
  now blanket-ignored rather than path-by-path.
- Added gitleaks pre-commit hook and a CI secret scan.
- Added `scripts/export_public_repo.py` — allow-list + denylist + secret scan,
  three independent gates before anything reaches the public repo.

### Added
- `SECURITY.md` with a disclosure policy and an explicit trust boundary
  (local-only execution, model files as trusted input, no cloud LLM calls).
- `DISCLAIMER.md`, `THIRD-PARTY-NOTICES.md`, `CONTRIBUTING.md`,
  `COMMERCIAL-LICENSE.md`, `CITATION.cff`, this changelog.
- CI: lint, tests on Python 3.10/3.11, secret scan, SPDX header check.
- SPDX headers on every source file.
- DejaVu font licence, which the Bitstream Vera terms require to be shipped.

## [7.0.0] — 2026-07

### Added
- **L7 — BEFE (Bayesian Evidence Fusion Engine).** Trust-weighted logit-space
  pooling of all upstream layers into one calibrated posterior. Reports the
  fusion pull ratio, a 0–100 Reliability Index, the source of disagreement when
  experts diverge, and dual Mahalanobis OOD detection over clinical and
  embryological feature subspaces.
- **L6 — GAT patient-similarity graph** over 1,172 clinical protocols. Passes
  effective neighbour count `N_eff`, attention entropy and neighbour outcome
  variance to L7 as trust features.
- Retrieval-grounded narrative layer with faithfulness scoring
  (`guideline_rag.py`, `faithfulness.py`, `eval_retrieval.py`).

## [6.2.0] — 2026-05

### Added
- **L5 — CSDI Hybrid v3**: diffusion count generation separated from binary
  prediction, replacing TabDDPM v3.
- Clinic-specific calibration workflow (`calibrate_for_clinic.py`,
  `validate_clinic_data.py`).

### Fixed
- TabDDPM v3's +15.8 pp prevalence bias. ECE 0.158 → 0.029, AUROC 0.578 → 0.661.

[7.0.1]: https://github.com/embryossa/IVF-Digital-Twin/releases/tag/v7.0.1
[7.0.0]: https://github.com/embryossa/IVF-Digital-Twin/releases/tag/v7.0.0
[6.2.0]: https://github.com/embryossa/IVF-Digital-Twin/releases/tag/v6.2-apache
