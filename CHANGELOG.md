# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
