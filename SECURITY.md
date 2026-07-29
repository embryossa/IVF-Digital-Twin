# Security Policy

## Reporting a vulnerability

Email **embryossa@gmail.com** with `SECURITY` in the subject line. Please do
not open a public issue.

Include: what you found, how to reproduce it, and what an attacker could do
with it. You will get an acknowledgement within 5 working days and an
assessment within 15. If the finding is valid you will be credited in the
release notes unless you prefer otherwise.

There is no bug bounty. This is a research project.

## Supported versions

Only the current release on `main` receives security fixes. Versions from the
Apache-2.0 era (up to tag `v6.2-apache`) are not maintained — see
[LICENSE-HISTORY.md](LICENSE-HISTORY.md).

## Trust boundary

Understanding what this software does and does not defend against matters more
than a list of hardening flags.

### The software runs locally and offline

- The Streamlit app binds to `localhost` and is intended for single-machine or
  intranet use behind the clinic's own network controls. **Do not expose it to
  the public internet.** It has no authentication, authorization, session
  management, or audit trail suitable for a multi-tenant deployment.
- The narrative layer (`llm_consultant.py`, `guideline_rag.py`) talks to a
  **local Ollama instance** at `127.0.0.1:11434`. No patient data is sent to
  any cloud LLM provider, and there are no third-party API keys anywhere in
  this codebase. If you repoint `OLLAMA_HOST` at a remote or hosted endpoint,
  that guarantee is yours to re-establish.

### Model files are trusted input

`torch.load`, `joblib.load` and `pickle.load` execute arbitrary code contained
in the file they read. This is inherent to those formats, not a defect in this
code. The consequence:

> **Load model weights only from a source you trust.** Weights obtained from
> the author over a verified channel are fine. Weights from a random download
> are equivalent to running an unknown executable.

`src/ivf_digital_twin.py` uses `weights_only=True` where the artifact is a
plain state dict and falls back only when the artifact legitimately contains
non-tensor state (LightGBM boosters, sklearn calibrators). The remaining
`weights_only=False` call sites are documented in code.

Verify artifacts before loading:

```bash
sha256sum -c models/CHECKSUMS.sha256
```

### Patient data never leaves the machine

The pipeline writes prediction logs to `dt_analytics_data/` and cached LLM
responses to `dt_llm_cache/`. Both may contain patient-derived values. Both are
in `.gitignore` and neither is published. Under GDPR/HIPAA these are your
processing records — apply your own retention and access controls to them.

`data/sample/sample_patients.csv` is synthetic. No real patient record is
included anywhere in this repository.

### What is not in this repository

By design, and enforced by `scripts/export_public_repo.py`:

- the offline licence engine and its keys
- trained neural network weights
- any clinic's real outcome data
- any patient-level record

## Secret hygiene

Every export runs an allow-list, a denylist and a regex secret scan before a
single file is copied. `.pre-commit-config.yaml` runs
[gitleaks](https://github.com/gitleaks/gitleaks) on every commit. If you fork
this repo and add credentials, those gates protect you too — do not remove
them.

## Known history event

Prior to 2026-07-30 an earlier revision of this repository contained a
hardcoded symmetric key inside the licence engine. The file has been removed
from the repository and purged from git history, and the key is treated as
compromised: it is being rotated and the artifacts it protected re-encrypted.
The old key protected only distributed model weights — **no patient data was
ever exposed**, and no patient data has ever been present in this repository.

Recorded here rather than quietly dropped, because a security policy that
hides its own incidents is not worth reading.
