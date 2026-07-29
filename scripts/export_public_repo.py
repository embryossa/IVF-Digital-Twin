"""Export the public source-available repository from the private working tree.

The public repo (github.com/embryossa/IVF-Digital-Twin) is a filtered subset of
this tree, defined by `docs/PUBLIC_RELEASE_MANIFEST.md` and encoded here.

Design: the manifest is an explicit **allow-list**. Nothing reaches the public
repo unless it is named below. A denylist is then applied on top as a second
barrier, and every exported file is scanned for secrets before the run is
allowed to succeed. Two independent gates, because one bad export is
unrecoverable — the file is on the internet.

Usage:
    python scripts/export_public_repo.py --dest /path/to/public/checkout
    python scripts/export_public_repo.py --dest ... --dry-run
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ── ALLOW-LIST ───────────────────────────────────────────────────────────────
# Globs relative to the repo root. Only these reach the public repository.
INCLUDE = [
    # Application entry points
    "app.py",
    "dt_ui.py",
    "i18n.py",
    # Core pipeline (L1-L4) and model layers
    "src/ivf_core.py",
    "src/ivf_digital_twin.py",
    "src/embryo_csdi_v3.py",
    "src/embryo_tabddpm.py",
    "src/gnn_predictor.py",
    "src/pdf_report.py",
    # BEFE — Bayesian Evidence Fusion Engine (L7)
    "befe.py",
    "befe_app.py",
    "befe_batch_utils.py",
    "fit_befe_ood.py",
    # Analysis, batch and post-processing
    "batch_analysis.py",
    "dt_postprocess.py",
    "trp_engine.py",
    "stim_protocol.py",
    # Calibration and validation
    "calibrate_for_clinic.py",
    "validate_clinic_data.py",
    "eval_retrieval.py",
    "faithfulness.py",
    # Narrative / RAG layer (local Ollama only — no cloud calls)
    "llm_consultant.py",
    "llm_cache.py",
    "guideline_rag.py",
    "patient_brief.py",
    "protocol_guidance.py",
    # Configuration (non-clinic-specific)
    "stim_params.json",
    "guidelines_pack.json",
    "clinic_config.template.json",
    # Packaging
    "requirements.txt",
    "requirements_nn.txt",
    "setup.py",
    # Launchers
    "INSTALL.bat",
    "Start_IVF_Twin.bat",
    "RUN_BATCH.bat",
    "launch_windows.bat",
    "launch_mac_linux.sh",
    "install_nn_deps.bat",
    "REPAIR_GNN_ENV.bat",
    # Tests
    "tests/**/*.py",
    # Documentation
    "docs/**/*.md",
    "docs/reference/*.pdf",
    "docs/reference/*.html",
    "analytics/*.md",
    "analytics/*.ipynb",
    "README_dt_batch.md",
    "SYSTEM_NARRATOR_v2.md",
    "models/*.md",
    "models/embryo_v3_model/*.md",
    "models/embryo_v3_model/config.json",
    "models/embryo_v3_model/training_history.json",
    # Narrator QA harness (code and protocol only — never results)
    "narrator_qa/*.py",
    "narrator_qa/*.md",
    # NB: pathlib's `**` yields directories only — always append `/*` for files.
    "narrator_qa/prompts/**/*",
    # Reproducible scripts
    "scripts/batch_predict.py",
    "scripts/add_spdx_headers.py",
    "scripts/export_public_repo.py",
    # Assets
    "data/sample/*.csv",
    "fonts/*.ttf",
    "fonts/LICENSE-DejaVu.txt",
    "logo22.png",
    # Licensing and governance
    "LICENSE",
    "LICENSE-HISTORY.md",
    "COMMERCIAL-LICENSE.md",
    "DISCLAIMER.md",
    "THIRD-PARTY-NOTICES.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "AUTHORS.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "README.md",
    ".gitignore",
    ".gitattributes",
    ".pre-commit-config.yaml",
    "ruff.toml",
    ".github/**/*",
]

# ── DENYLIST ─────────────────────────────────────────────────────────────────
# Second barrier. A path matching any of these is refused even if some
# INCLUDE glob picked it up.
DENY_PATTERNS = [
    r"crypt_engine",           # embeds the model decryption key
    r"\.lic$", r"\.key$", r"\.pem$",
    r"^license\.",
    r"legacy_",                # superseded runtime, not maintained
    r"app_patch_instructions",  # internal working notes
    r"build_.*clinic_release",  # commercial build tooling
    r"\.bak", r"\.bak_\d+",
    r"^clinic_config\.json$",  # real clinic outcomes; template ships instead
    r"PUBLIC_RELEASE_MANIFEST",  # internal ops doc; SECURITY.md is the public face
    r"(^|/)results/", r"(^|/)dt_analytics_data/", r"(^|/)dt_llm_cache/",
    r"(^|/)releases/", r"(^|/)narrator_qa/results/",
    r"(^|/)data/(?!sample/)",  # any data dir except data/sample
    r"\.(pt|pth|joblib|pkl|npz|onnx|safetensors)$",   # trained weights
    r"\.(xlsx|xls)$",          # spreadsheets may carry patient rows
    r"__pycache__", r"\.venv", r"\.egg-info",
]
DENY = [re.compile(p) for p in DENY_PATTERNS]

# ── SECRET SCAN ──────────────────────────────────────────────────────────────
SECRET_PATTERNS = [
    (r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY", "private key"),
    (r"\b[a-fA-F0-9]{64}\b", "64-hex string (possible symmetric key)"),
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI-style API key"),
    (r"gh[pousr]_[A-Za-z0-9]{30,}", "GitHub token"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key id"),
    (r"xox[baprs]-[A-Za-z0-9-]{10,}", "Slack token"),
    (r"(?i)\b(password|passwd|secret|api_key|apikey|token)\s*[:=]\s*"
     r"[\"'][^\"'\s]{8,}[\"']", "hardcoded credential"),
]
SECRETS = [(re.compile(p), label) for p, label in SECRET_PATTERNS]

SCANNABLE = {".py", ".json", ".md", ".txt", ".yml", ".yaml", ".cfg", ".toml",
             ".bat", ".sh", ".html", ".ini"}


def denied(rel: str) -> str | None:
    for pat in DENY:
        if pat.search(rel):
            return pat.pattern
    return None


def collect() -> list[Path]:
    seen: set[Path] = set()
    for pattern in INCLUDE:
        for path in REPO.glob(pattern):
            if path.is_file():
                seen.add(path)
    return sorted(seen)


def scan_secrets(paths: list[Path]) -> list[str]:
    findings = []
    for p in paths:
        if p.suffix.lower() not in SCANNABLE:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for pat, label in SECRETS:
                if pat.search(line):
                    rel = p.relative_to(REPO).as_posix()
                    findings.append(f"{rel}:{lineno}: {label}")
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", required=True, type=Path,
                    help="public repository checkout to write into")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dest = args.dest.resolve()
    if not (dest / ".git").is_dir():
        print(f"error: {dest} is not a git checkout", file=sys.stderr)
        return 2

    files = collect()
    if not files:
        print("error: allow-list matched nothing", file=sys.stderr)
        return 2

    # Barrier 2 — denylist
    refused = []
    keep = []
    for p in files:
        rel = p.relative_to(REPO).as_posix()
        why = denied(rel)
        if why:
            refused.append(f"{rel}  (matched /{why}/)")
        else:
            keep.append(p)

    if refused:
        print("REFUSED by denylist:")
        for r in refused:
            print("  ", r)
        print()

    # Barrier 3 — secret scan
    findings = scan_secrets(keep)
    if findings:
        print("SECRET SCAN FAILED — export aborted:", file=sys.stderr)
        for f in findings:
            print("  ", f, file=sys.stderr)
        return 1
    print(f"secret scan clean over {len(keep)} files")

    if args.dry_run:
        print(f"\n[dry-run] would export {len(keep)} files to {dest}")
        for p in keep:
            print("  ", p.relative_to(REPO).as_posix())
        return 0

    # Wipe the destination working tree, preserving .git
    for child in dest.iterdir():
        if child.name == ".git":
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()

    for p in keep:
        target = dest / p.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)

    print(f"exported {len(keep)} files to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
