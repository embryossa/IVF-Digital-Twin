# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""Add SPDX + copyright headers to every project .py file, idempotently.

Usage:  python add_spdx.py [--check]
  --check   report what would change, write nothing
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPDX = "# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0"
HEADER = (
    "# Copyright 2025-2026 Sergei Sergeev\n"
    f"{SPDX}\n"
    "# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md\n"
)

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "env", "site-packages",
             "build", "dist", ".eggs", "node_modules"}


def targets(root: Path):
    for p in root.rglob("*.py"):
        if SKIP_DIRS.isdisjoint(p.parts):
            yield p


def insert(text: str) -> str | None:
    """Return new text, or None if the header is already present."""
    if "SPDX-License-Identifier" in text.split("\n\n", 1)[0]:
        return None
    lines = text.splitlines(keepends=True)
    i = 0
    # keep shebang and PEP 263 encoding line at the very top
    if lines and lines[0].startswith("#!"):
        i = 1
    if i < len(lines) and "coding" in lines[i] and lines[i].startswith("#"):
        i += 1
    return "".join(lines[:i]) + HEADER + "".join(lines[i:])


def main() -> int:
    check = "--check" in sys.argv
    # CI uses --fail-on-missing so a file added without a header breaks the
    # build rather than silently shipping unlicensed.
    strict = "--fail-on-missing" in sys.argv
    if strict:
        check = True
    changed = skipped = 0
    for path in sorted(targets(ROOT)):
        text = path.read_text(encoding="utf-8")
        new = insert(text)
        if new is None:
            skipped += 1
            continue
        changed += 1
        print(f"{'would patch' if check else 'patched'}: {path.relative_to(ROOT)}")
        if not check:
            path.write_text(new, encoding="utf-8")
    print(f"\n{changed} file(s) {'to patch' if check else 'patched'}, "
          f"{skipped} already had a header.")
    if strict and changed:
        print(f"error: {changed} file(s) missing an SPDX header; "
              f"run `python scripts/add_spdx_headers.py`", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
