# License History

This project changed its license on **2026-07-30**. This file records exactly
which versions are governed by which terms, so there is no ambiguity for
anyone who obtained a copy before the change.

## Timeline

| Period | Commits | License | Status |
|---|---|---|---|
| 2026-05-15 → 2026-07-29 | up to and including `f5d21c007c9debd1f2362c27f90ad261e60cfd33` | Apache License 2.0 | Frozen, still valid for those copies |
| 2026-07-30 → present | everything after `f5d21c00` | PolyForm Noncommercial 1.0.0 | Current |

The last Apache-licensed state is tagged **`v6.2-apache`**.

## What this means in practice

**If you obtained the software before 2026-07-30**, your Apache 2.0 rights to
*that copy* are perpetual and irrevocable. The licensor cannot and does not
attempt to revoke them. You may continue to use, modify and redistribute that
snapshot under Apache 2.0, including commercially.

**If you obtain the software on or after 2026-07-30**, or you take any update
released after that date, the PolyForm Noncommercial 1.0.0 terms in `LICENSE`
apply. Noncommercial use is free; commercial use requires a separate license
(see `COMMERCIAL-LICENSE.md`).

**You cannot mix the two.** Pulling a post-change commit into an Apache-era
fork brings that commit under PolyForm Noncommercial. If you want to stay on
Apache 2.0, stay on the `v6.2-apache` tag.

## Why the change

The project is developed as a clinical decision-support research platform and
is funded by licensing it to fertility clinics. Apache 2.0 permitted any third
party to commercialize the work without contributing to its continued
development or validation. The source remains public so that the methods,
coefficients and model architecture can be inspected, reproduced and cited —
which is what a clinical prediction tool should allow — while commercial
deployment stays under the author's control.

## Authority to relicense

All source code in this repository is the sole copyright of Sergei Sergeev.
Earlier `LICENSE` headers in some working copies listed additional names; those
individuals are co-authors of the associated **publications and clinical
methodology**, not copyright holders of the source code. They are credited in
`AUTHORS.md` and `CITATION.cff`. No third-party code contributions have been
merged into this repository (all commits to date are authored by the copyright
holder), so no external consent was required for this change.
