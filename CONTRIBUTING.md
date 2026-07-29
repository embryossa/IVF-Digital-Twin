# Contributing

Thank you for your interest in IVF Digital Twin. Please read this before
opening a pull request — the licensing model imposes a constraint that most
projects do not have.

## The constraint

This project is source-available under the
[PolyForm Noncommercial License 1.0.0](LICENSE), and the author sells
commercial licenses to the same codebase (see `COMMERCIAL-LICENSE.md`).

Selling a commercial license to code requires holding rights to all of it.
If someone contributes code and retains copyright in it, the author cannot
license that code commercially without their permission — which would make
the contribution unlicensable in practice and would have to be reverted later.

That means **code contributions can only be accepted with a signed
contributor agreement.**

## What is always welcome, with no agreement needed

- **Bug reports.** Especially reproducible ones with input data shape,
  Python version, and traceback.
- **Methodological critique.** If a coefficient, a model assumption, or a
  calibration choice looks wrong, open an issue with the reasoning and the
  reference. This is the most valuable kind of feedback for a clinical
  prediction tool.
- **Validation results.** If you ran the models against your own cohort and
  the calibration differs from what is published here, please say so.
- **Documentation corrections** typos, unclear instructions, broken links.
- **Questions** about the methods, in Issues or Discussions.

## Code contributions

1. **Open an issue first.** Describe the change before writing it. This
   avoids work that cannot be merged.
2. **If the maintainer agrees the change should go in**, you will be asked to
   sign a Contributor License Agreement. The CLA grants the maintainer the
   right to license your contribution under both the noncommercial license
   and commercial licenses. You keep your own copyright and may reuse your
   contribution elsewhere. This is the standard arrangement used by
   commercially-licensed open projects.
3. Only after the CLA is on file will the pull request be merged.

Pull requests opened without a prior issue and a CLA cannot be merged, no
matter how good the code is. This is not a judgment on the contribution.

## Code standards

- Python 3.9+ compatible.
- Every new source file carries the SPDX header:
  `# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0`
- Do not add dependencies under copyleft licenses (GPL, LGPL for linked
  libraries, AGPL). They are incompatible with the commercial licensing model.
  If you need functionality only a copyleft library provides, raise it in an
  issue.
- Changes touching prediction logic must state what they do to calibration.
  A change that improves discrimination while degrading calibration is a
  regression for this project's purpose.

## Security issues

Do not open a public issue. Email **embryossa@gmail.com** directly. This
applies particularly to anything touching `src/crypt_engine.py` or the
license key mechanism.
