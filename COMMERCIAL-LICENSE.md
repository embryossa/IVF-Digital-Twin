# Commercial License

The source in this repository is published under the
[PolyForm Noncommercial License 1.0.0](LICENSE). That license covers
noncommercial use only. This document explains when you need a commercial
license and how to obtain one.

## You do NOT need a commercial license if

- You are reading, studying, auditing or reviewing the code.
- You are reproducing published results or benchmarking the methods.
- You are a university, public research institute, public hospital,
  government body, or registered charity, using it for research,
  teaching or non-fee-generating clinical research.
- You are an individual using it for personal study or a hobby project.
- You are citing the work in a paper or thesis.

In PolyForm's wording: *any noncommercial purpose is a permitted purpose*,
and use by educational, public research, public health and government
institutions is permitted regardless of how that institution is funded.

## You DO need a commercial license if

- A private fertility clinic, laboratory or medical group uses the software
  as part of delivering paid services to patients.
- You integrate it, in whole or in part, into a product or service that you
  sell, license, or offer on a subscription or per-use basis.
- You offer it as a hosted or managed service.
- You use it in the course of paid consulting or contract work for a client.
- You use its outputs to generate reports that your organization charges for.

If you are unsure which side of the line you fall on, ask. A short email
resolves it faster than a legal review.

## What a commercial license includes

- Rights to deploy the software in commercial clinical operations.
- A license key for the protected components (see `src/crypt_engine.py`
  and the licensing model in `README.md`).
- Access to the trained model weights not distributed in the public
  repository.
- Version updates and validated model refreshes for the license term.
- Support for clinic-specific calibration (`calibrate_for_clinic.py`).
- Written terms suitable for a clinic's compliance and procurement review.

Pricing depends on clinic size, number of sites, and whether clinic-specific
recalibration is included. Academic-adjacent commercial pilots are negotiable.

## How to request one

Email **embryossa@gmail.com** with:

1. Organization name, country, and website.
2. Intended use — clinical decision support, internal analytics, product
   integration, or other.
3. Approximate annual cycle volume, and number of sites.
4. Whether you need clinic-specific calibration on your own historical data.
5. Whether your jurisdiction requires the software to carry a medical-device
   registration for your intended use (see `DISCLAIMER.md` — it currently
   does not carry one).

Please put "IVF Digital Twin — commercial license" in the subject line.

## Contributions and this license

Because the licensor sells commercial licenses to this codebase, any accepted
external contribution must come with rights broad enough to permit that.
See `CONTRIBUTING.md` before opening a pull request.
