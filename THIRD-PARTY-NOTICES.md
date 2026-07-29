# Third-Party Notices

IVF Digital Twin is licensed under the PolyForm Noncommercial License 1.0.0
(see `LICENSE`). That license applies to this project's own source code only.

The third-party components listed below are **not** covered by that license.
Each remains governed by its own terms, and those terms are unaffected by the
licensing of this project. All listed dependencies are under permissive
licenses (MIT, BSD, Apache-2.0, PSF or equivalent) that impose no copyleft
obligation on this project and place no restriction on offering this project
commercially.

## Python dependencies

Declared in `requirements.txt` and `requirements_nn.txt`. Not vendored — they
are installed by pip at install time from PyPI.

| Component | License |
|---|---|
| numpy | BSD-3-Clause |
| scipy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| joblib | BSD-3-Clause |
| cloudpickle | BSD-3-Clause |
| matplotlib | PSF-based (matplotlib license) |
| plotly | MIT |
| kaleido | MIT |
| streamlit | Apache-2.0 |
| lightgbm | MIT |
| cryptography | Apache-2.0 OR BSD-3-Clause |
| reportlab | BSD-3-Clause |
| pdfkit | MIT |
| torch | BSD-3-Clause |
| torch-geometric, torch-scatter, torch-sparse, torch-cluster, torch-spline-conv | MIT |
| mambular | MIT |
| crepes | BSD-3-Clause |
| model-unpickler | see note below |

**Note on `pdfkit`:** the Python wrapper is MIT, but it drives the external
`wkhtmltopdf` binary, which is LGPL-3.0. `wkhtmltopdf` is invoked as a
separate process and is not distributed with this project, so no LGPL
obligation attaches to this codebase. Users must install it themselves.

**Note on `model-unpickler`:** license not verified at the time of writing.
Confirm its terms before a commercial release, or remove the dependency.

**Note on `mambular==0.2.2` and `crepes==0.8.0`:** pinned versions. If those
pins are changed, re-check the licenses at the new versions.

## Bundled fonts

| Component | License | Location |
|---|---|---|
| DejaVu Sans (Regular, Bold, Oblique) | Bitstream Vera Fonts License / DejaVu changes public domain | `fonts/`, license text at `fonts/LICENSE-DejaVu.txt` |

The Bitstream Vera license requires that its copyright notice be included with
any distribution of the fonts. `fonts/LICENSE-DejaVu.txt` satisfies this.

## Model weights and clinical coefficients

Trained model weights distributed with commercial licenses are the licensor's
own work product, derived from clinical datasets used with permission. They
are not third-party components and are not distributed in the public
repository.

Published coefficients incorporated into the models are cited in the project
documentation. Facts and numerical coefficients drawn from published
literature are not themselves copyrightable; the citations exist for
scientific attribution, not license compliance.

## Maintaining this file

Re-generate the dependency table whenever `requirements.txt` or
`requirements_nn.txt` changes:

```bash
pip install pip-licenses
pip-licenses --format=markdown --with-urls --order=license
```
