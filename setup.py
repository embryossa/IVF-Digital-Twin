# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name="ivf_digital_twin",
    version="6.2.0",
    author="Sergei Sergeev",
    author_email="embryossa@gmail.com",
    description="An Integrated Multi-Source Ensemble Platform for Stage-Stratified IVF Outcome Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/embryossa/IVF-Digital-Twin",
    license="LicenseRef-PolyForm-Noncommercial-1.0.0",
    license_files=("LICENSE", "THIRD-PARTY-NOTICES.md"),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "nn": [
            "torch==2.4.1",
            "joblib==2.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: Other/Proprietary License",
        "Private :: Do Not Upload",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="IVF, ART, reproductive medicine, clinical prediction, Bayesian, Monte Carlo, digital twin",
    entry_points={
        "console_scripts": [
            "ivf-predict=ivf_digital_twin:main",
        ],
    },
)
