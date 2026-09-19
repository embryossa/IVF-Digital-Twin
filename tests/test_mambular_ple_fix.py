# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""FT-Transformer encoding must not depend on the other rows of the batch."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ple_encoding = pytest.importorskip("mambular.preprocessing.ple_encoding")

import mambular_ple_fix  # noqa: E402


def test_single_row_equals_row_in_batch():
    assert mambular_ple_fix.apply()
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 20, size=(400, 1))
    y = (x[:, 0] > 8).astype(float) + rng.normal(0, 0.1, 400)
    ple = ple_encoding.PLE(n_bins=8, task="regression").fit(x, y)
    # A batch whose maximum lies below the last fitted threshold: the upstream
    # code replaced that threshold with the batch maximum.
    batch = np.array([[1.0], [3.0], [5.0]])
    together = ple.transform(batch)
    for i in range(len(batch)):
        np.testing.assert_allclose(ple.transform(batch[i:i + 1]), together[i:i + 1], atol=1e-7)


def test_apply_is_idempotent():
    assert mambular_ple_fix.apply()
    assert mambular_ple_fix.apply()
