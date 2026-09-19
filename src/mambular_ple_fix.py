# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""FT-Transformer feature encoding fix of the 7.1 clinic build (2026-09-12).

mambular 0.2.2 `PLE.transform` replaces the last fitted tree threshold with
the maximum of the batch being encoded. The FT-Transformer output of one row
then depends on the other rows of the batch (in the Digital Twin, the Monte
Carlo scenarios of the same patient), so single-patient inference is not
reproducible. The 7.1 clinic build ships mambular with those two lines
removed; this module applies the same change to the pip package, so the
repository gives the same KAT numbers as the clinic build.

apply() is idempotent and a no-op when mambular is not installed.
"""
import re

import numpy as np

_APPLIED = False


def _transform(self, feature):
    """mambular 0.2.2 PLE.transform, keeping the fitted thresholds."""
    if feature.shape == (feature.shape[0], 1):
        feature = np.squeeze(feature, axis=1)
    result_list = []
    for idx, cond in enumerate(self.conditions):
        # `feature` is the name used inside the fitted tree conditions.
        result_list.append(eval(cond) * (idx + 1))

    encoded_feature = np.expand_dims(np.sum(np.stack(result_list).T, axis=1), 1)
    encoded_feature = np.array(encoded_feature - 1, dtype=np.int64)

    locations = []
    for string in self.conditions:
        locations.extend(re.findall(self.pattern, string))
    locations = np.sort(list({float(number) for number in locations}))

    ple_encoded_feature = np.zeros((len(feature), locations.shape[0] + 1))
    # The upstream code replaced locations[-1] with max(feature) here.

    for idx in range(len(encoded_feature)):
        if feature[idx] >= locations[-1]:
            ple_encoded_feature[idx][encoded_feature[idx]] = feature[idx]
            ple_encoded_feature[idx, : encoded_feature[idx][0]] = 1
        elif feature[idx] <= locations[0]:
            ple_encoded_feature[idx][encoded_feature[idx]] = feature[idx]
        else:
            ple_encoded_feature[idx][encoded_feature[idx]] = (
                feature[idx] - locations[(encoded_feature[idx] - 1)[0]]
            ) / (
                locations[(encoded_feature[idx])[0]]
                - locations[(encoded_feature[idx] - 1)[0]]
            )
            ple_encoded_feature[idx, : encoded_feature[idx][0]] = 1

    if ple_encoded_feature.shape[1] == 1:
        return np.zeros([len(feature), self.n_bins])
    return np.array(ple_encoded_feature, dtype=np.float32)


def apply():
    """Patch mambular's PLE encoder in this process. Returns True if active."""
    global _APPLIED
    if _APPLIED:
        return True
    try:
        from mambular.preprocessing import ple_encoding
    except Exception:
        return False
    ple_encoding.PLE.transform = _transform
    _APPLIED = True
    return True
