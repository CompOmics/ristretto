"""Target-decoy competition q-values."""

from __future__ import annotations

import numpy as np


def tdc_qvalues(scores: np.ndarray, is_target: np.ndarray) -> np.ndarray:
    """Target-decoy competition q-values. Higher score = better."""
    scores = np.asarray(scores, dtype=np.float64)
    is_target = np.asarray(is_target, dtype=bool)

    order = np.argsort(-scores, kind="mergesort")
    sorted_targets = is_target[order]

    cum_targets = np.cumsum(sorted_targets)
    cum_decoys = np.cumsum(~sorted_targets)

    fdr = (cum_decoys + 1) / np.maximum(cum_targets, 1)
    q_sorted = np.minimum.accumulate(fdr[::-1])[::-1]

    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q
