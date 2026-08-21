"""Target-decoy competition q-values."""

from __future__ import annotations

import numpy as np


def tdc_qvalues(scores: np.ndarray, is_target: np.ndarray) -> np.ndarray:
    """
    Target-decoy competition q-values. Higher score = better.

    Notes
    -----
    PSMs with tied scores are treated as one block: each gets the FDR computed
    from that block's full, fully-accumulated target/decoy counts (not the
    partial counts implied by their arbitrary position within the tie), so
    all tied PSMs share one q-value. Without this, PSMs early in a large tie
    (e.g. a low-cardinality feature) get an artificially optimistic q-value
    based on only part of the tied decoys/targets having been counted yet.

    """
    scores = np.asarray(scores, dtype=np.float64)
    is_target = np.asarray(is_target, dtype=bool)

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_targets = is_target[order]

    cum_targets = np.cumsum(sorted_targets)
    cum_decoys = np.cumsum(~sorted_targets)

    fdr = (cum_decoys + 1) / np.maximum(cum_targets, 1)

    # Collapse ties: every PSM in a tied-score block gets the FDR from that
    # block's last (fully-accumulated) position, not its own partial count.
    is_new_group = np.empty(len(sorted_scores), dtype=bool)
    is_new_group[0] = True
    is_new_group[1:] = sorted_scores[1:] != sorted_scores[:-1]
    group_id = np.cumsum(is_new_group) - 1
    group_sizes = np.bincount(group_id)
    group_end_idx = np.cumsum(group_sizes) - 1
    group_fdr = fdr[group_end_idx][group_id]

    q_sorted = np.minimum.accumulate(group_fdr[::-1])[::-1]

    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q
