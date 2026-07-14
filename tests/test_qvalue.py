"""Unit tests for target-decoy competition q-values."""

from __future__ import annotations

import numpy as np

from ristretto._qvalue import tdc_qvalues


def test_hand_computed():
    # scores already descending; labels T,T,D,T,D
    scores = np.array([10.0, 9.0, 8.0, 7.0, 6.0])
    is_target = np.array([True, True, False, True, False])
    q = tdc_qvalues(scores, is_target)
    # fdr = (cum_decoys+1)/cum_targets, then cumulative min from the tail
    expected = np.array([0.5, 0.5, 2 / 3, 2 / 3, 1.0])
    np.testing.assert_allclose(q, expected)


def test_order_independence():
    scores = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
    is_target = np.array([False, True, True, False, True])
    q = tdc_qvalues(scores, is_target)
    # Permuting inputs permutes outputs identically.
    perm = np.array([3, 0, 4, 1, 2])
    q_perm = tdc_qvalues(scores[perm], is_target[perm])
    np.testing.assert_allclose(q_perm, q[perm])


def test_monotone_nonincreasing_with_score():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=500)
    is_target = rng.random(500) < 0.7
    q = tdc_qvalues(scores, is_target)
    # Sorted by descending score, q must be non-decreasing (cumulative min tail).
    q_desc = q[np.argsort(-scores)]
    assert np.all(np.diff(q_desc) >= -1e-12)


def test_range_and_all_targets():
    scores = np.array([3.0, 2.0, 1.0])
    is_target = np.array([True, True, True])
    q = tdc_qvalues(scores, is_target)
    assert np.all((q >= 0) & (q <= 1))
    # No decoys: fdr = 1/cum_targets, decreasing then tail-min.
    np.testing.assert_allclose(q, [1 / 3, 1 / 3, 1 / 3])
