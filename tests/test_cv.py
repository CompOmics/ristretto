"""Unit tests for spectrum-grouped k-fold cross-validation."""

from __future__ import annotations

import numpy as np

from ristretto._cv import spectrum_grouped_kfold


def _groups(n_groups, per_group, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per_group)
    rng.shuffle(g)
    return g


def test_groups_stay_together():
    groups = _groups(30, 4)
    for train_idx, test_idx in spectrum_grouped_kfold(groups, 3, seed=1):
        train_g = set(groups[train_idx].tolist())
        test_g = set(groups[test_idx].tolist())
        assert train_g.isdisjoint(test_g)


def test_full_coverage_and_disjoint_tests():
    groups = _groups(30, 4)
    n = len(groups)
    splits = spectrum_grouped_kfold(groups, 3, seed=1)
    all_test = np.concatenate([test for _, test in splits])
    assert np.array_equal(np.sort(all_test), np.arange(n))  # partition of all rows
    for train_idx, test_idx in splits:
        assert len(train_idx) + len(test_idx) == n
        assert set(train_idx).isdisjoint(set(test_idx))


def test_determinism():
    groups = _groups(20, 3)
    a = spectrum_grouped_kfold(groups, 4, seed=7)
    b = spectrum_grouped_kfold(groups, 4, seed=7)
    for (ta, sa), (tb, sb) in zip(a, b, strict=True):
        assert np.array_equal(ta, tb) and np.array_equal(sa, sb)


def test_balanced_group_counts():
    groups = _groups(30, 5)
    splits = spectrum_grouped_kfold(groups, 3, seed=2)
    # 30 groups over 3 folds -> 10 groups each.
    for _, test_idx in splits:
        assert groups[test_idx].astype(str).size > 0
        assert len(np.unique(groups[test_idx])) == 10
