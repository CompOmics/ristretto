"""Unit tests for histogram-based PEP estimation."""

from __future__ import annotations

import numpy as np

from ristretto._pep import nonparametric_pep


def _scores(seed=0):
    rng = np.random.default_rng(seed)
    target = rng.normal(3.0, 1.0, 300)
    decoy = rng.normal(0.0, 1.0, 100)
    scores = np.concatenate([target, decoy])
    is_target = np.concatenate([np.ones(300, bool), np.zeros(100, bool)])
    return scores, is_target


def test_range():
    scores, is_target = _scores()
    pep = nonparametric_pep(scores, is_target)
    assert pep.shape == scores.shape
    assert np.all(np.isfinite(pep))
    assert np.all((pep >= 0) & (pep <= 1))


def test_monotone_nonincreasing_with_score():
    scores, is_target = _scores()
    pep = nonparametric_pep(scores, is_target)
    order = np.argsort(scores)  # ascending score
    # Higher score = better = lower PEP, so ascending-score PEP is non-increasing.
    assert np.all(np.diff(pep[order]) <= 1e-9)


def test_high_scores_lower_pep():
    scores, is_target = _scores()
    pep = nonparametric_pep(scores, is_target)
    top = scores >= np.quantile(scores, 0.9)
    bottom = scores <= np.quantile(scores, 0.1)
    assert pep[top].mean() < pep[bottom].mean()


def test_degenerate_all_one_class():
    scores = np.arange(10.0)
    assert np.all(nonparametric_pep(scores, np.ones(10, bool)) == 1.0)
    assert np.all(nonparametric_pep(scores, np.zeros(10, bool)) == 1.0)
