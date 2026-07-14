"""Unit tests for the fitting strategies and FittedModel."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ristretto._iterative import (
    FittedModel,
    _best_initial_direction,
    _flip_sign,
    _single_pass_fit,
    iterative_fit,
)
from ristretto._model import get_factory


def _separable(seed=0, n=200, n_feat=3):
    rng = np.random.default_rng(seed)
    n_t = n * 3 // 4
    X = rng.normal(0.0, 1.0, (n, n_feat))
    is_target = np.zeros(n, bool)
    is_target[:n_t] = True
    X[is_target, 0] += 3.0  # feature 0 separates classes
    return X, is_target


def test_fittedmodel_score_estimator_mode():
    X, is_target = _separable()
    scaler = StandardScaler().fit(X)
    est = LinearSVC(dual=False).fit(scaler.transform(X), is_target.astype(int))
    fm = FittedModel(scaler=scaler, estimator=est, n_iters=1)
    s = fm.score(X)
    assert s.shape == (len(X),)
    np.testing.assert_allclose(s, est.decision_function(scaler.transform(X)).ravel())


def test_fittedmodel_score_feature_mode():
    X = np.arange(12.0).reshape(6, 2)
    scaler = StandardScaler().fit(X)
    fm = FittedModel(scaler=scaler, estimator=None, n_iters=0, feature_index=1, feature_sign=-1)
    np.testing.assert_allclose(fm.score(X), -scaler.transform(X)[:, 1])


def test_fittedmodel_weights():
    X, is_target = _separable(n_feat=4)
    scaler = StandardScaler().fit(X)
    est = LinearSVC(dual=False).fit(scaler.transform(X), is_target.astype(int))
    fm = FittedModel(scaler=scaler, estimator=est, n_iters=1)
    assert fm.weights(4).shape == (4,)

    fallback = FittedModel(
        scaler=scaler, estimator=None, n_iters=0, feature_index=2, feature_sign=-1
    )
    w = fallback.weights(4)
    assert w[2] == -1 and np.count_nonzero(w) == 1


def test_flip_sign():
    X, is_target = _separable()
    est = LinearSVC(dual=False).fit(X, is_target.astype(int))
    coef0 = est.coef_.copy()
    _flip_sign(est)
    np.testing.assert_allclose(est.coef_, -coef0)


def test_best_initial_direction_finds_separating_feature():
    X, is_target = _separable()
    Xs = StandardScaler().fit_transform(X)
    idx, sign, n = _best_initial_direction(Xs, is_target, train_fdr=0.05)
    assert idx == 0  # feature 0 is the separating one
    assert n > 0


def test_single_pass_fit_targets_score_higher():
    X, is_target = _separable()
    fm = _single_pass_fit(X, is_target, get_factory("lda"))
    s = fm.score(X)
    assert s[is_target].mean() > s[~is_target].mean()
    assert fm.n_iters == 1


def test_iterative_fit_converges_and_orients():
    X, is_target = _separable()
    fm = iterative_fit(X, is_target, get_factory("lda"), train_fdr=0.05, max_iter=10)
    assert fm.estimator is not None
    s = fm.score(X)
    assert s[is_target].mean() > s[~is_target].mean()


def test_iterative_fit_fallback_to_feature_when_no_model():
    # max_iter=0 -> loop never runs -> no trained model -> best-feature fallback.
    X, is_target = _separable()
    fm = iterative_fit(X, is_target, get_factory("lda"), train_fdr=0.05, max_iter=0)
    assert fm.estimator is None
    assert fm.feature_index == 0  # the separating feature
    s = fm.score(X)
    assert s[is_target].mean() > s[~is_target].mean()
