"""Käll 2007 semi-supervised iterative target-selection loop."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler

from ristretto._qvalue import tdc_qvalues

logger = logging.getLogger(__name__)

EstimatorFactory = Callable[[], BaseEstimator]


@dataclass
class FittedModel:
    """
    A fitted rescoring model plus the scaler used to standardize features.

    Either wraps a fitted estimator, or (as a fallback) a single feature index
    and sign when no trained model beat the best single feature.
    """

    scaler: StandardScaler
    estimator: BaseEstimator | None
    n_iters: int
    feature_index: int | None = None
    feature_sign: int = 1

    def score(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        if self.estimator is not None:
            return np.asarray(self.estimator.decision_function(Xs)).ravel()
        return self.feature_sign * Xs[:, self.feature_index]

    def weights(self, n_features: int) -> np.ndarray:
        """
        Linear feature weights in standardized space.

        For a trained linear model, the estimator's ``coef_``. For a
        best-single-feature fallback, a one-hot vector on that feature.
        """
        if self.estimator is None:
            w = np.zeros(n_features)
            w[self.feature_index] = self.feature_sign
            return w
        inner = getattr(self.estimator, "best_estimator_", self.estimator)
        return np.asarray(inner.coef_, dtype=np.float64).ravel()


def _decision(est: BaseEstimator, X: np.ndarray) -> np.ndarray:
    return np.asarray(est.decision_function(X)).ravel()


def _n_passing(scores: np.ndarray, is_target: np.ndarray, train_fdr: float) -> int:
    q = tdc_qvalues(scores, is_target)
    return int(np.sum(is_target & (q <= train_fdr)))


def _best_initial_direction(
    X_scaled: np.ndarray, is_target: np.ndarray, train_fdr: float
) -> tuple[int, int, int]:
    """Return (feature_index, sign, n_passing) of the best single feature."""
    best_idx, best_sign, best_n = 0, 1, -1
    for j in range(X_scaled.shape[1]):
        for sign in (1, -1):
            n = _n_passing(sign * X_scaled[:, j], is_target, train_fdr)
            if n > best_n:
                best_n, best_idx, best_sign = n, j, sign
    return best_idx, best_sign, best_n


def _flip_sign(est: BaseEstimator) -> None:
    inner = getattr(est, "best_estimator_", est)
    if hasattr(inner, "coef_"):
        inner.coef_ = -inner.coef_
    if hasattr(inner, "intercept_"):
        inner.intercept_ = -inner.intercept_


def _single_pass_fit(
    X: np.ndarray, is_target: np.ndarray, factory: EstimatorFactory
) -> FittedModel:
    """Fit a model once on all PSMs (targets vs decoys), no iteration."""
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    est = factory()
    est.fit(Xs, is_target.astype(int))
    scores = _decision(est, Xs)
    if scores[is_target].mean() < scores[~is_target].mean():
        _flip_sign(est)
    return FittedModel(scaler=scaler, estimator=est, n_iters=1)


def iterative_fit(
    X: np.ndarray,
    is_target: np.ndarray,
    factory: EstimatorFactory,
    train_fdr: float = 0.01,
    max_iter: int = 10,
) -> FittedModel:
    """
    Käll 2007 loop: relabel positives by q <= train_fdr, refit, repeat.

    Hyperparameters (e.g. the SVM class weight) are tuned only on the first
    iteration. If ``factory`` produces a tuning estimator (one exposing
    ``best_estimator_`` after fitting, such as ``GridSearchCV``), the chosen
    estimator is reused for every subsequent iteration by cloning its
    parameters, avoiding a full hyperparameter search per iteration.

    If the trained model passes fewer targets at ``train_fdr`` than the best
    single feature does, the best single feature is returned instead (matching
    the Percolator/mokapot fallback).
    """
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    best_feat_idx, best_feat_sign, best_feat_n = _best_initial_direction(Xs, is_target, train_fdr)
    scores = best_feat_sign * Xs[:, best_feat_idx]

    prev_positive: np.ndarray | None = None
    est: BaseEstimator | None = None
    tuned: BaseEstimator | None = None
    n_iters = 0

    for it in range(1, max_iter + 1):
        q = tdc_qvalues(scores, is_target)
        positives_mask = is_target & (q <= train_fdr)
        negatives_mask = ~is_target

        if prev_positive is not None and np.array_equal(positives_mask, prev_positive):
            logger.debug("Käll loop converged at iteration %d (positive set stable)", it)
            break
        prev_positive = positives_mask.copy()

        if positives_mask.sum() == 0:
            raise RuntimeError(
                "Käll loop: no targets passed train_fdr. Increase train_fdr "
                "or check input features."
            )

        train_mask = positives_mask | negatives_mask
        y_train = positives_mask[train_mask].astype(int)

        if tuned is None:
            search = factory()
            search.fit(Xs[train_mask], y_train)
            est = getattr(search, "best_estimator_", search)
            tuned = est
            best_params = getattr(search, "best_params_", None)
            if best_params is not None:
                logger.debug("Tuned hyperparameters: %s", best_params)
        else:
            est = clone(tuned)
            est.fit(Xs[train_mask], y_train)

        scores = _decision(est, Xs)
        if scores[is_target].mean() < scores[~is_target].mean():
            _flip_sign(est)
            scores = -scores
        n_iters = it
        logger.debug(
            "Käll iteration %d: %d positives at train_fdr=%.3g",
            it,
            int(positives_mask.sum()),
            train_fdr,
        )

    if n_iters == max_iter:
        logger.debug("Käll loop hit max_iter=%d without a stable positive set", max_iter)

    # Fallback: if the best single feature separates better than the trained
    # model on the training fold, use the feature direction instead.
    if est is None or _n_passing(scores, is_target, train_fdr) < best_feat_n:
        logger.debug(
            "Falling back to best single feature (index %d, sign %d): "
            "it passes %d targets vs the model's %d",
            best_feat_idx,
            best_feat_sign,
            best_feat_n,
            _n_passing(scores, is_target, train_fdr) if est is not None else 0,
        )
        return FittedModel(
            scaler=scaler,
            estimator=None,
            n_iters=n_iters,
            feature_index=best_feat_idx,
            feature_sign=best_feat_sign,
        )

    return FittedModel(scaler=scaler, estimator=est, n_iters=n_iters)
