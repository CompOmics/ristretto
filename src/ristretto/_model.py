"""Classifier factories: LDA and linear SVM."""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import LinearSVC

EstimatorFactory = Callable[[], BaseEstimator]

_CLASS_WEIGHT_GRID = [{0: wn, 1: wp} for wn in (0.1, 1.0, 10.0) for wp in (0.1, 1.0, 10.0)]


def lda_factory() -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis()


def svm_factory(seed: int = 7, n_jobs: int = 1) -> GridSearchCV:
    base = LinearSVC(dual=False, random_state=seed, max_iter=5000)
    return GridSearchCV(
        base,
        param_grid={"class_weight": _CLASS_WEIGHT_GRID},
        cv=KFold(n_splits=3, shuffle=True, random_state=seed),
        scoring="accuracy",
        n_jobs=n_jobs,
        refit=True,
    )


def get_factory(name: str, *, seed: int = 7, n_jobs: int = 1) -> EstimatorFactory:
    if name == "lda":
        return lda_factory
    if name == "svm":
        return lambda: svm_factory(seed=seed, n_jobs=n_jobs)
    raise ValueError(f"Unknown model: {name!r}. Expected 'lda' or 'svm'.")
