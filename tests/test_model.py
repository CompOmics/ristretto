"""Unit tests for classifier factories."""

from __future__ import annotations

import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

from ristretto._model import get_factory


def test_lda_factory():
    factory = get_factory("lda")
    est = factory()
    assert isinstance(est, LinearDiscriminantAnalysis)


def test_svm_factory_is_grid_search():
    factory = get_factory("svm", seed=1, n_jobs=2)
    est = factory()
    assert isinstance(est, GridSearchCV)
    assert est.n_jobs == 2
    # 9-point class-weight grid.
    assert len(est.param_grid["class_weight"]) == 9


def test_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        get_factory("xgboost")
