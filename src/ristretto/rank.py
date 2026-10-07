"""Within-group candidate ranking with known negatives (e.g. impossible modification sites)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ristretto._cv import spectrum_grouped_kfold
from ristretto.result import RankResult

logger = logging.getLogger(__name__)


def rank_within_groups(
    features: pd.DataFrame,
    *,
    group_col: str,
    negative_col: str,
    initial_score_col: str,
    feature_cols: list[str],
    train_col: str | None = None,
    n_folds: int = 3,
    max_rounds: int = 5,
    tol: float = 0.01,
    C: float = 0.1,
    seed: int = 42,
) -> RankResult:
    """
    Rank competing candidates within groups, learning from known-negative candidates.

    Typical use: the competing candidate identifications of one spectrum, with candidates that
    are wrong by construction as known negatives. In modification-aware rescoring a group holds
    every explanation of a spectrum (the unmodified hit and each modification at each site) and
    the negatives are the same modifications placed on residues they cannot occupy. Features are
    centred within each group, so only differences between candidates of the same group remain
    and group-level features (retention time, precursor mass, charge) drop out. Exclude features
    that separate candidates by construction rather than by evidence, such as precursor mass
    error when the candidates differ in mass.

    Semi-supervised: the top non-negative candidate per group under ``initial_score_col`` is
    the initial positive, known negatives are the negatives, all other candidates are
    unlabelled. A logistic regression is fitted with group-wise cross-validation, positives are
    relabelled with the new top per group, and this repeats until fewer than ``tol`` of the
    groups change their top candidate or ``max_rounds`` is reached.

    Parameters
    ----------
    features
        One row per candidate. Must contain ``group_col``, ``negative_col``,
        ``initial_score_col`` and ``feature_cols``.
    group_col
        Column identifying the group a candidate competes in.
    negative_col
        Boolean column marking known-negative candidates.
    initial_score_col
        Score used to pick the initial positive per group (e.g. a rescoring score or
        hyperscore). Ignored for negatives, so it may be missing for them.
    feature_cols
        Feature columns; centred within group and standardized before fitting.
    train_col
        Optional boolean column marking rows that may be used for training. Rows outside it
        are still scored. Use it to exclude decoy-peptide groups whose "positive" is wrong.
    n_folds, max_rounds, tol, C, seed
        Cross-validation folds, relabelling rounds, convergence tolerance (fraction of groups),
        logistic regression regularisation, random seed.

    Returns
    -------
    RankResult
        ``scores`` (index preserved) with ``score`` and ``rank`` (1 = best in group, over all
        candidates including negatives), the learned feature weights per fold, the fraction of
        groups whose top candidate is a known negative, and the number of rounds run.

    """
    groups = pd.factorize(features[group_col])[0]
    negative = features[negative_col].to_numpy(dtype=bool)
    trainable = (
        features[train_col].to_numpy(dtype=bool) if train_col else np.ones(len(features), bool)
    )
    X = features[feature_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X)
    X -= pd.DataFrame(X).groupby(groups).transform("mean").to_numpy()
    folds = spectrum_grouped_kfold(groups, n_folds, seed)

    def top_per_group(score: np.ndarray) -> np.ndarray:
        """Row index of the best non-negative candidate per group."""
        order = np.lexsort((-np.where(negative, -np.inf, score), groups))
        first = np.r_[True, groups[order][1:] != groups[order][:-1]]
        return order[first]

    initial = features[initial_score_col].to_numpy(dtype=np.float64)
    positives = top_per_group(np.nan_to_num(initial, nan=-np.inf))
    scores = np.zeros(len(features))
    n_rounds = 0
    fold_weights: list[np.ndarray] = []
    for n_rounds in range(1, max_rounds + 1):
        y = np.full(len(features), -1)
        y[negative] = 0
        y[positives] = 1
        fold_weights = []  # keep the final round's weights only
        for train_idx, test_idx in folds:
            train_idx = train_idx[(y[train_idx] >= 0) & trainable[train_idx]]
            scaler = StandardScaler().fit(X[train_idx])
            est = LogisticRegression(C=C, max_iter=2000).fit(
                scaler.transform(X[train_idx]), y[train_idx]
            )
            scores[test_idx] = est.decision_function(scaler.transform(X[test_idx]))
            fold_weights.append(np.asarray(est.coef_, dtype=np.float64).ravel())
        new_positives = top_per_group(scores)
        changed = np.mean(new_positives != positives)
        positives = new_positives
        logger.debug(f"Round {n_rounds}: top candidate changed in {changed:.1%} of groups")
        if changed < tol:
            break

    order = np.lexsort((-scores, groups))
    rank = np.empty(len(features), dtype=np.int64)
    rank[order] = np.arange(len(features)) - np.searchsorted(groups[order], groups[order])
    rank += 1
    negative_top_rate = float(negative[rank == 1].mean())
    logger.info(
        f"Ranked {len(np.unique(groups))} groups in {n_rounds} rounds; top candidate is a known "
        f"negative in {negative_top_rate:.1%} of groups."
    )
    return RankResult(
        scores=pd.DataFrame({"score": scores, "rank": rank}, index=features.index),
        feature_weights=pd.DataFrame(
            {f"fold_{i}": w for i, w in enumerate(fold_weights, start=1)},
            index=pd.Index(feature_cols, name="feature"),
        ),
        negative_top_rate=negative_top_rate,
        n_rounds=n_rounds,
    )
