"""Unit tests for within-group ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ristretto import rank_within_groups


def _site_families(seed=0, n_groups=300):
    """Each group: one true site, one wrong real site, two decoy sites.

    Feature 0 carries site evidence (high for the true site), feature 1 is group-level noise
    (identical within a group), feature 2 is pure noise. The initial score is only weakly
    informative, so the ranker has to learn from the decoys.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        level = rng.normal(0, 5)
        for kind in ("true", "wrong", "decoy", "decoy"):
            rows.append(
                {
                    "group": g,
                    "negative": kind == "decoy",
                    "kind": kind,
                    "f0": rng.normal(3.0 if kind == "true" else 0.0, 1.0),
                    "f1": level,
                    "f2": rng.normal(),
                    "initial": rng.normal(0.5 if kind == "true" else 0.0, 1.0),
                }
            )
    return pd.DataFrame(rows)


def test_rank_within_groups_recovers_true_site():
    df = _site_families()
    result = rank_within_groups(
        df,
        group_col="group",
        negative_col="negative",
        initial_score_col="initial",
        feature_cols=["f0", "f1", "f2"],
    )
    assert list(result.scores.index) == list(df.index)
    top = df[result.scores["rank"] == 1]
    assert (top["kind"] == "true").mean() > 0.9
    assert result.negative_top_rate < 0.05
    assert 1 <= result.n_rounds <= 5
    # every group has exactly one rank-1 candidate and ranks run 1..4
    assert (result.scores.groupby(df["group"])["rank"].min() == 1).all()
    assert (result.scores.groupby(df["group"])["rank"].max() == 4).all()


def test_initial_score_alone_is_worse_than_ranker():
    df = _site_families(seed=1)
    initial_top = df.loc[df.groupby("group")["initial"].idxmax(), "kind"]
    result = rank_within_groups(
        df,
        group_col="group",
        negative_col="negative",
        initial_score_col="initial",
        feature_cols=["f0", "f1", "f2"],
    )
    ranker_top = df.loc[result.scores["rank"] == 1, "kind"]
    assert (ranker_top == "true").mean() > (initial_top == "true").mean() + 0.2
