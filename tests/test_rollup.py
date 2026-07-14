"""Unit tests for peptide/protein rollup."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ristretto._rollup import picked_rollup, rollup


def test_rollup_best_psm_per_group():
    df = pd.DataFrame({"peptide": ["A", "A", "B", "B", "B"]})
    scores = np.array([1.0, 5.0, 2.0, 3.0, 0.0])
    is_target = np.array([True, True, True, False, True])
    out = rollup(df, "peptide", scores, is_target)

    out = out.set_index("peptide")
    assert out.loc["A", "score"] == 5.0  # best of A
    assert out.loc["B", "score"] == 3.0  # best of B
    assert out.loc["A", "n_psms"] == 2
    assert out.loc["B", "n_psms"] == 3
    # B's best PSM is a decoy.
    assert bool(out.loc["B", "is_decoy"]) is True


def test_rollup_counts_sum_to_input():
    df = pd.DataFrame({"g": ["x", "y", "x", "z", "y", "x"]})
    scores = np.arange(6.0)
    is_target = np.array([True] * 4 + [False] * 2)
    out = rollup(df, "g", scores, is_target)
    assert out["n_psms"].sum() == len(df)


def test_picked_strips_tag_and_pairs():
    # Two proteins, each with a target and a decoy form (prefixed).
    df = pd.DataFrame({"prot": ["P1", "rev_P1", "P2", "rev_P2"]})
    scores = np.array([5.0, 1.0, 2.0, 9.0])  # P1 target wins; P2 decoy wins
    is_target = np.array([True, False, True, False])
    out = picked_rollup(df, "prot", scores, is_target, decoy_pattern="rev_").set_index("prot")

    assert set(out.index) == {"P1", "P2"}  # base ids, tag stripped
    assert out.loc["P1", "score"] == 5.0
    assert bool(out.loc["P1", "is_decoy"]) is False  # target form won
    assert out.loc["P2", "score"] == 9.0
    assert bool(out.loc["P2", "is_decoy"]) is True  # decoy form won


def test_picked_group_string_unanchored():
    df = pd.DataFrame({"prot": ["P1;P2", "rev_P1;rev_P2"]})
    scores = np.array([4.0, 7.0])
    is_target = np.array([True, False])
    out = picked_rollup(df, "prot", scores, is_target, decoy_pattern="rev_")
    assert list(out["prot"]) == ["P1;P2"]
    assert out["n_psms"].iloc[0] == 2  # both forms counted
    assert bool(out["is_decoy"].iloc[0]) is True


def test_picked_counts_sum():
    df = pd.DataFrame({"prot": ["P1", "P1", "rev_P1", "P2"]})
    scores = np.array([1.0, 3.0, 2.0, 4.0])
    is_target = np.array([True, True, False, True])
    out = picked_rollup(df, "prot", scores, is_target, decoy_pattern="rev_")
    assert out["n_psms"].sum() == len(df)
