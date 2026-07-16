"""Tests for the public evaluate() entry point."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ristretto
from ristretto._qvalue import tdc_qvalues
from ristretto._rollup import rollup


def _synthetic(seed=0, n_spectra=200, per_spectrum=3):
    """Separable synthetic feature table with all identifier columns and a precomputed score."""
    rng = np.random.default_rng(seed)
    n = n_spectra * per_spectrum
    is_decoy = rng.random(n) < 0.3
    score = rng.normal(0, 1, n) + np.where(is_decoy, 0.0, 3.0)
    spectrum_id = np.repeat(np.arange(n_spectra), per_spectrum)
    seq = np.array([f"PEPTIDE{i % 137}" for i in range(n)])
    charge = rng.integers(2, 4, n).astype(str)
    return pd.DataFrame(
        {
            "spectrum_id": spectrum_id,
            "is_decoy": is_decoy,
            "peptidoform": np.char.add(np.char.add(seq, "/"), charge),
            "peptide": seq,
            "score": score,
        }
    )


def test_evaluate_well_formed():
    df = _synthetic()
    result = ristretto.evaluate(df)

    q = result.psms["qvalue"].to_numpy()
    pep = result.psms["pep"].to_numpy()
    assert np.all(np.isfinite(q)) and np.all((q >= 0) & (q <= 1))
    assert np.all(np.isfinite(pep)) and np.all((pep >= 0) & (pep <= 1))
    assert result.n_iterations == []
    assert result.feature_weights.empty


def test_evaluate_matches_direct_tdc_qvalues_when_multi_rank():
    df = _synthetic()
    result = ristretto.evaluate(df, multi_rank_rescoring=True)

    is_target = ~df["is_decoy"].to_numpy(bool)
    expected_q = tdc_qvalues(df["score"].to_numpy(), is_target)
    assert np.allclose(result.psms["qvalue"].to_numpy(), expected_q)
    assert len(result.psms) == len(df)


def test_evaluate_competes_to_best_per_spectrum_by_default():
    df = _synthetic()
    result = ristretto.evaluate(df)
    assert len(result.psms) == df["spectrum_id"].nunique()


def test_evaluate_peptidoform_rollup_matches_rollup_helper():
    df = _synthetic()
    result = ristretto.evaluate(df, multi_rank_rescoring=True)

    is_target = ~df["is_decoy"].to_numpy(bool)
    expected = rollup(df, "peptidoform", df["score"].to_numpy(), is_target)
    pd.testing.assert_frame_equal(
        result.peptidoforms.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_evaluate_peptide_col_optional():
    df = _synthetic()
    result = ristretto.evaluate(df, peptide_col="peptide")
    assert result.peptides is not None
    assert result.proteins is None


def test_evaluate_picked_protein_rollup():
    df = _synthetic()
    base = (np.arange(len(df)) % 40).astype(str)
    df = df.assign(protein=np.where(df["is_decoy"], "rev_PROT" + base, "PROT" + base))
    result = ristretto.evaluate(df, protein_col="protein", decoy_pattern="^rev_")
    assert result.proteins is not None
    assert not result.proteins["protein"].str.startswith("rev_").any()


def test_evaluate_missing_score_col_raises():
    df = _synthetic().drop(columns=["score"])
    with pytest.raises(ValueError, match="score_col"):
        ristretto.evaluate(df)


def test_evaluate_missing_identifier_col_raises():
    df = _synthetic().drop(columns=["is_decoy"])
    with pytest.raises(ValueError, match="missing required column"):
        ristretto.evaluate(df)


def test_evaluate_single_class_raises():
    df = _synthetic()
    df = df[~df["is_decoy"]]
    with pytest.raises(ValueError, match="both target and decoy"):
        ristretto.evaluate(df)


def test_evaluate_psms_columns_are_identifiers_only():
    df = _synthetic()
    result = ristretto.evaluate(df, peptide_col="peptide")
    assert list(result.psms.columns) == [
        "spectrum_id",
        "is_decoy",
        "peptidoform",
        "peptide",
        "score",
        "qvalue",
        "pep",
    ]
