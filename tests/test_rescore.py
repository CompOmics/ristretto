"""Integration tests for the public rescore() entry point."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ristretto

FIXTURE = Path(__file__).parent.parent / "test-data" / "smoke_sample.parquet"


def _synthetic(seed=0, n_spectra=200, per_spectrum=3):
    """Separable synthetic feature table with all identifier columns."""
    rng = np.random.default_rng(seed)
    n = n_spectra * per_spectrum
    is_decoy = rng.random(n) < 0.3
    f1 = rng.normal(0, 1, n) + np.where(is_decoy, 0.0, 3.0)
    f2 = rng.normal(0, 1, n)
    spectrum_id = np.repeat(np.arange(n_spectra), per_spectrum)
    seq = np.array([f"PEPTIDE{i % 137}" for i in range(n)])
    charge = rng.integers(2, 4, n).astype(str)
    return pd.DataFrame(
        {
            "spectrum_id": spectrum_id,
            "is_decoy": is_decoy,
            "peptidoform": np.char.add(np.char.add(seq, "/"), charge),
            "peptide": seq,
            "f1": f1,
            "f2": f2,
        }
    )


def test_synthetic_scores_well_formed():
    df = _synthetic()
    result = ristretto.rescore(df, model="lda", n_folds=3, seed=42)

    q = result.psms["qvalue"].to_numpy()
    pep = result.psms["pep"].to_numpy()
    is_target = ~result.psms["is_decoy"].to_numpy(dtype=bool)
    assert np.all(np.isfinite(q)) and np.all((q >= 0) & (q <= 1))
    assert np.all(np.isfinite(pep)) and np.all((pep >= 0) & (pep <= 1))
    assert result.psms.loc[is_target, "score"].mean() > result.psms.loc[~is_target, "score"].mean()


def test_psms_columns_are_identifiers_only():
    df = _synthetic()
    result = ristretto.rescore(df, model="lda", seed=42, peptide_col="peptide")
    assert list(result.psms.columns) == [
        "spectrum_id",
        "is_decoy",
        "peptidoform",
        "peptide",
        "score",
        "qvalue",
        "pep",
    ]


def test_order_and_index_preserved():
    df = _synthetic()
    # Keep all PSMs: identical index and row order to input.
    keep_all = ristretto.rescore(df, model="lda", seed=42, multi_rank_rescoring=True)
    assert keep_all.psms.index.equals(df.index)

    # Competed: a subset of the input index, still in original order, rejoinable.
    competed = ristretto.rescore(df, model="lda", seed=42)
    assert competed.psms.index.is_monotonic_increasing
    assert competed.psms.index.isin(df.index).all()
    rejoined = df.loc[competed.psms.index]
    assert np.array_equal(
        rejoined["peptidoform"].to_numpy(), competed.psms["peptidoform"].to_numpy()
    )


def test_svm_end_to_end():
    df = _synthetic()
    result = ristretto.rescore(df, model="svm", n_folds=3, seed=42)
    q = result.psms["qvalue"].to_numpy()
    is_target = ~result.psms["is_decoy"].to_numpy(dtype=bool)
    assert np.all((q >= 0) & (q <= 1))
    assert result.psms.loc[is_target, "score"].mean() > result.psms.loc[~is_target, "score"].mean()
    assert result.feature_weights.shape == (2, 3)
    assert all(n >= 1 for n in result.n_iterations)


def test_multi_rank_rescoring_default_and_on():
    df = _synthetic()
    competed = ristretto.rescore(df, model="lda", seed=42)
    assert len(competed.psms) == df["spectrum_id"].nunique()

    keep_all = ristretto.rescore(df, model="lda", seed=42, multi_rank_rescoring=True)
    assert len(keep_all.psms) == len(df)


def test_pi0_matches_ratio():
    df = _synthetic()
    r = ristretto.rescore(df, model="lda", seed=42, multi_rank_rescoring=True)
    is_t = ~df["is_decoy"].to_numpy(bool)
    assert r.pi0 == pytest.approx((~is_t).sum() / is_t.sum())


def test_feature_weights_shape():
    df = _synthetic()
    r = ristretto.rescore(df, model="lda", n_folds=4, seed=42)
    assert list(r.feature_weights.columns) == [f"fold_{i}" for i in range(1, 5)]
    assert list(r.feature_weights.index) == ["f1", "f2"]


def test_peptidoform_rollup_always():
    df = _synthetic()
    r = ristretto.rescore(df, model="lda", seed=42)
    assert r.peptidoforms is not None
    assert {"peptidoform", "score", "qvalue", "pep", "is_decoy", "n_psms"}.issubset(
        r.peptidoforms.columns
    )
    assert r.peptidoforms["n_psms"].sum() == len(r.psms)
    assert r.peptides is None and r.proteins is None


def test_peptide_rollup_optional():
    df = _synthetic()
    r = ristretto.rescore(df, model="lda", seed=42, peptide_col="peptide")
    assert r.peptides is not None
    # peptidoform_col includes a charge suffix in this fixture, so distinct
    # charge states of the same sequence form separate peptidoform groups but
    # collapse into one peptide group.
    assert len(r.peptides) <= len(r.peptidoforms)
    assert r.peptides["n_psms"].sum() == len(r.psms)


def test_peptidoform_col_is_an_opaque_key():
    # peptidoform_col is grouped verbatim, with no parsing or charge stripping:
    # distinct charge-suffixed strings form distinct peptidoform groups, exactly
    # like rollup() applied to peptide_col/protein_col.
    df = _synthetic()
    r = ristretto.rescore(df, model="lda", seed=42, multi_rank_rescoring=True)
    assert r.peptidoforms["peptidoform"].str.contains("/").all()


def test_picked_protein_rollup():
    df = _synthetic()
    base = (np.arange(len(df)) % 40).astype(str)
    df = df.assign(protein=np.where(df["is_decoy"], "rev_PROT" + base, "PROT" + base))
    r = ristretto.rescore(df, model="lda", seed=42, protein_col="protein", decoy_pattern="^rev_")
    assert r.proteins is not None
    assert not r.proteins["protein"].str.startswith("rev_").any()
    assert r.proteins["n_psms"].sum() == len(r.psms)


def test_validation_missing_column():
    df = _synthetic()
    with pytest.raises(ValueError, match="missing required column"):
        ristretto.rescore(df.drop(columns=["is_decoy"]), model="lda")


def test_validation_missing_peptidoform():
    df = _synthetic().drop(columns=["peptidoform"])
    with pytest.raises(ValueError, match="missing required column"):
        ristretto.rescore(df, model="lda")


def test_validation_empty_features():
    df = _synthetic()[["spectrum_id", "is_decoy", "peptidoform"]]
    with pytest.raises(ValueError, match="No feature columns"):
        ristretto.rescore(df, model="lda")


def test_validation_reserved_group_col():
    df = _synthetic().rename(columns={"f1": "score"})
    with pytest.raises(ValueError, match="cannot be one of"):
        ristretto.rescore(df, model="lda", peptide_col="score")


def test_validation_bad_model():
    df = _synthetic()
    with pytest.raises(ValueError, match="model must be"):
        ristretto.rescore(df, model="randomforest")


def test_validation_single_class():
    df = _synthetic()
    df = df[~df["is_decoy"]]
    with pytest.raises(ValueError, match="both target and decoy"):
        ristretto.rescore(df, model="lda")


def test_decoy_pattern_without_protein_warns(caplog):
    df = _synthetic()
    with caplog.at_level(logging.WARNING, logger="ristretto.rescore"):
        ristretto.rescore(df, model="lda", seed=42, decoy_pattern="^rev_")
    assert any("decoy_pattern" in r.message for r in caplog.records)


def test_fixture_end_to_end():
    df = pd.read_parquet(FIXTURE)
    r = ristretto.rescore(df, model="lda", seed=42)
    assert len(r.psms) == df["spectrum_id"].nunique()
    assert r.peptidoforms is not None
    q = r.psms["qvalue"].to_numpy()
    is_t = ~r.psms["is_decoy"].to_numpy(bool)
    assert int((is_t & (q <= 0.05)).sum()) > 0
