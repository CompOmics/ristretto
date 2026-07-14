"""Public entry point: `rescore(features, ...)`."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ristretto._cv import spectrum_grouped_kfold
from ristretto._iterative import _single_pass_fit, iterative_fit
from ristretto._model import get_factory
from ristretto._pep import nonparametric_pep
from ristretto._qvalue import tdc_qvalues
from ristretto._rollup import picked_rollup, rollup
from ristretto.result import RescoreResult

logger = logging.getLogger(__name__)

_RESERVED_COLS = {"score", "qvalue", "pep"}


def rescore(
    features: pd.DataFrame,
    *,
    is_decoy_col: str = "is_decoy",
    spectrum_id_col: str = "spectrum_id",
    peptidoform_col: str = "peptidoform",
    peptide_col: str | None = None,
    protein_col: str | None = None,
    feature_cols: list[str] | None = None,
    decoy_pattern: str | None = None,
    model: str = "svm",
    n_folds: int = 3,
    train_fdr: float = 0.01,
    max_iter: int = 10,
    seed: int = 42,
    n_jobs: int = 1,
    multi_rank_rescoring: bool = False,
) -> RescoreResult:
    """
    Rescore PSMs with a semi-supervised model and target-decoy FDR control.

    Parameters
    ----------
    features
        One row per PSM. Must contain ``is_decoy_col``, ``spectrum_id_col``, and
        ``peptidoform_col``. All other numeric columns are treated as features
        unless ``feature_cols`` is given.
    is_decoy_col
        Boolean column marking decoy PSMs.
    spectrum_id_col
        Column used to group PSMs by spectrum for CV splitting and competition.
    peptidoform_col
        Column with a unique peptidoform key. Required; a peptidoform-level rollup
        is always returned, grouping PSMs by this column verbatim (no parsing or
        transformation, same as ``peptide_col``/``protein_col``). By convention,
        peptidoform-level FDR is charge-independent, so this should typically hold
        sequence + modifications only, without charge -- include a charge suffix
        only if charge-resolved (precursor-level) grouping is what you want.
    peptide_col
        Column with a stripped peptide sequence (no modifications or charge). If
        given, a peptide-level rollup is returned.
    protein_col
        Column with a per-protein key. If given, a protein-level rollup is returned.
    feature_cols
        Explicit list of feature columns. If None, inferred as all numeric columns
        except the metadata columns and any existing score/qvalue/pep columns.
    decoy_pattern
        Regular expression matching the decoy tag in a protein accession
        (e.g. ``"rev_"`` for Sage, ``"_DECOY$"``). If given, protein-level FDR
        uses picked-protein competition: decoy and target accessions are paired
        by stripping this pattern, and only the higher-scoring form of each pair
        survives. If None, classic target-decoy competition is used. For protein
        groups (e.g. ``"rev_P1;rev_P2"``), use an unanchored pattern so the tag
        is stripped from every member; pairing then assumes target and decoy
        list members in the same order.
    model
        ``"svm"`` uses LinearSVC with iterative Käll 2007 training. ``"lda"``
        uses single-pass Fisher LDA.
    n_folds
        Number of outer CV folds (spectrum-grouped).
    train_fdr
        q-value threshold for positive selection in the iterative loop.
    max_iter
        Maximum Käll iterations. Ignored for ``model="lda"``.
    seed
        RNG seed for fold assignment and model training.
    n_jobs
        Parallel workers for inner GridSearchCV (SVM only).
    multi_rank_rescoring
        If False (default), keep only the best-scoring PSM per spectrum before
        computing q-values, PEP, and rollups, matching Percolator/mokapot/Sage.
        If True, all PSMs (all search-engine ranks) are retained and scored.

    Returns
    -------
    RescoreResult
        Scores and FDR at PSM, peptidoform, and (optionally) peptide and protein
        level, plus per-fold feature weights. ``result.psms`` keeps the input row
        order and index (as a subset), so features or other input columns can be
        rejoined with ``features.loc[result.psms.index]``.

    """
    if model not in ("svm", "lda"):
        raise ValueError(f"model must be 'svm' or 'lda', got {model!r}")

    feature_cols = _validate_and_resolve_feature_cols(
        features,
        is_decoy_col,
        spectrum_id_col,
        peptidoform_col,
        peptide_col,
        protein_col,
        decoy_pattern,
        feature_cols,
    )

    X, is_target, groups = _extract_arrays(
        features, feature_cols, is_decoy_col, spectrum_id_col, model
    )

    scores, iters, fold_weights = _cross_validate(
        X,
        is_target,
        groups,
        model=model,
        n_folds=n_folds,
        train_fdr=train_fdr,
        max_iter=max_iter,
        seed=seed,
        n_jobs=n_jobs,
        n_features=len(feature_cols),
    )

    keep, q, pep, pi0 = _compete_and_estimate_fdr(scores, is_target, groups, multi_rank_rescoring)

    id_cols = [
        c
        for c in (spectrum_id_col, is_decoy_col, peptidoform_col, peptide_col, protein_col)
        if c is not None
    ]
    psms, feature_weights = _assemble_output(
        features, keep, id_cols, feature_cols, scores[keep], q, pep, fold_weights
    )

    peptidoforms, peptides, proteins = _build_rollups(
        psms,
        scores[keep],
        is_target[keep],
        peptidoform_col,
        peptide_col,
        protein_col,
        decoy_pattern,
    )

    return RescoreResult(
        psms=psms,
        peptidoforms=peptidoforms,
        peptides=peptides,
        proteins=proteins,
        pi0=pi0,
        n_iterations=iters,
        feature_weights=feature_weights,
    )


def _validate_and_resolve_feature_cols(
    features: pd.DataFrame,
    is_decoy_col: str,
    spectrum_id_col: str,
    peptidoform_col: str,
    peptide_col: str | None,
    protein_col: str | None,
    decoy_pattern: str | None,
    feature_cols: list[str] | None,
) -> list[str]:
    """Validate required/reserved columns and resolve the feature column list."""
    required = {is_decoy_col, spectrum_id_col, peptidoform_col}
    required.update(c for c in (peptide_col, protein_col) if c is not None)
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"features is missing required column(s): {sorted(missing)}")

    group_cols = (peptidoform_col, peptide_col, protein_col)
    reserved = {c for c in group_cols if c in _RESERVED_COLS}
    if reserved:
        raise ValueError(
            f"peptidoform_col/peptide_col/protein_col cannot be one of "
            f"{sorted(_RESERVED_COLS)}; got {sorted(reserved)} "
            f"(these names are used in rollup output)"
        )

    if decoy_pattern is not None and protein_col is None:
        logger.warning("decoy_pattern is set but protein_col is None; it will be ignored")

    if feature_cols is not None:
        return feature_cols

    meta_cols = {is_decoy_col, spectrum_id_col, peptidoform_col, peptide_col, protein_col}
    meta_cols -= {None}
    resolved = [
        c
        for c in features.select_dtypes(include="number").columns
        if c not in meta_cols and c not in _RESERVED_COLS
    ]
    if not resolved:
        raise ValueError("No feature columns found; pass feature_cols explicitly")
    return resolved


def _extract_arrays(
    features: pd.DataFrame,
    feature_cols: list[str],
    is_decoy_col: str,
    spectrum_id_col: str,
    model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the feature matrix, target labels, and spectrum groups; validate them."""
    X = features[feature_cols].to_numpy(dtype=np.float64)
    is_target = ~features[is_decoy_col].to_numpy(dtype=bool)
    groups = features[spectrum_id_col].to_numpy()

    if not np.all(np.isfinite(X)):
        raise ValueError("Feature matrix contains NaN or inf")
    if is_target.all() or (~is_target).all():
        raise ValueError("features must contain both target and decoy PSMs")

    n_targets = int(is_target.sum())
    logger.info(
        "Rescoring %d PSMs (%d targets, %d decoys) with %d features using model %r",
        len(features),
        n_targets,
        len(features) - n_targets,
        len(feature_cols),
        model,
    )
    return X, is_target, groups


def _cross_validate(
    X: np.ndarray,
    is_target: np.ndarray,
    groups: np.ndarray,
    *,
    model: str,
    n_folds: int,
    train_fdr: float,
    max_iter: int,
    seed: int,
    n_jobs: int,
    n_features: int,
) -> tuple[np.ndarray, list[int], list[np.ndarray]]:
    """Fit and score each spectrum-grouped fold; return held-out scores and diagnostics."""
    factory = get_factory(model, seed=seed, n_jobs=n_jobs)
    scores = np.full(len(X), np.nan, dtype=np.float64)
    iters: list[int] = []
    fold_weights: list[np.ndarray] = []

    for fold, (train_idx, test_idx) in enumerate(
        spectrum_grouped_kfold(groups, n_folds, seed), start=1
    ):
        logger.info(
            "Fold %d/%d: training on %d PSMs, scoring %d",
            fold,
            n_folds,
            len(train_idx),
            len(test_idx),
        )
        if model == "lda":
            fitted = _single_pass_fit(X[train_idx], is_target[train_idx], factory)
        else:
            fitted = iterative_fit(
                X[train_idx],
                is_target[train_idx],
                factory,
                train_fdr=train_fdr,
                max_iter=max_iter,
            )
        scores[test_idx] = fitted.score(X[test_idx])
        iters.append(fitted.n_iters)
        fold_weights.append(fitted.weights(n_features))
        logger.debug("Fold %d converged after %d iteration(s)", fold, fitted.n_iters)

    assert np.all(np.isfinite(scores)), "Some PSMs were not scored"
    return scores, iters, fold_weights


def _compete_and_estimate_fdr(
    scores: np.ndarray,
    is_target: np.ndarray,
    groups: np.ndarray,
    multi_rank_rescoring: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply spectrum competition unless multi-rank rescoring; compute q-values, PEP, pi0."""
    if multi_rank_rescoring:
        keep = np.arange(len(scores))
    else:
        keep = _best_per_spectrum(scores, groups)
        logger.info(
            "Spectrum competition: %d PSMs -> %d best per spectrum", len(scores), len(keep)
        )

    kept_scores = scores[keep]
    kept_is_target = is_target[keep]

    logger.info("Computing PSM-level q-values and PEPs")
    q = tdc_qvalues(kept_scores, kept_is_target)
    pep = nonparametric_pep(kept_scores, kept_is_target)
    pi0 = float((~kept_is_target).sum() / max(int(kept_is_target.sum()), 1))
    logger.debug("PSM-level pi0 estimate: %.4f", pi0)

    return keep, q, pep, pi0


def _assemble_output(
    features: pd.DataFrame,
    keep: np.ndarray,
    id_cols: list[str],
    feature_cols: list[str],
    kept_scores: np.ndarray,
    q: np.ndarray,
    pep: np.ndarray,
    fold_weights: list[np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the PSM output table (identifiers + score/qvalue/pep) and feature-weight table.

    Feature columns are intentionally left out of the PSM table: the caller already
    holds them, and the original index is preserved so they can be rejoined via
    ``features.loc[psms.index]``.
    """
    meta = features.iloc[keep][id_cols]
    added = pd.DataFrame({"score": kept_scores, "qvalue": q, "pep": pep}, index=meta.index)
    psms = pd.concat([meta, added], axis=1)

    feature_weights = pd.DataFrame(
        {f"fold_{i}": w for i, w in enumerate(fold_weights, start=1)},
        index=pd.Index(feature_cols, name="feature"),
    )
    return psms, feature_weights


def _build_rollups(
    psms: pd.DataFrame,
    kept_scores: np.ndarray,
    kept_is_target: np.ndarray,
    peptidoform_col: str,
    peptide_col: str | None,
    protein_col: str | None,
    decoy_pattern: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Compute the peptidoform (always), peptide (optional), and protein (optional) rollups."""
    logger.info("Rolling up to peptidoform level on column %r", peptidoform_col)
    peptidoforms = rollup(psms, peptidoform_col, kept_scores, kept_is_target)

    peptides = None
    if peptide_col:
        logger.info("Rolling up to peptide level on column %r", peptide_col)
        peptides = rollup(psms, peptide_col, kept_scores, kept_is_target)

    proteins = None
    if protein_col:
        if decoy_pattern:
            logger.info(
                "Picked-protein rollup on column %r (decoy_pattern=%r)",
                protein_col,
                decoy_pattern,
            )
            proteins = picked_rollup(psms, protein_col, kept_scores, kept_is_target, decoy_pattern)
        else:
            logger.info("Rolling up to protein level on column %r", protein_col)
            proteins = rollup(psms, protein_col, kept_scores, kept_is_target)

    return peptidoforms, peptides, proteins


def _best_per_spectrum(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Return sorted indices of the best-scoring PSM per spectrum group."""
    _, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(scores, kind="stable")
    best = np.empty(inverse.max() + 1, dtype=np.int64)
    best[inverse[order]] = order
    return np.sort(best)
