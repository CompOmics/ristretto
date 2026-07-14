"""Rollup from PSM level to peptide or protein level."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ristretto._pep import nonparametric_pep
from ristretto._qvalue import tdc_qvalues


def _best_per_group(
    keys: np.ndarray, psm_scores: np.ndarray, psm_is_target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (unique_keys, best_scores, best_is_target, n_psms) per group key."""
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    n_groups = len(unique_keys)

    n_psms = np.bincount(inverse, minlength=n_groups)

    # Best PSM per group: writing indices in ascending-score order means the
    # last write per group (its highest score) wins.
    order = np.argsort(psm_scores, kind="stable")
    best_idx = np.empty(n_groups, dtype=np.int64)
    best_idx[inverse[order]] = order

    return unique_keys, psm_scores[best_idx], psm_is_target[best_idx], n_psms


def rollup(
    psms: pd.DataFrame,
    group_col: str,
    psm_scores: np.ndarray,
    psm_is_target: np.ndarray,
) -> pd.DataFrame:
    """
    Aggregate PSMs to a higher level by picking the best-scoring PSM per group.

    For each unique value in ``group_col``, selects the PSM with the highest
    score, then computes TDC q-values and PEP on the resulting set. Target and
    decoy keys are distinct groups (classic target-decoy competition).

    Parameters
    ----------
    psms
        PSM-level DataFrame. Must contain ``group_col``.
    group_col
        Column to group by (e.g. stripped peptide sequence, protein accession).
    psm_scores
        Score per PSM, aligned with ``psms`` index.
    psm_is_target
        Target/decoy label per PSM.

    Returns
    -------
    pd.DataFrame
        One row per unique value in ``group_col`` with columns:
        ``group_col``, ``score``, ``qvalue``, ``pep``, ``is_decoy``, ``n_psms``.

    """
    keys = psms[group_col].to_numpy()
    unique_keys, best_scores, group_is_target, n_psms = _best_per_group(
        keys, psm_scores, psm_is_target
    )
    q = tdc_qvalues(best_scores, group_is_target)
    pep = nonparametric_pep(best_scores, group_is_target)

    return pd.DataFrame(
        {
            group_col: unique_keys,
            "score": best_scores,
            "qvalue": q,
            "pep": pep,
            "is_decoy": ~group_is_target,
            "n_psms": n_psms,
        }
    )


def picked_rollup(
    psms: pd.DataFrame,
    group_col: str,
    psm_scores: np.ndarray,
    psm_is_target: np.ndarray,
    decoy_pattern: str,
) -> pd.DataFrame:
    """
    Picked-group competition (Savitski et al. 2015).

    Each group has a target form and a decoy form whose IDs share a base once
    ``decoy_pattern`` is stripped (e.g. ``rev_P12345`` -> ``P12345``). Within a
    base, only the higher-scoring form survives; TDC q-values and PEP are then
    computed on the surviving set. Used for protein-level FDR, where decoy
    accessions carry a prefix or suffix tag.

    Parameters
    ----------
    psms
        PSM-level DataFrame. Must contain ``group_col``.
    group_col
        Column of protein (or group) accessions.
    psm_scores
        Score per PSM, aligned with ``psms`` index.
    psm_is_target
        Target/decoy label per PSM.
    decoy_pattern
        Regular expression matching the decoy tag in an accession. It is
        removed to pair a decoy accession with its target counterpart.

    Returns
    -------
    pd.DataFrame
        One row per base accession with columns ``group_col`` (base id),
        ``score``, ``qvalue``, ``pep``, ``is_decoy``, ``n_psms``.

    """
    keys = psms[group_col].to_numpy()
    unique_keys, best_scores, group_is_target, n_psms = _best_per_group(
        keys, psm_scores, psm_is_target
    )

    rx = re.compile(decoy_pattern)
    base = np.array([rx.sub("", k) for k in unique_keys])
    ubase, binv = np.unique(base, return_inverse=True)
    n_base = len(ubase)

    base_counts = np.bincount(binv, weights=n_psms, minlength=n_base).astype(np.int64)

    # Winner form per base = highest-scoring form.
    order = np.argsort(best_scores, kind="stable")
    win = np.empty(n_base, dtype=np.int64)
    win[binv[order]] = order

    win_scores = best_scores[win]
    win_is_target = group_is_target[win]
    q = tdc_qvalues(win_scores, win_is_target)
    pep = nonparametric_pep(win_scores, win_is_target)

    return pd.DataFrame(
        {
            group_col: ubase,
            "score": win_scores,
            "qvalue": q,
            "pep": pep,
            "is_decoy": ~win_is_target,
            "n_psms": base_counts,
        }
    )
