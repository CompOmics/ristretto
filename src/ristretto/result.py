"""Result container returned by `rescore`."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RescoreResult:
    """Scores and FDR estimates at PSM level and each requested rollup level."""

    psms: pd.DataFrame
    """Identifier columns plus ``score``/``qvalue``/``pep``; row order and index preserved."""
    peptidoforms: pd.DataFrame
    """Peptidoform-level rollup (always computed)."""
    peptides: pd.DataFrame | None
    """Peptide-level rollup. None if no ``peptide_col`` was provided."""
    proteins: pd.DataFrame | None
    """Protein-level rollup. None if no ``protein_col`` was provided."""
    pi0: float
    """Estimated null-target fraction at PSM level (n_decoys / n_targets)."""
    n_iterations: list[int]
    """Number of refinement iterations per outer CV fold."""
    feature_weights: pd.DataFrame
    """Learned feature weights per fold (index = feature, columns = fold_1..k),
    in standardized-feature space. Fallback folds show a one-hot on the chosen feature."""


@dataclass
class RankResult:
    """Within-group ranking returned by `rank_within_groups`."""

    scores: pd.DataFrame
    """``score`` and ``rank`` (1 = best in group) per candidate; row order and index preserved."""
    negative_top_rate: float
    """Fraction of groups whose top-ranked candidate is a known negative (a false-localisation
    rate estimate when negatives are decoy sites and one decoy is generated per real site)."""
    n_rounds: int
    """Number of relabelling rounds run."""
