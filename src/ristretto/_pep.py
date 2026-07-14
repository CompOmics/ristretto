"""
PEP estimation using a histogram-based binomial model.

Mirrors the hist_nnls approach from mokapot (Fondrie et al.), replacing the
NNLS monotonization step with sklearn's weighted isotonic regression, which
avoids adding scipy as a dependency.

For TDC data the null-target fraction is simply n_decoys / n_targets
(Käll et al. 2008), so no separate pi0 estimation step is needed.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def nonparametric_pep(
    scores: np.ndarray,
    is_target: np.ndarray,
    n_bins: int | None = None,
) -> np.ndarray:
    """
    Estimate PEP from scores using a histogram-based binomial model.

    Steps:
      1. Bin scores into a joint histogram for targets and decoys.
      2. Estimate the expected number of null targets per bin as
         factor * decoy_count, where factor = n_decoys / n_targets (TDC).
      3. Raw PEP per bin = expected_nulls / observed_targets.
      4. Weighted isotonic regression (decreasing, weight = target count)
         enforces monotonicity and suppresses noise in sparse bins.
      5. Linearly interpolate from bin centers back to per-PSM scores.

    Parameters
    ----------
    scores
        Per-PSM scores. Higher = better.
    is_target
        Target/decoy label per PSM.
    n_bins
        Number of histogram bins. Auto-selected if None.

    """
    scores = np.asarray(scores, dtype=np.float64)
    is_target = np.asarray(is_target, dtype=bool)

    n_targets = int(is_target.sum())
    n_decoys = int((~is_target).sum())

    if n_targets == 0 or n_decoys == 0:
        return np.ones(len(scores))

    # TDC: each decoy represents one expected null target
    factor = n_decoys / n_targets

    if n_bins is None:
        n_bins = max(10, min(500, len(scores) // 10))

    bin_edges = np.histogram_bin_edges(scores, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    target_counts, _ = np.histogram(scores[is_target], bins=bin_edges)
    decoy_counts, _ = np.histogram(scores[~is_target], bins=bin_edges)

    n = target_counts.astype(float)
    k = np.clip(factor * decoy_counts, 0.0, None)

    with np.errstate(invalid="ignore", divide="ignore"):
        raw_pep = np.where(n > 0, np.clip(k / n, 0.0, 1.0), 1.0)

    # Weight by target count: bins with more data pull the fit harder
    weights = np.maximum(n, 1.0)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=0.0, y_max=1.0)
    smooth_pep = iso.fit_transform(bin_centers, raw_pep, sample_weight=weights)

    pep = np.interp(scores, bin_centers, smooth_pep)
    return np.clip(pep, 0.0, 1.0)
