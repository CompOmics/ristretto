"""Spectrum-grouped k-fold cross-validation."""

from __future__ import annotations

import numpy as np


def spectrum_grouped_kfold(
    groups: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition PSMs into n_splits folds so every group stays in one fold."""
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(groups, return_inverse=True)

    # Round-robin fold assignment over a shuffled group order.
    perm = rng.permutation(len(unique))
    fold_of_unique = np.empty(len(unique), dtype=np.int64)
    fold_of_unique[perm] = np.arange(len(unique)) % n_splits
    fold_ids = fold_of_unique[inverse]

    return [(np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]) for k in range(n_splits)]
