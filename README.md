# ristretto

Lean, fast PSM rescoring for proteomics, purpose-built for use in
[MS²Rescore](https://github.com/compomics/ms2rescore).

---

## About

ristretto is a small, dependency-light reimplementation of the
[Percolator](https://doi.org/10.1038/nmeth.1664) semi-supervised identification
rescoring algorithm. Like [Mokapot](https://doi.org/10.1021/acs.jproteome.0c01010)
it leverages the scikit-learn library for machine learning operations. However,
as ristretto was written for leanness, it is not as flexible as Mokapot.

Given a table of peptide-spectrum matches (PSMs) with rescoring features, a
target/decoy label, and a spectrum identifier, it trains a classifier to separate
correct from incorrect matches, then reports scores, q-values, and posterior error
probabilities (PEPs) at the PSM, peptide, and protein level.

Design goals:

- **Lean.** Three runtime dependencies: `numpy`, `scikit-learn`, `pandas`.
- **Fast.** Hyperparameters are tuned once rather than every iteration, and hot
  paths are vectorized.
- **Simple API.** One function in, one result object out. A pandas DataFrame is
  the only input; column roles are given by name.

Two models are available:

- `svm` (default): a linear SVM trained with the semi-supervised iterative
  procedure of [Käll et al. 2007](https://doi.org/10.1038/nmeth1113), matching
  the Percolator/mokapot default.
- `lda`: single-pass Fisher linear discriminant analysis, in the style of
  [Sage](https://doi.org/10.1021/acs.jproteome.3c00486). Faster, no iteration.

## Installation

```bash
pip install ristretto
```

## Quickstart

```python
import pandas as pd
from ristretto import rescore

# One row per PSM: feature columns + a decoy flag + a spectrum identifier.
features = pd.read_parquet("rescoring_features.parquet")

result = rescore(
    features,
    is_decoy_col="is_decoy",
    spectrum_id_col="spectrum_id",
    peptidoform_col="peptidoform",  # required
    peptide_col="sequence",         # optional: stripped seq, peptide-level rollup
    protein_col="proteins",         # optional: protein-level rollup
    decoy_pattern="rev_",           # optional: enables picked-protein competition
    model="svm",
    n_jobs=8,
)

# PSM table: identifier columns + score, qvalue, pep
psms = result.psms
confident = psms[(~psms["is_decoy"]) & (psms["qvalue"] <= 0.01)]
print(f"{len(confident)} PSMs at 1% FDR")

# Higher aggregation levels (columns: <key>, score, qvalue, pep, is_decoy, n_psms)
result.peptidoforms   # always
result.peptides       # if peptide_col
result.proteins       # if protein_col

# Rejoin features to scored PSMs via the preserved index
features.loc[result.psms.index]

# Per-fold learned feature weights (feature x fold)
result.feature_weights.to_csv("weights.tsv", sep="\t")
```

## Input

`rescore` takes a single DataFrame with one row per PSM. Required:

- **feature columns**: all numeric columns by default, or pass `feature_cols`
  explicitly. Must be finite (no NaN/inf).
- **`is_decoy_col`**: boolean, True for decoy PSMs.
- **`spectrum_id_col`**: groups PSMs by spectrum for cross-validation and
  competition.
- **`peptidoform_col`**: a unique peptidoform key. A peptidoform rollup is always
  returned, grouping by this column verbatim (no parsing, same as `peptide_col`/
  `protein_col`). By convention, peptidoform-level FDR is charge-independent, so
  this should typically hold sequence + modifications only, without charge.

Optional `peptide_col` (stripped sequence) and `protein_col` enable the
peptide- and protein-level rollups.

## How it works

1. **Spectrum-grouped cross-validation.** PSMs are split into `n_folds` folds by
   spectrum, so all PSMs of a spectrum stay in one fold. Each fold is scored by a
   model trained on the others, preventing data leakage.
2. **Model training.** For `svm`, the refinement loop (Käll et al. 2007): label targets below
   `train_fdr` as positives and all decoys as negatives, fit, rescore, repeat
   until the positive set stabilizes. The SVM class weight is tuned once (first
   iteration) and reused. If no trained model beats the best single feature, that
   feature is used instead. For `lda`, a single fit on all targets vs decoys.
3. **Spectrum competition.** By default (`multi_rank_rescoring=False`) only the
   best-scoring PSM per spectrum is kept, matching Percolator/mokapot/Sage. Set to
   True to keep and score all PSMs (all search-engine ranks).
4. **FDR estimation.** q-values by target-decoy competition; PEPs by a
   histogram-based binomial model with isotonic smoothing.
5. **Rollup.** Peptidoform and peptide levels use classic target-decoy
   competition (best PSM per group). Protein level uses picked-protein
   competition when `decoy_pattern` is given (target and decoy accessions paired
   by stripping the tag, higher score wins), or classic competition otherwise.

## Result

`rescore` returns a `RescoreResult` with:

- `psms`: the identifier columns (`spectrum_id_col`, `is_decoy_col`,
  `peptidoform_col`, and `peptide_col`/`protein_col` if given) plus `score`,
  `qvalue`, `pep`. Features and other input columns are left out; the input row
  order and index are preserved, so they can be rejoined with
  `features.loc[result.psms.index]`.
- `peptidoforms`: peptidoform-level table (always).
- `peptides`: peptide-level table, or None if no `peptide_col`.
- `proteins`: protein-level table, or None if no `protein_col`.
- `pi0`: estimated fraction of incorrect targets.
- `n_iterations`: refinement iterations per fold.
- `feature_weights`: per-fold learned feature weights.

## Related projects

- [MS²Rescore](https://github.com/compomics/ms2rescore): the rescoring pipeline
  ristretto is built to plug into.
- [MS²PIP](https://github.com/compomics/ms2pip): fragment intensity prediction,
  a source of rescoring features.
- [DeepLC](https://github.com/compomics/deeplc): retention time prediction.
- [psm_utils](https://github.com/compomics/psm_utils): PSM parsing and handling.
- [mokapot](https://github.com/wfondrie/mokapot): the reference semi-supervised
  rescorer this library is modeled on.
- [Percolator](https://github.com/percolator/percolator): the original
  semi-supervised target-decoy rescoring method.

## References

- Käll, L., Canterbury, J. D., Weston, J., Noble, W. S., & MacCoss, M. J. (2007).
  Semi-supervised learning for peptide identification from shotgun proteomics
  datasets. *Nature Methods*.
  [doi:10.1038/nmeth1113](https://doi.org/10.1038/nmeth1113)
- Käll, L., Storey, J. D., MacCoss, M. J., & Noble, W. S. (2008). Posterior error
  probabilities and false discovery rates: two sides of the same coin.
  *Journal of Proteome Research*.
  [doi:10.1021/pr700739d](https://doi.org/10.1021/pr700739d)
- Savitski, M. M., Wilhelm, M., Hahne, H., Kuster, B., & Bantscheff, M. (2015). A
  scalable approach for protein false discovery rate estimation in large
  proteomic data sets. *Molecular & Cellular Proteomics*.
  [doi:10.1074/mcp.M114.046995](https://doi.org/10.1074/mcp.M114.046995)
- Fondrie, W. E., & Noble, W. S. (2021). mokapot: fast and flexible
  semisupervised learning for peptide detection. *Journal of Proteome Research*.
  [doi:10.1021/acs.jproteome.0c01010](https://doi.org/10.1021/acs.jproteome.0c01010)
