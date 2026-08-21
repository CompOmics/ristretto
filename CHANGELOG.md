# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.1] - 2026-08-21

### Fixed

- `tdc_qvalues()` no longer gives PSMs with identical scores different,
  order-dependent q-values. Previously each tied PSM was scored from the
  partial target/decoy counts implied by its arbitrary position within the
  tie, letting low-cardinality features (e.g. missed-cleavage count, or a raw
  search-engine score with many repeated values) inflate PSM-level FDR passes
  by chance and destabilize the refinement loop's positive-set relabeling.
  Ties are now grouped and assigned one shared, fully-accumulated FDR/q-value
  per block, mirroring mokapot's tie-grouping in `qvalues.py`.

### Changed

- Rewrote the README About section: ristretto is now described as a lean,
  dependency-light reimplementation of the Percolator algorithm, noting the
  scikit-learn dependency shared with mokapot and the leanness/flexibility
  tradeoff.
- Renamed "Käll loop" to "refinement loop" throughout logging and
  docstrings, keeping one Käll et al. (2007) citation per file instead of
  repeating the attribution on every log line.
- Per-fold, spectrum competition, q-value/PEP, and rollup log messages moved
  from info to debug level; only the top-level "Rescoring N PSMs..." summary
  stays at info.

## [0.3.0] - 2026-07-16

### Added

- `run_col` support for multi-run input, so spectrum IDs that are not
  globally unique across runs are grouped correctly for CV splitting and
  spectrum competition.

## [0.2.0] - 2026-07-16

### Added

- `evaluate()`, for running competition, FDR, PEP, and rollups on an
  already-computed score column without training a model.

## [0.1.0] - 2026-07-14

### Changed

- Initial PyPI release, as `ristretto-ms`.
