# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `rank_within_groups()`: semi-supervised ranking of competing candidates within groups
  (e.g. modification-site candidates of one spectrum), learning from known-negative
  candidates such as decoy sites. Features are centred within group, a logistic regression
  is fitted with group-wise cross-validation and positives are relabelled with the new top
  per group until convergence. Returns a `RankResult` with per-candidate score and rank and
  the fraction of groups whose top candidate is a known negative.

## [0.3.1] - 2026-08-21

### Fixed

- `tdc_qvalues()` gives PSMs with identical scores one shared q-value. Ties are grouped and
  each block gets one FDR/q-value from the full target/decoy counts, as in mokapot's
  `qvalues.py`. Previously, q-values within a tie depended on PSM order.

### Changed

- Rewrote the README About section.
- Renamed "Käll loop" to "refinement loop" in logging and docstrings.
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
