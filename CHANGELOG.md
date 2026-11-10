# Changelog

## Unreleased

- Synchronized the CLI and built distribution version through the package version source.

- Added timed cross-validation diagnostics and out-of-fold prediction helpers.
- Added deterministic finite parameter grids, sampling, exhaustive and randomized search.
- Added weighted hard/soft voting ensembles with estimator cloning and class alignment.
- Added regression coverage and package installation for the new model-selection utilities.

- Added mean, median, most-frequent, and constant feature imputation with empty-column policies.
- Added leakage-safe transformer and estimator pipelines.
- Added deterministic CART decision-tree classification with Gini/entropy criteria and feature importances.
- Added reproducible cross-validation scoring with optional stratification and fresh estimator copies.
- Added finite-weight validation for Dijkstra and minimum-spanning-tree inputs.
- Added Bellman-Ford, Floyd-Warshall, and iterative strongly connected components.
- Added stratified train/test and K-fold splitting utilities.
- Added multiclass confusion matrices and tie-aware ROC AUC scoring.
- Added a dependency-free K-nearest-neighbors classifier.
- Added RobustScaler and polynomial feature expansion.
- Added interpolated quantile statistics and overflow-safe even-sized medians.
- Added collision-verified Rabin-Karp matching and LCS sequence reconstruction.
- Added generator support to counting sort and strict quickselect index validation.
- Added finite-real validation and overflow checks to matrix multiplication.
- Tightened CSV profiler validation for option types, delimiters, encodings, and missing markers.
- Added balanced accuracy, multiclass Matthews correlation, ROC and precision-recall curves, average precision, Brier score, and binary log loss.
- Added median absolute error, maximum error, explained variance, and squared logarithmic regression errors.
- Added variance-threshold selection, one-hot encoding, and reversible label encoding.
- Added binary logistic regression, Gaussian Naive Bayes, and K-nearest-neighbors regression.
- Added dependency-free PCA with a Jacobi symmetric eigensolver and k-means clustering with k-means++ initialization.
- Normalized oversized numeric conversion errors across preprocessing, regression, and nearest-neighbor inputs.
- Expanded installed-package smoke tests for the new machine-learning modules.
- Expanded regression tests for edge cases, invalid inputs, and iterable behavior.

## 0.1.0 — 2026-11-04

- Added a standard-library CSV profiler with delimiter detection, type inference,
  missing/distinct counts, numeric ranges, and bounded structural diagnostics.
- Added text and JSON command-line reports plus strict-mode exit status.
- Added DSA regression coverage and cycle-safe linked-list conversion.
- Added package metadata, usage documentation, contribution guidance, and CI.
