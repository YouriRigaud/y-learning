# Changelog

<!-- Use this template for new versions:
## [v.v.v] - yyyy-mm-dd
### Added
-

### Changed
-

### Fixed
-

### Deprecated
-
-->

## [0.1.2] - 2025-11-05
### Added
- `BaseLinearModel` abstract class with `fit`, `predict` and `score` for linear regression method
- Solvers for the linear models (`normal`, `qr` and `qr_ridge`)
- `SolverFactory` for ease of use of the solvers
- `OLS` ordinary least square regression model
- `Ridge` regression model

## [0.1.1] - 2025-11-04
### Added
- `KNNRegressor` with `_predict` method using mean
- `MSE` metric
- `compare_knn_regressor` test

### Changed
- `euclidean_distance` is now in the new `utils` module, no more in `base_knn`

## [0.1.0] - 2025-11-04
### Added
- `BaseKNN` abstract class with `fit`, `predict` and `score`
- `KNNClassifier` with `_predict` method using majority vote
- `metrics` module with `r2_score` and `accuracy_score`
- `tests` module with `knn_tests` for knn classifier

## [0.0.0] - 2025-11-04
### Added
- `BaseEstimator` abstract class with `fit`, `predict` and `score`