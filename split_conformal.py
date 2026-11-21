"""Dependency-free split conformal prediction for regression."""

import math
from numbers import Real
from typing import Iterable


def _finite_values(values: Iterable[Real], name: str) -> list[float]:
    try:
        items = list(values)
    except TypeError:
        raise ValueError(f"{name} must be iterable") from None
    if not items:
        raise ValueError(f"{name} must not be empty")
    converted: list[float] = []
    for value in items:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must contain finite real numbers")
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError):
            raise ValueError(f"{name} must contain finite real numbers") from None
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain finite real numbers")
        converted.append(number)
    return converted


def _paired_scores(
    calibration_targets: Iterable[Real], calibration_predictions: Iterable[Real]
) -> list[float]:
    targets = _finite_values(calibration_targets, "calibration_targets")
    predictions = _finite_values(calibration_predictions, "calibration_predictions")
    if len(targets) != len(predictions):
        raise ValueError("calibration targets and predictions must have equal lengths")
    # Finite operands can overflow when subtracted; an infinite residual is
    # conservative and produces unbounded intervals if selected by the rank.
    return [abs(target - prediction) for target, prediction in zip(targets, predictions)]


def _validated_intervals(
    intervals: Iterable[Iterable[Real]],
) -> list[tuple[float, float]]:
    try:
        pairs = list(intervals)
    except TypeError:
        raise ValueError("intervals must be iterable") from None
    if not pairs:
        raise ValueError("intervals must not be empty")
    result: list[tuple[float, float]] = []
    for pair in pairs:
        if isinstance(pair, (str, bytes)):
            raise ValueError("each interval must contain exactly two numeric endpoints")
        try:
            endpoints = list(pair)
        except TypeError:
            raise ValueError("each interval must contain exactly two numeric endpoints") from None
        if len(endpoints) != 2:
            raise ValueError("each interval must contain exactly two numeric endpoints")
        converted: list[float] = []
        for endpoint in endpoints:
            if isinstance(endpoint, bool) or not isinstance(endpoint, Real):
                raise ValueError("interval endpoints must be real numbers")
            try:
                number = float(endpoint)
            except (OverflowError, TypeError, ValueError):
                raise ValueError("interval endpoints must not be NaN") from None
            if math.isnan(number):
                raise ValueError("interval endpoints must not be NaN")
            converted.append(number)
        lower, upper = converted
        if lower > upper:
            raise ValueError("interval lower endpoint must not exceed upper endpoint")
        result.append((lower, upper))
    return result


class SplitConformalRegressor:
    """Create symmetric regression intervals from held-out calibration residuals.

    Fit using predictions from a model trained on a separate training split.
    Predictions passed to predict should come from that already-fitted model;
    this estimator does not fit a predictive model itself.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if isinstance(alpha, bool) or not isinstance(alpha, Real):
            raise ValueError("alpha must be a real number strictly between 0 and 1")
        try:
            value = float(alpha)
        except (OverflowError, TypeTypeError, ValueError):
            raise ValueError("alpha must be a real number strictly between 0 and 1") from None
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError("alpha must be a real number strictly between 0 and 1")
        self._alpha = value
        self._fitted = False

    @property
    def alpha(self) -> float:
        """Nominal miscoverage rate supplied at construction."""
        return self._alpha

    def fit(
        self,
        calibration_targets: Iterable[Real],
        calibration_predictions: Iterable[Real],
    ) -> "SplitConformalRegressor":
        """Calibrate the residual quantile from paired held-out predictions.

        Uses absolute residuals and the finite-sample rank
        ceil((n + 1) * (1 - alpha)). If that rank exceeds the calibration
        sample count, the threshold is infinity as required by the correction.
        """
        self._fitted = False
        for name in (
            "calibration_scores_", "n_calibration_", "rank_", "quantile_", "alpha_"
        ):
            if hasattr(self, name):
                delattr(self, name)
        scores = _paired_scores(calibration_targets, calibration_predictions)
        n = len(scores)
        rank = math.ceil((n + 1) * (1.0 - self.alpha))
        sorted_scores = sorted(scores)
        quantile = sorted_scores[rank - 1] if rank <= n else math.inf
        self.calibration_scores_ = sorted_scores
        self.n_calibration_ = n
        self.rank_ = rank
        self.quantile_ = quantile
        self.alpha_ = self.alpha
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before predict")

    def predict(self, predictions: Iterable[Real]) -> list[tuple[float, float]]:
        """Return inclusive symmetric prediction intervals for model predictions."""
        self._check_fitted()
        centers = _finite_values(predictions, "predictions")
        q = self.quantile_
        return [(center - q, center + q) for center in centers]


def empirical_coverage(
    targets: Iterable[Real], intervals: Iterable[Iterable[Real]]
) -> float:
    """Return the observed fraction of targets contained in inclusive intervals."""
    actual = _finite_values(targets, "targets")
    bounds = _validated_intervals(intervals)
    if len(actual) != len(bounds):
        raise ValueError("targets and intervals must have equal lengths")
    return math.fsum(
        1.0 for target, (lower, upper) in zip(actual, bounds)
        if lower <= target <= upper
    ) / len(actual)


def mean_interval_width(intervals: Iterable[Iterable[Real]]) -> float:
    """Return mean interval width, or infinity when any interval is unbounded."""
    bounds = _validated_intervals(intervals)
    widths: list[float] = []
    for lower, upper in bounds:
        if not math.isfinite(lower) or not math.isfinite(upper):
            return math.inf
        width = upper - lower
        if not math.isfinite(width):
            return math.inf
        widths.append(width)
    scale = max(widths)
    if scale == 0.0:
        return 0.0
    return scale * (math.fsum(width / scale for width in widths) / len(widths))
