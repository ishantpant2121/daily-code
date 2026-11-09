"""Dependency-free regression metrics for numeric prediction targets."""

from itertools import zip_longest
import math
from numbers import Real
from typing import Iterable


def _paired_values(y_true: Iterable[Real], y_pred: Iterable[Real]) -> tuple[list[float], list[float]]:
    missing = object()
    actual_values: list[float] = []
    predicted_values: list[float] = []
    for actual, predicted in zip_longest(y_true, y_pred, fillvalue=missing):
        if actual is missing or predicted is missing:
            raise ValueError("y_true and y_pred must have equal lengths")
        if (
            isinstance(actual, bool)
            or isinstance(predicted, bool)
            or not isinstance(actual, Real)
            or not isinstance(predicted, Real)
        ):
            raise ValueError("targets and predictions must be real numbers")
        try:
            actual_float = float(actual)
            predicted_float = float(predicted)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("targets and predictions must be finite") from None
        if not math.isfinite(actual_float) or not math.isfinite(predicted_float):
            raise ValueError("targets and predictions must be finite")
        actual_values.append(actual_float)
        predicted_values.append(predicted_float)
    if not actual_values:
        raise ValueError("y_true and y_pred must not be empty")
    return actual_values, predicted_values


def mean_absolute_error(y_true: Iterable[Real], y_pred: Iterable[Real]) -> float:
    """Return the mean absolute prediction error."""
    actual, predicted = _paired_values(y_true, y_pred)
    return math.fsum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def mean_squared_error(y_true: Iterable[Real], y_pred: Iterable[Real]) -> float:
    """Return the mean squared prediction error."""
    actual, predicted = _paired_values(y_true, y_pred)
    return math.fsum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)


def root_mean_squared_error(y_true: Iterable[Real], y_pred: Iterable[Real]) -> float:
    """Return the square root of mean squared error."""
    return math.sqrt(mean_squared_error(y_true, y_pred))


def r2_score(
    y_true: Iterable[Real],
    y_pred: Iterable[Real],
    *,
    force_finite: bool = True,
) -> float:
    """Return R-squared; at least two samples are required.

    For constant targets, the finite policy returns 1.0 for perfect
    predictions and 0.0 otherwise. Disable it to receive NaN or -infinity.
    """
    if type(force_finite) is not bool:
        raise ValueError("force_finite must be a boolean")
    actual, predicted = _paired_values(y_true, y_pred)
    if len(actual) < 2:
        raise ValueError("r2_score requires at least two samples")
    average = math.fsum(actual) / len(actual)
    residual_sum = math.fsum((a - p) ** 2 for a, p in zip(actual, predicted))
    total_sum = math.fsum((a - average) ** 2 for a in actual)
    if total_sum == 0:
        if force_finite:
            return 1.0 if residual_sum == 0 else 0.0
        return math.nan if residual_sum == 0 else -math.inf
    return 1.0 - residual_sum / total_sum


def _absolute_errors(actual: list[float], predicted: list[float]) -> list[float]:
    """Return finite absolute residuals for already validated paired data."""
    errors = []
    for target, estimate in zip(actual, predicted):
        error = abs(target - estimate)
        if not math.isfinite(error):
            raise ValueError("prediction error exceeds finite float range")
        errors.append(error)
    return errors


def median_absolute_error(y_true: Iterable[Real], y_pred: Iterable[Real]) -> float:
    """Return the median absolute difference between targets and predictions."""
    actual, predicted = _paired_values(y_true, y_pred)
    errors = sorted(_absolute_errors(actual, predicted))
    middle = len(errors) // 2
    if len(errors) % 2:
        return errors[middle]
    return errors[middle - 1] / 2.0 + errors[middle] / 2.0


def max_error(y_true: Iterable[Real], y_pred: Iterable[Real]) -> float:
    """Return the largest absolute difference between targets and predictions."""
    actual, predicted = _paired_values(y_true, y_pred)
    return max(_absolute_errors(actual, predicted))


def explained_variance_score(
    y_true: Iterable[Real],
    y_pred: Iterable[Real],
    *,
    force_finite: bool = True,
) -> float:
    """Return 1 - Var(y_true - y_pred) / Var(y_true), using population variance.

    Unlike R-squared, a constant prediction offset does not reduce this score:
    explained variance measures residual spread rather than residual bias.
    For constant targets, force_finite returns 1.0 when residual variance is
    zero and 0.0 otherwise; disabling it returns NaN or negative infinity.
    """
    if type(force_finite) is not bool:
        raise ValueError("force_finite must be a boolean")
    actual, predicted = _paired_values(y_true, y_pred)
    residuals = [target - estimate for target, estimate in zip(actual, predicted)]
    if any(not math.isfinite(value) for value in residuals):
        raise ValueError("prediction error exceeds finite float range")
    actual_mean = math.fsum(actual) / len(actual)
    residual_mean = math.fsum(residuals) / len(residuals)
    actual_variance = math.fsum(
        (value - actual_mean) ** 2 for value in actual
    ) / len(actual)
    residual_variance = math.fsum(
        (value - residual_mean) ** 2 for value in residuals
    ) / len(residuals)
    if actual_variance == 0:
        if force_finite:
            return 1.0 if residual_variance == 0 else 0.0
        return math.nan if residual_variance == 0 else -math.inf
    return 1.0 - residual_variance / actual_variance
