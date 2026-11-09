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
