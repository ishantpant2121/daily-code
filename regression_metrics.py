"""Dependency-free regression metrics for numeric prediction targets."""

from itertools import zip_longest
import math
import sys
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


def _stable_mean(values: list[float], message: str) -> float:
    """Return a mean without overflowing an otherwise representable average."""
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    normalized_mean = math.fsum(value / scale for value in values) / len(values)
    result = scale * normalized_mean
    if not math.isfinite(result):
        raise ValueError(message)
    return result


def mean_absolute_error(y_true: Iterable[Real], y_pred: Iterable[Real]) -> float:
    """Return the mean absolute prediction error."""
    actual, predicted = _paired_values(y_true, y_pred)
    return _stable_mean(
        _absolute_errors(actual, predicted),
        "mean absolute error exceeds finite floating-point range",
    )


def _scaled_mean_square(values: list[float], message: str) -> float:
    """Return a mean square after normalizing values before squaring."""
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    normalized_mean_square = math.fsum((value / scale) ** 2 for value in values) / len(values)
    result = scale * (scale * normalized_mean_square)
    if not math.isfinite(result):
        raise ValueError(message)
    return result


def _scaled_root_mean_square(values: list[float], message: str) -> float:
    """Return an RMS directly so a representable root is not lost to square underflow."""
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    normalized_mean_square = math.fsum((value / scale) ** 2 for value in values) / len(values)
    result = scale * math.sqrt(normalized_mean_square)
    if not math.isfinite(result):
        raise ValueError(message)
    return result


def mean_squared_error(y_true: Iterable[Real], y_pred: Iterable[Real]) -> float:
    """Return the mean squared prediction error."""
    actual, predicted = _paired_values(y_true, y_pred)
    return _scaled_mean_square(
        _absolute_errors(actual, predicted),
        "mean squared error exceeds finite floating-point range",
    )


def root_mean_squared_error(y_true: Iterable[Real], y_pred: Iterable[Real]) -> float:
    """Return the square root of mean squared error."""
    actual, predicted = _paired_values(y_true, y_pred)
    return _scaled_root_mean_square(
        _absolute_errors(actual, predicted),
        "root mean squared error exceeds finite floating-point range",
    )


def _scaled_square_ratio(
    numerator_scale: float,
    denominator_scale: float,
    numerator: float,
    denominator: float,
) -> float:
    """Return a ratio of scaled sums of squares without squaring large scales."""
    if numerator == 0.0:
        return 0.0
    if denominator == 0.0:
        return math.inf
    normalized_ratio = numerator / denominator
    scale_ratio = numerator_scale / denominator_scale
    if not math.isfinite(normalized_ratio) or not math.isfinite(scale_ratio):
        return math.inf
    root = scale_ratio * math.sqrt(normalized_ratio)
    if not math.isfinite(root) or root > math.sqrt(sys.float_info.max):
        return math.inf
    result = root * root
    return result if math.isfinite(result) else math.inf


def r2_score(
    y_true: Iterable[Real],
    y_pred: Iterable[Real],
    *,
    force_finite: bool = True,
) -> float:
    """Return R-squared; at least two samples are required.

    Variance and residual sums are scale-normalized to avoid intermediate
    overflow. For constant targets, the finite policy returns 1.0 for perfect
    predictions and 0.0 otherwise. Disable it to receive NaN or -infinity.
    """
    if type(force_finite) is not bool:
        raise ValueError("force_finite must be a boolean")
    actual, predicted = _paired_values(y_true, y_pred)
    if len(actual) < 2:
        raise ValueError("r2_score requires at least two samples")

    actual_scale = max(abs(value) for value in actual)
    if actual_scale == 0.0:
        perfect = all(a == p for a, p in zip(actual, predicted))
        if force_finite:
            return 1.0 if perfect else 0.0
        return math.nan if perfect else -math.inf

    normalized_actual = [value / actual_scale for value in actual]
    average = math.fsum(normalized_actual) / len(normalized_actual)
    total_sum = math.fsum((value - average) ** 2 for value in normalized_actual)
    if total_sum == 0.0:
        perfect = all(a == p for a, p in zip(actual, predicted))
        if force_finite:
            return 1.0 if perfect else 0.0
        return math.nan if perfect else -math.inf

    residual_scale = max(
        max(abs(a), abs(p)) for a, p in zip(actual, predicted)
    )
    if residual_scale == 0.0:
        residual_sum = 0.0
    else:
        residual_sum = math.fsum(
            (a / residual_scale - p / residual_scale) ** 2
            for a, p in zip(actual, predicted)
        )
    ratio = _scaled_square_ratio(
        residual_scale, actual_scale, residual_sum, total_sum
    )
    return 1.0 - ratio


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


def mean_squared_log_error(y_true: Iterable[Real], y_pred: Iterable[Real]) -> float:
    """Return mean squared error after applying log1p to non-negative values."""
    actual, predicted = _paired_values(y_true, y_pred)
    if any(value < 0 for value in actual + predicted):
        raise ValueError("mean squared log error requires non-negative values")
    return math.fsum(
        (math.log1p(target) - math.log1p(estimate)) ** 2
        for target, estimate in zip(actual, predicted)
    ) / len(actual)


def root_mean_squared_log_error(y_true: Iterable[Real], y_pred: Iterable[Real]) -> float:
    """Return the square root of mean squared logarithmic error."""
    return math.sqrt(mean_squared_log_error(y_true, y_pred))
