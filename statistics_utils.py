"""Descriptive statistics for finite numeric iterables."""

import math
import sys
from itertools import zip_longest
from numbers import Real
from typing import Iterable


def _values(values: Iterable[Real], name: str) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must contain real numbers")
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError):
            raise ValueError(f"{name} must contain finite numbers") from None
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain finite numbers")
        result.append(number)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _scaled_centered(
    numbers: list[float],
) -> tuple[float, list[float], float, float]:
    """Return normalized values, their mean, and centered sum of squares.

    Scaling before aggregation keeps intermediate sums and squared deviations
    within a small finite range, even when the original values are near the
    largest representable float.
    """
    scale = max(abs(value) for value in numbers)
    if scale == 0.0:
        return 0.0, [0.0] * len(numbers), 0.0, 0.0
    normalized = [value / scale for value in numbers]
    average = math.fsum(normalized) / len(normalized)
    average = min(1.0, max(-1.0, average))
    sum_squares = math.fsum((value - average) ** 2 for value in normalized)
    return scale, normalized, average, sum_squares


def _scaled_mean(numbers: list[float]) -> float:
    scale, _, average, _ = _scaled_centered(numbers)
    return scale * average


def _scaled_standard_deviation(numbers: list[float], denominator: int) -> float:
    scale, _, _, sum_squares = _scaled_centered(numbers)
    if scale == 0.0:
        return 0.0
    result = scale * math.sqrt(sum_squares / denominator)
    if not math.isfinite(result):
        raise ValueError("standard deviation exceeds finite float range")
    return result


def _finite_product(*factors: float, name: str) -> float:
    """Multiply finite factors without overflowing an avoidable intermediate."""
    sign = 1.0
    mantissa = 1.0
    exponent = 0
    for factor in factors:
        if factor == 0.0:
            return 0.0
        if factor < 0:
            sign = -sign
        part, power = math.frexp(abs(factor))
        mantissa *= part
        exponent += power
    try:
        result = math.ldexp(mantissa, exponent) * sign
    except OverflowError:
        raise ValueError(f"{name} exceeds finite float range") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} exceeds finite float range")
    return result


def mean(values: Iterable[Real]) -> float:
    """Return the arithmetic mean without overflowing a representable result."""
    return _scaled_mean(_values(values, "values"))


def median(values: Iterable[Real]) -> float:
    """Return the median, averaging the middle pair for even-sized inputs."""
    numbers = sorted(_values(values, "values"))
    middle = len(numbers) // 2
    if len(numbers) % 2:
        return numbers[middle]
    return numbers[middle - 1] / 2.0 + numbers[middle] / 2.0


def variance(values: Iterable[Real], *, sample: bool = True) -> float:
    """Return sample variance by default, or population variance if requested."""
    if type(sample) is not bool:
        raise ValueError("sample must be a boolean")
    numbers = _values(values, "values")
    denominator = len(numbers) - 1 if sample else len(numbers)
    if denominator <= 0:
        raise ValueError("sample variance requires at least two values")
    deviation = _scaled_standard_deviation(numbers, denominator)
    if deviation > math.sqrt(sys.float_info.max):
        raise ValueError("variance exceeds finite float range")
    return deviation * deviation


def standard_deviation(values: Iterable[Real], *, sample: bool = True) -> float:
    """Return the sample or population standard deviation."""
    if type(sample) is not bool:
        raise ValueError("sample must be a boolean")
    numbers = _values(values, "values")
    denominator = len(numbers) - 1 if sample else len(numbers)
    if denominator <= 0:
        raise ValueError("sample variance requires at least two values")
    return _scaled_standard_deviation(numbers, denominator)


def covariance(first: Iterable[Real], second: Iterable[Real], *, sample: bool = True) -> float:
    """Return sample or population covariance for paired observations."""
    if type(sample) is not bool:
        raise ValueError("sample must be a boolean")
    missing = object()
    left: list[float] = []
    right: list[float] = []
    for x_value, y_value in zip_longest(first, second, fillvalue=missing):
        if x_value is missing or y_value is missing:
            raise ValueError("inputs must have equal lengths")
        left.append(_values([x_value], "first")[0])
        right.append(_values([y_value], "second")[0])
    if not left:
        raise ValueError("inputs must not be empty")
    denominator = len(left) - 1 if sample else len(left)
    if denominator <= 0:
        raise ValueError("sample covariance requires at least two pairs")
    x_scale, x_normalized, x_mean, _ = _scaled_centered(left)
    y_scale, y_normalized, y_mean, _ = _scaled_centered(right)
    if x_scale == 0.0 or y_scale == 0.0:
        return 0.0
    normalized_covariance = math.fsum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_normalized, y_normalized)
    ) / denominator
    return _finite_product(
        x_scale, y_scale, normalized_covariance, name="covariance"
    )


def pearson_correlation(first: Iterable[Real], second: Iterable[Real]) -> float:
    """Return Pearson's correlation coefficient for paired observations."""
    left = _values(first, "first")
    right = _values(second, "second")
    if len(left) != len(right):
        raise ValueError("inputs must have equal lengths")
    if len(left) < 2:
        raise ValueError("at least two pairs are required")
    _, x_normalized, x_mean, x_ss = _scaled_centered(left)
    _, y_normalized, y_mean, y_ss = _scaled_centered(right)
    if x_ss == 0.0 or y_ss == 0.0:
        raise ValueError("correlation is undefined for a constant input")
    cross = math.fsum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_normalized, y_normalized)
    )
    correlation = cross / (math.sqrt(x_ss) * math.sqrt(y_ss))
    return min(1.0, max(-1.0, correlation))


def quantile(values: Iterable[Real], q: Real) -> float:
    """Return the q quantile using linear interpolation between sorted values.

    q is a fraction in [0, 1]. The interpolation uses position (n - 1) * q.
    """
    if isinstance(q, bool) or not isinstance(q, Real):
        raise ValueError("q must be a real number between 0 and 1")
    try:
        fraction = float(q)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("q must be a finite number between 0 and 1") from None
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("q must be a finite number between 0 and 1")
    numbers = sorted(_values(values, "values"))
    position = (len(numbers) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(numbers) - 1)
    weight = position - lower
    return math.fsum((numbers[lower] * (1.0 - weight), numbers[upper] * weight))
