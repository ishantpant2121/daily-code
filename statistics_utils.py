"""Descriptive statistics for finite numeric iterables."""

import math
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


def mean(values: Iterable[Real]) -> float:
    """Return the arithmetic mean."""
    numbers = _values(values, "values")
    return math.fsum(numbers) / len(numbers)


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
    average = math.fsum(numbers) / len(numbers)
    return math.fsum((value - average) ** 2 for value in numbers) / denominator


def standard_deviation(values: Iterable[Real], *, sample: bool = True) -> float:
    """Return the square root of variance."""
    return math.sqrt(variance(values, sample=sample))


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
    x_mean = math.fsum(left) / len(left)
    y_mean = math.fsum(right) / len(right)
    return math.fsum((x - x_mean) * (y - y_mean) for x, y in zip(left, right)) / denominator


def pearson_correlation(first: Iterable[Real], second: Iterable[Real]) -> float:
    """Return Pearson's correlation coefficient for paired observations."""
    left = _values(first, "first")
    right = _values(second, "second")
    if len(left) != len(right):
        raise ValueError("inputs must have equal lengths")
    if len(left) < 2:
        raise ValueError("at least two pairs are required")
    x_mean = math.fsum(left) / len(left)
    y_mean = math.fsum(right) / len(right)
    x_ss = math.fsum((x - x_mean) ** 2 for x in left)
    y_ss = math.fsum((y - y_mean) ** 2 for y in right)
    if x_ss == 0 or y_ss == 0:
        raise ValueError("correlation is undefined for a constant input")
    cross = math.fsum((x - x_mean) * (y - y_mean) for x, y in zip(left, right))
    return cross / math.sqrt(x_ss * y_ss)


def quantile(values: Iterable[Real], q: Real) -> float:
    """Return the q quantile using linear interpolation between sorted values.

    q is a fraction in [0, 1]. The interpolation uses position (n - 1) * q.
    """
    if isinstance(q, bool) or not isinstance(q, Real):
        raise ValueError("q must be a real number between 0 and 1")
    try:
        fraction = float(q)
    except (OverflowError, ValueError):
        raise ValueError("q must be a finite number between 0 and 1") from None
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("q must be a finite number between 0 and 1")
    numbers = sorted(_values(values, "values"))
    position = (len(numbers) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(numbers) - 1)
    weight = position - lower
    return math.fsum((numbers[lower] * (1.0 - weight), numbers[upper] * weight))
