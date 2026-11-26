"""Diagnostics for evaluating regression prediction intervals.

Coverage and width are reported together: wide intervals can achieve high
coverage without being informative. Optional group summaries expose uneven
coverage across known cohorts; they do not imply conditional coverage.
"""
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Iterable


@dataclass(frozen=True)
class IntervalSummary:
    """Coverage and sharpness statistics for a collection of intervals."""
    count: int
    coverage: float
    mean_width: float
    median_width: float
    mean_interval_score: float
    covered: int
    below: int
    above: int


@dataclass(frozen=True)
class IntervalDiagnostics:
    """Overall interval metrics and optional per-group summaries."""
    overall: IntervalSummary
    by_group: dict[Any, IntervalSummary]


def _real_values(values: Iterable[Real], name: str) -> list[float]:
    try:
        items = list(values)
    except TypeError:
        raise ValueError(f"{name} must be iterable") from None
    if not items:
        raise ValueError(f"{name} must not be empty")
    result = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{name} must contain finite real numbers")
        try:
            value = float(item)
        except (OverflowError, TypeError, ValueError):
            raise ValueError(f"{name} must contain finite real numbers") from None
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain finite real numbers")
        result.append(value)
    return result


def _intervals(values: Iterable[Iterable[Real]]) -> list[tuple[float, float]]:
    try:
        raw = list(values)
    except TypeError:
        raise ValueError("intervals must be iterable") from None
    if not raw:
        raise ValueError("intervals must not be empty")
    result = []
    for pair in raw:
        if isinstance(pair, (str, bytes)):
            raise ValueError("each interval must contain two endpoints")
        try:
            endpoints = list(pair)
        except TypeError:
            raise ValueError("each interval must contain two endpoints") from None
        if len(endpoints) != 2:
            raise ValueError("each interval must contain two endpoints")
        converted = []
        for endpoint in endpoints:
            if isinstance(endpoint, bool) or not isinstance(endpoint, Real):
                raise ValueError("interval endpoints must be real numbers")
            try:
                value = float(endpoint)
            except (OverflowError, TypeError, ValueError):
                raise ValueError("interval endpoints must not be NaN") from None
            if math.isnan(value):
                raise ValueError("interval endpoints must not be NaN")
            converted.append(value)
        lower, upper = converted
        if lower > upper:
            raise ValueError("interval lower endpoint must not exceed upper endpoint")
        result.append((lower, upper))
    return result


def _summary(targets, bounds, alpha: float) -> IntervalSummary:
    below = sum(target < lower for target, (lower, _) in zip(targets, bounds))
    above = sum(target > upper for target, (_, upper) in zip(targets, bounds))
    covered = len(targets) - below - above
    widths = [upper - lower for lower, upper in bounds]
    if any(not math.isfinite(width) for width in widths):
        mean_width = median_width = math.inf
    else:
        # Scale before summing: fsum can overflow even when the mean is
        # representable (for example, two widths near 1e308).
        scale = max(widths)
        mean_width = (0.0 if scale == 0.0 else
                      scale * (math.fsum(width / scale for width in widths) / len(widths)))
        ordered = sorted(widths)
        middle = len(ordered) // 2
        median_width = (ordered[middle] if len(ordered) % 2 else
                        ordered[middle - 1] / 2.0 + ordered[middle] / 2.0)
    scores = []
    for target, (lower, upper), width in zip(targets, bounds, widths):
        penalty = (2.0 / alpha) * (lower - target) if target < lower else 0.0
        penalty += (2.0 / alpha) * (target - upper) if target > upper else 0.0
        scores.append(width + penalty)
    return IntervalSummary(
        count=len(targets), coverage=covered / len(targets),
        mean_width=mean_width, median_width=median_width,
        mean_interval_score=(
            math.inf if any(not math.isfinite(score) for score in scores)
            else (0.0 if max(scores) == 0.0 else
                  max(scores) * (math.fsum(score / max(scores) for score in scores) / len(scores)))
        ),
        covered=covered, below=below, above=above,
    )


def interval_diagnostics(
    targets: Iterable[Real],
    intervals: Iterable[Iterable[Real]],
    *,
    alpha: float = 0.1,
    groups: Iterable[Any] | None = None,
) -> IntervalDiagnostics:
    """Evaluate inclusive intervals using coverage, width, and Winkler score.

    Alpha is the nominal miscoverage rate used by the interval score.
    Optional groups assign each row to a hashable cohort and return independent
    summaries in first-seen order. Group summaries are descriptive and do not
    establish conditional or subgroup coverage guarantees. Use held-out data.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValueError("alpha must be strictly between zero and one")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    actual = _real_values(targets, "targets")
    bounds = _intervals(intervals)
    if len(actual) != len(bounds):
        raise ValueError("targets and intervals must have equal lengths")
    overall = _summary(actual, bounds, alpha)
    if groups is None:
        return IntervalDiagnostics(overall=overall, by_group={})
    try:
        labels = list(groups)
    except TypeError:
        raise ValueError("groups must be iterable") from None
    if len(labels) != len(actual):
        raise ValueError("groups and targets must have equal lengths")
    partitions = {}
    for target, interval, label in zip(actual, bounds, labels):
        try:
            hash(label)
        except TypeError:
            raise ValueError("group labels must be hashable") from None
        if label not in partitions:
            partitions[label] = ([], [])
        partitions[label][0].append(target)
        partitions[label][1].append(interval)
    grouped = {label: _summary(rows, pairs, alpha)
               for label, (rows, pairs) in partitions.items()}
    return IntervalDiagnostics(overall=overall, by_group=grouped)
