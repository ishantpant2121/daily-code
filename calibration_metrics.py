"""Dependency-free probability calibration metrics for binary classifiers."""

import math
from numbers import Real
from typing import Any, Iterable


def _binary_inputs(
    y_true: Iterable[Any],
    y_prob: Iterable[Real],
    n_bins: int,
    positive_label: Any,
) -> tuple[list[bool], list[float]]:
    if type(n_bins) is not int or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    try:
        labels = list(y_true)
        probabilities = list(y_prob)
    except TypeError:
        raise ValueError("labels and probabilities must be iterable") from None
    if not labels or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be non-empty and have equal lengths")
    actual = [label == positive_label for label in labels]
    converted = []
    for value in probabilities:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("probabilities must be finite numbers between zero and one")
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("probabilities must be finite numbers between zero and one") from None
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError("probabilities must be finite numbers between zero and one")
        converted.append(number)
    return actual, converted


def calibration_curve(
    y_true: Iterable[Any],
    y_prob: Iterable[Real],
    *,
    n_bins: int = 10,
    positive_label: Any = 1,
) -> dict[str, list[float] | list[int]]:
    """Summarize observed frequency and mean confidence in uniform-width bins.

    Empty bins are omitted. The final bin includes probability 1.0. Returned
    lists are ordered by increasing confidence and include bin counts so callers
    can distinguish well-supported bins from bins with few observations.
    """
    actual, probabilities = _binary_inputs(y_true, y_prob, n_bins, positive_label)
    counts = [0] * n_bins
    probability_sums = [0.0] * n_bins
    positive_sums = [0] * n_bins
    for label, probability in zip(actual, probabilities):
        index = min(int(probability * n_bins), n_bins - 1)
        counts[index] += 1
        probability_sums[index] += probability
        positive_sums[index] += int(label)

    mean_probability = []
    fraction_positive = []
    non_empty_counts = []
    for count, probability_sum, positive_sum in zip(
        counts, probability_sums, positive_sums
    ):
        if count:
            mean_probability.append(probability_sum / count)
            fraction_positive.append(positive_sum / count)
            non_empty_counts.append(count)
    return {
        "prob_true": fraction_positive,
        "prob_pred": mean_probability,
        "bin_counts": non_empty_counts,
    }


def expected_calibration_error(
    y_true: Iterable[Any],
    y_prob: Iterable[Real],
    *,
    n_bins: int = 10,
    positive_label: Any = 1,
) -> float:
    """Return the count-weighted mean absolute calibration gap."""
    curve = calibration_curve(
        y_true, y_prob, n_bins=n_bins, positive_label=positive_label
    )
    counts = curve["bin_counts"]
    gaps = [
        abs(observed - confidence)
        for observed, confidence in zip(curve["prob_true"], curve["prob_pred"])
    ]
    return math.fsum(count * gap for count, gap in zip(counts, gaps)) / sum(counts)


def maximum_calibration_error(
    y_true: Iterable[Any],
    y_prob: Iterable[Real],
    *,
    n_bins: int = 10,
    positive_label: Any = 1,
) -> float:
    """Return the largest absolute calibration gap among non-empty bins."""
    curve = calibration_curve(
        y_true, y_prob, n_bins=n_bins, positive_label=positive_label
    )
    return max(
        abs(observed - confidence)
        for observed, confidence in zip(curve["prob_true"], curve["prob_pred"])
    )
