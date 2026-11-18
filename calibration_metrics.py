"""Dependency-free probability calibration metrics for binary classifiers."""

from bisect import bisect_left
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


def _calibration_bin_indices(
    probabilities: list[float],
    n_bins: int,
    strategy: str,
) -> list[int]:
    """Assign probabilities to uniform-width or tie-preserving quantile bins."""
    if type(n_bins) is not int or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if strategy not in ("uniform", "quantile"):
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    if strategy == "uniform":
        return [min(int(value * n_bins), n_bins - 1) for value in probabilities]

    ordered = sorted(probabilities)
    # Quantile boundaries are observed values. bisect_left keeps equal
    # probabilities together rather than splitting ties across adjacent bins.
    edges = [
        ordered[(index * len(ordered) + n_bins - 1) // n_bins - 1]
        for index in range(1, n_bins)
    ]
    return [bisect_left(edges, value) for value in probabilities]


def calibration_curve(
    y_true: Iterable[Any],
    y_prob: Iterable[Real],
    *,
    n_bins: int = 10,
    positive_label: Any = 1,
    strategy: str = "uniform",
) -> dict[str, list[float] | list[int]]:
    """Summarize observed frequency and mean confidence in probability bins.

    strategy may be "uniform" for equal-width bins or "quantile" for
    approximately equal-frequency bins. Quantile boundaries use observed
    probabilities and keep tied values in the same bin. Empty bins are omitted.
    The final uniform bin includes probability 1.0. Returned lists are ordered
    by increasing confidence and include bin counts.
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
    strategy: str = "uniform",
) -> float:
    """Return the count-weighted mean absolute calibration gap."""
    curve = calibration_curve(
        y_true, y_prob, n_bins=n_bins, positive_label=positive_label,
        strategy=strategy,
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
    strategy: str = "uniform",
) -> float:
    """Return the largest absolute calibration gap among non-empty bins."""
    curve = calibration_curve(
        y_true, y_prob, n_bins=n_bins, positive_label=positive_label,
        strategy=strategy,
    )
    return max(
        abs(observed - confidence)
        for observed, confidence in zip(curve["prob_true"], curve["prob_pred"])
    )


def _multiclass_calibration_inputs(
    y_true: Iterable[Any],
    y_probability: Iterable[Iterable[Real]],
    labels: Iterable[Any],
    n_bins: int,
) -> tuple[list[Any], list[Any], list[list[float]], list[float], list[int]]:
    """Validate class-aligned probability rows and derive top-label outcomes."""
    if type(n_bins) is not int or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    try:
        actual = list(y_true)
        rows = [list(row) for row in y_probability]
        classes = list(labels)
    except TypeError:
        raise ValueError("labels and probabilities must be iterable") from None
    if len(classes) < 2:
        raise ValueError("at least two class labels are required")
    if not actual or len(actual) != len(rows):
        raise ValueError("labels and probabilities must be non-empty and have equal lengths")

    positions: dict[Any, int] = {}
    try:
        for index, label in enumerate(classes):
            if label in positions:
                raise ValueError("class labels must be unique")
            positions[label] = index
        targets = [positions[label] for label in actual]
    except TypeError:
        raise ValueError("class labels must be hashable") from None
    except KeyError:
        raise ValueError("y_true contains a label not present in labels") from None

    normalized_rows = []
    confidences = []
    correct = []
    for target, row in zip(targets, rows):
        if len(row) != len(classes):
            raise ValueError("each probability row must contain one value per class")
        converted = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("probabilities must be finite values in [0, 1]")
            try:
                number = float(value)
            except (OverflowError, TypeError, ValueError):
                raise ValueError("probabilities must be finite values in [0, 1]") from None
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError("probabilities must be finite values in [0, 1]")
            converted.append(number)
        if not math.isclose(math.fsum(converted), 1.0, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("each probability row must sum to one")
        winner = max(range(len(classes)), key=converted.__getitem__)
        confidences.append(converted[winner])
        correct.append(int(winner == target))
        normalized_rows.append(converted)
    return actual, classes, normalized_rows, confidences, correct


def multiclass_calibration_curve(
    y_true: Iterable[Any],
    y_probability: Iterable[Iterable[Real]],
    *,
    labels: Iterable[Any],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, list[float] | list[int]]:
    """Return top-label accuracy and confidence for multiclass probabilities.

    Each row of y_probability must follow the supplied labels order and sum
    to one. For each sample, the maximum class probability is its confidence
    and whether that class is correct is its outcome. strategy supports
    uniform-width or tie-preserving quantile bins. Empty bins are omitted.
    """
    _, _, _, confidences, correct = _multiclass_calibration_inputs(
        y_true, y_probability, labels, n_bins
    )
    indices = _calibration_bin_indices(confidences, n_bins, strategy)
    counts = [0] * n_bins
    confidence_sums = [0.0] * n_bins
    correct_sums = [0] * n_bins
    for index, confidence, hit in zip(indices, confidences, correct):
        counts[index] += 1
        confidence_sums[index] += confidence
        correct_sums[index] += hit
    return {
        "prob_true": [
            value / count for value, count in zip(correct_sums, counts) if count
        ],
        "prob_pred": [
            value / count for value, count in zip(confidence_sums, counts) if count
        ],
        "bin_counts": [count for count in counts if count],
    }


def _multiclass_calibration_gaps(
    y_true: Iterable[Any],
    y_probability: Iterable[Iterable[Real]],
    *,
    labels: Iterable[Any],
    n_bins: int,
    strategy: str,
) -> tuple[list[int], list[float]]:
    curve = multiclass_calibration_curve(
        y_true, y_probability, labels=labels, n_bins=n_bins, strategy=strategy
    )
    gaps = [
        observed - confidence
        for observed, confidence in zip(curve["prob_true"], curve["prob_pred"])
    ]
    return curve["bin_counts"], gaps


def multiclass_expected_calibration_error(
    y_true: Iterable[Any],
    y_probability: Iterable[Iterable[Real]],
    *,
    labels: Iterable[Any],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Return count-weighted absolute top-label calibration error (L1)."""
    counts, gaps = _multiclass_calibration_gaps(
        y_true, y_probability, labels=labels, n_bins=n_bins, strategy=strategy
    )
    return math.fsum(count * abs(gap) for count, gap in zip(counts, gaps)) / sum(counts)


def multiclass_maximum_calibration_error(
    y_true: Iterable[Any],
    y_probability: Iterable[Iterable[Real]],
    *,
    labels: Iterable[Any],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Return the largest absolute top-label calibration gap (L-infinity)."""
    _, gaps = _multiclass_calibration_gaps(
        y_true, y_probability, labels=labels, n_bins=n_bins, strategy=strategy
    )
    return max(abs(gap) for gap in gaps)


def multiclass_root_mean_squared_calibration_error(
    y_true: Iterable[Any],
    y_probability: Iterable[Iterable[Real]],
    *,
    labels: Iterable[Any],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Return count-weighted root mean squared top-label calibration error."""
    counts, gaps = _multiclass_calibration_gaps(
        y_true, y_probability, labels=labels, n_bins=n_bins, strategy=strategy
    )
    return math.sqrt(
        math.fsum(count * gap * gap for count, gap in zip(counts, gaps)) / sum(counts)
    )


def classwise_calibration_curves(
    y_true: Iterable[Any],
    y_probability: Iterable[Iterable[Real]],
    *,
    labels: Iterable[Any],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[Any, dict[str, list[float] | list[int]]]:
    """Return one-vs-rest reliability curves for every class.

    The returned mapping preserves the supplied class order. Each curve
    compares that class's probability column with its one-vs-rest outcomes.
    """
    actual, classes, rows, _, _ = _multiclass_calibration_inputs(
        y_true, y_probability, labels, n_bins
    )
    curves = {}
    for column, label in enumerate(classes):
        probabilities = [row[column] for row in rows]
        indices = _calibration_bin_indices(probabilities, n_bins, strategy)
        counts = [0] * n_bins
        probability_sums = [0.0] * n_bins
        positive_sums = [0] * n_bins
        for actual_label, probability, index in zip(actual, probabilities, indices):
            counts[index] += 1
            probability_sums[index] += probability
            positive_sums[index] += int(actual_label == label)
        curves[label] = {
            "prob_true": [
                value / count for value, count in zip(positive_sums, counts) if count
            ],
            "prob_pred": [
                value / count for value, count in zip(probability_sums, counts) if count
            ],
            "bin_counts": [count for count in counts if count],
        }
    return curves
