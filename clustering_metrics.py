"""Dependency-free internal validation metrics for clustering results."""

import math
from collections.abc import Iterable
from numbers import Real
from typing import Any


def _matrix(values: Iterable[Iterable[Real]]) -> list[list[float]]:
    try:
        rows = [list(row) for row in values]
    except TypeError:
        raise ValueError("features must be an iterable of rows") from None
    if not rows or not rows[0]:
        raise ValueError("features must contain at least one row and one column")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all feature rows must have equal lengths")
    converted = []
    for row in rows:
        current = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("features must contain finite real numbers")
            try:
                number = float(value)
            except (OverflowError, TypeError, ValueError):
                raise ValueError("features must contain finite real numbers") from None
            if not math.isfinite(number):
                raise ValueError("features must contain finite real numbers")
            current.append(number)
        converted.append(current)
    return converted


def _cluster_groups(labels: Iterable[Any], sample_count: int) -> tuple[list[Any], dict[Any, list[int]]]:
    try:
        values = list(labels)
    except TypeError:
        raise ValueError("labels must be an iterable of hashable values") from None
    if len(values) != sample_count:
        raise ValueError("features and labels must have equal sample counts")
    groups: dict[Any, list[int]] = {}
    try:
        for index, label in enumerate(values):
            groups.setdefault(label, []).append(index)
    except TypeError:
        raise ValueError("cluster labels must be hashable") from None
    if not 2 <= len(groups) <= sample_count - 1:
        raise ValueError("the number of clusters must be between 2 and n_samples - 1")
    return values, groups


def _distance(left: list[float], right: list[float]) -> float:
    value = math.dist(left, right)
    if not math.isfinite(value):
        raise ValueError("feature distance exceeds finite float range")
    return value


def silhouette_samples(
    features: Iterable[Iterable[Real]],
    labels: Iterable[Any],
) -> list[float]:
    """Return one silhouette coefficient per sample using Euclidean distance.

    Singleton clusters receive a coefficient of zero. The implementation uses
    O(n^2) distance work and O(n) auxiliary memory; it does not materialize a
    full pairwise-distance matrix.
    """
    matrix = _matrix(features)
    _, groups = _cluster_groups(labels, len(matrix))
    sample_scores = [0.0] * len(matrix)
    cluster_items = list(groups.items())

    for label, indices in cluster_items:
        for index in indices:
            if len(indices) == 1:
                sample_scores[index] = 0.0
                continue
            within = math.fsum(
                _distance(matrix[index], matrix[other]) / (len(indices) - 1)
                for other in indices
                if other != index
            )
            nearest = min(
                math.fsum(
                    _distance(matrix[index], matrix[other]) / len(other_indices)
                    for other in other_indices
                )
                for other_label, other_indices in cluster_items
                if other_label != label
            )
            denominator = max(within, nearest)
            sample_scores[index] = (
                0.0 if denominator == 0.0 else (nearest - within) / denominator
            )
    return sample_scores


def silhouette_score(
    features: Iterable[Iterable[Real]],
    labels: Iterable[Any],
) -> float:
    """Return the mean silhouette coefficient over all samples."""
    scores = silhouette_samples(features, labels)
    return math.fsum(scores) / len(scores)


def adjusted_rand_score(
    labels_true: Iterable[Any],
    labels_pred: Iterable[Any],
) -> float:
    """Return the adjusted Rand index for two cluster assignments.

    The score is invariant to cluster-label renaming, equals 1 for identical
    partitions, and is adjusted for agreement expected by chance.
    """
    try:
        actual = list(labels_true)
        predicted = list(labels_pred)
    except TypeError:
        raise ValueError("labels must be iterable") from None
    if not actual or len(actual) != len(predicted):
        raise ValueError("label arrays must be non-empty and have equal lengths")
    if len(actual) < 2:
        raise ValueError("at least two samples are required")

    actual_groups: dict[Any, int] = {}
    predicted_groups: dict[Any, int] = {}
    contingency: dict[tuple[Any, Any], int] = {}
    try:
        for left, right in zip(actual, predicted):
            actual_groups[left] = actual_groups.get(left, 0) + 1
            predicted_groups[right] = predicted_groups.get(right, 0) + 1
            key = (left, right)
            contingency[key] = contingency.get(key, 0) + 1
    except TypeError:
        raise ValueError("cluster labels must be hashable") from None

    def pairs(count: int) -> int:
        return count * (count - 1) // 2

    index = sum(pairs(count) for count in contingency.values())
    sum_actual = sum(pairs(count) for count in actual_groups.values())
    sum_predicted = sum(pairs(count) for count in predicted_groups.values())
    total_pairs = pairs(len(actual))
    expected = (sum_actual * sum_predicted) / total_pairs
    maximum = (sum_actual + sum_predicted) / 2
    denominator = maximum - expected
    if denominator == 0:
        return 1.0
    return (index - expected) / denominator


def davies_bouldin_score(
    features: Iterable[Iterable[Real]],
    labels: Iterable[Any],
) -> float:
    """Return the mean worst-case ratio of within- to between-cluster spread.

    Lower values indicate more compact, well-separated clusters. Coincident
    centroids with non-zero spread produce positive infinity, reflecting an
    undefined separation rather than silently returning a misleading finite
    score.
    """
    matrix = _matrix(features)
    _, groups = _cluster_groups(labels, len(matrix))
    centers = []
    scatters = []
    for indices in groups.values():
        center = [
            math.fsum(matrix[index][column] / len(indices) for index in indices)
            for column in range(len(matrix[0]))
        ]
        centers.append(center)
        scatter = math.fsum(
            _distance(matrix[index], center) / len(indices)
            for index in indices
        )
        scatters.append(scatter)

    cluster_ratios = []
    for left in range(len(centers)):
        worst = 0.0
        for right in range(len(centers)):
            if left == right:
                continue
            separation = _distance(centers[left], centers[right])
            if separation == 0.0:
                ratio = (
                    math.inf
                    if scatters[left] > 0.0 or scatters[right] > 0.0
                    else 0.0
                )
            else:
                ratio = (
                    scatters[left] / separation
                    + scatters[right] / separation
                )
            worst = max(worst, ratio)
        cluster_ratios.append(worst)
    return (
        math.fsum(value / len(cluster_ratios) for value in cluster_ratios)
        if all(math.isfinite(value) for value in cluster_ratios)
        else math.inf
    )
