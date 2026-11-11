"""Model-agnostic feature inspection utilities."""

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from model_selection import _feature_rows, _score_estimator


@dataclass(frozen=True)
class PermutationImportance:
    """Raw and summarized score drops from independently permuting features."""

    importances: list[list[float]]
    importances_mean: list[float]
    importances_std: list[float]
    baseline_score: float


def permutation_importance(
    estimator: Any,
    features: Iterable[Iterable[Any]],
    labels: Iterable[Any],
    *,
    scoring: str | Callable[..., float] | None = None,
    n_repeats: int = 5,
    random_state: int | None = None,
) -> PermutationImportance:
    """Measure score degradation after shuffling one feature at a time.

    ``estimator`` must already be fitted. For every feature, its values are
    shuffled independently ``n_repeats`` times while all other columns remain
    unchanged. Importance is ``baseline_score - permuted_score``; negative
    values are possible when a permutation improves the score.

    A held-out validation set is recommended. Correlated features can mask one
    another's importance, so this statistic describes reliance of this fitted
    model on the supplied dataset, not intrinsic feature value.
    """
    if type(n_repeats) is not int or n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer")
    if random_state is not None and type(random_state) is not int:
        raise ValueError("random_state must be an integer or None")
    matrix = _feature_rows(features)
    try:
        target = list(labels)
    except TypeError:
        raise ValueError("labels must be an iterable of targets") from None
    if len(target) != len(matrix):
        raise ValueError("features and labels must have equal sample counts")
    if scoring is not None and not callable(scoring) and scoring not in (
        "accuracy",
        "balanced_accuracy",
    ):
        raise ValueError("scoring must be None, a supported name, or a callable")

    baseline = _score_estimator(estimator, matrix, target, scoring)
    generator = random.Random(random_state)
    columns = len(matrix[0])
    importances = [[] for _ in range(columns)]

    for column in range(columns):
        for _ in range(n_repeats):
            permuted = [row[:] for row in matrix]
            values = [row[column] for row in matrix]
            generator.shuffle(values)
            for row, value in zip(permuted, values):
                row[column] = value
            permuted_score = _score_estimator(estimator, permuted, target, scoring)
            importances[column].append(baseline - permuted_score)

    means = [math.fsum(values) / len(values) for values in importances]
    standard_deviations = []
    for values, mean in zip(importances, means):
        variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
        standard_deviations.append(math.sqrt(variance))

    return PermutationImportance(
        importances=importances,
        importances_mean=means,
        importances_std=standard_deviations,
        baseline_score=baseline,
    )
