"""Model-agnostic feature inspection utilities."""

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from model_selection import _feature_rows, _mean_and_std, _score_estimator


@dataclass(frozen=True)
class PermutationImportance:
    """Raw and summarized score drops from independently permuting features."""

    importances: list[list[float]]
    importances_mean: list[float]
    importances_std: list[float]
    baseline_score: float
    n_samples_used: int


def permutation_importance(
    estimator: Any,
    features: Iterable[Iterable[Any]],
    labels: Iterable[Any],
    *,
    scoring: str | Callable[..., float] | None = None,
    n_repeats: int = 5,
    random_state: int | None = None,
    max_samples: int | float = 1.0,
) -> PermutationImportance:
    """Measure score degradation after shuffling one feature at a time.

    The estimator must already be fitted. For every feature, its values are
    shuffled independently n_repeats times while all other columns remain
    unchanged. Importance is baseline_score - permuted_score; negative
    values are possible when a permutation improves the score.

    Built-in scorers include classification accuracy metrics and regression
    metrics (r2, neg_mean_squared_error, neg_mean_absolute_error, and
    neg_root_mean_squared_error). A custom scorer receives
    (estimator, features, labels) and must return a finite real number.

    max_samples limits the evaluation rows. An integer selects that many
    rows; a float in (0, 1] selects that fraction, rounded up to at least one
    row. The same deterministic subset is used for the baseline and every
    permutation so score differences remain comparable. n_samples_used
    reports the resulting subset size.

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
        "r2",
        "neg_mean_squared_error",
        "neg_mean_absolute_error",
        "neg_root_mean_squared_error",
    ):
        raise ValueError("scoring must be None, a supported name, or a callable")

    sample_count = len(matrix)
    if isinstance(max_samples, bool) or not isinstance(max_samples, (int, float)):
        raise ValueError("max_samples must be a positive integer or a fraction in (0, 1]")
    if isinstance(max_samples, int):
        if not 1 <= max_samples <= sample_count:
            raise ValueError("integer max_samples must be between one and the sample count")
        n_samples_used = max_samples
    else:
        if not math.isfinite(max_samples) or not 0.0 < max_samples <= 1.0:
            raise ValueError("fractional max_samples must be finite and in (0, 1]")
        n_samples_used = min(sample_count, max(1, math.ceil(max_samples * sample_count)))

    generator = random.Random(random_state)
    sample_indices = (
        list(range(sample_count))
        if n_samples_used == sample_count
        else generator.sample(range(sample_count), n_samples_used)
    )
    evaluation_matrix = [matrix[index][:] for index in sample_indices]
    evaluation_target = [target[index] for index in sample_indices]

    baseline = _score_estimator(
        estimator, evaluation_matrix, evaluation_target, scoring
    )
    columns = len(matrix[0])
    importances = [[] for _ in range(columns)]

    for column in range(columns):
        for _ in range(n_repeats):
            permuted = [row[:] for row in evaluation_matrix]
            values = [row[column] for row in evaluation_matrix]
            generator.shuffle(values)
            for row, value in zip(permuted, values):
                row[column] = value
            permuted_score = _score_estimator(
                estimator, permuted, evaluation_target, scoring
            )
            importance = baseline - permuted_score
            if not math.isfinite(importance):
                raise ValueError("permutation importance must be a finite real number")
            importances[column].append(importance)

    aggregates = [_mean_and_std(values) for values in importances]
    means = [mean for mean, _ in aggregates]
    standard_deviations = [std for _, std in aggregates]

    return PermutationImportance(
        importances=importances,
        importances_mean=means,
        importances_std=standard_deviations,
        baseline_score=baseline,
        n_samples_used=n_samples_used,
    )
