"""Paired bootstrap and permutation tests for comparing fixed model predictions."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Iterable

Scorer = Callable[[list[Any], list[Any]], Real]


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Bootstrap distribution and percentile interval for score A minus score B."""

    observed_difference: float
    mean_difference: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    n_bootstrap: int
    bootstrap_differences: list[float]
    scoring: str


@dataclass(frozen=True)
class PairedPermutationResult:
    """Randomization-test distribution for the paired score difference."""

    observed_difference: float
    p_value: float
    n_permutations: int
    permutation_differences: list[float]
    scoring: str


def _materialize(values: Iterable[Any], name: str) -> list[Any]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of sample values")
    try:
        result = list(values)
    except TypeError:
        raise ValueError(f"{name} must be an iterable") from None
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _paired_inputs(
    y_true: Iterable[Any],
    predictions_a: Iterable[Any],
    predictions_b: Iterable[Any],
) -> tuple[list[Any], list[Any], list[Any]]:
    actual = _materialize(y_true, "y_true")
    first = _materialize(predictions_a, "predictions_a")
    second = _materialize(predictions_b, "predictions_b")
    if len(actual) != len(first) or len(actual) != len(second):
        raise ValueError("y_true and both prediction arrays must have equal lengths")
    return actual, first, second


def _resolve_scorer(scoring: str | Scorer) -> tuple[Scorer, str]:
    if callable(scoring):
        return scoring, getattr(scoring, "__name__", "callable")
    if not isinstance(scoring, str):
        raise ValueError("scoring must be a supported name or callable")
    if scoring == "accuracy":
        def accuracy(actual: list[Any], predicted: list[Any]) -> float:
            return sum(a == p for a, p in zip(actual, predicted)) / len(actual)
        return accuracy, scoring
    if scoring == "balanced_accuracy":
        from classification_metrics import balanced_accuracy_score
        return balanced_accuracy_score, scoring
    if scoring == "r2":
        from regression_metrics import r2_score
        return r2_score, scoring
    if scoring == "neg_mean_squared_error":
        from regression_metrics import mean_squared_error
        return lambda actual, predicted: -mean_squared_error(actual, predicted), scoring
    if scoring == "neg_mean_absolute_error":
        from regression_metrics import mean_absolute_error
        return lambda actual, predicted: -mean_absolute_error(actual, predicted), scoring
    if scoring == "neg_root_mean_squared_error":
        from regression_metrics import root_mean_squared_error
        return lambda actual, predicted: -root_mean_squared_error(actual, predicted), scoring
    raise ValueError("scoring must be one of the supported names or a callable")


def _score(scorer: Scorer, actual: list[Any], predicted: list[Any]) -> float:
    value = scorer(actual, predicted)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("scoring must return a real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("scoring must return a finite real number") from None
    if not math.isfinite(result):
        raise ValueError("scoring must return a finite real number")
    return result


def _difference(
    scorer: Scorer,
    actual: list[Any],
    first: list[Any],
    second: list[Any],
) -> float:
    difference = _score(scorer, actual, first) - _score(scorer, actual, second)
    if not math.isfinite(difference):
        raise ValueError("score difference exceeds finite float range")
    return difference


def _stable_mean(values: list[float]) -> float:
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    normalized_mean = math.fsum(value / scale for value in values) / len(values)
    normalized_mean = min(1.0, max(-1.0, normalized_mean))
    return scale * normalized_mean


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _validate_random_state(random_state: int | None) -> None:
    if random_state is not None and type(random_state) is not int:
        raise ValueError("random_state must be an integer or None")


def paired_bootstrap_score_difference(
    y_true: Iterable[Any],
    predictions_a: Iterable[Any],
    predictions_b: Iterable[Any],
    *,
    scoring: str | Scorer = "accuracy",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> PairedBootstrapResult:
    """Estimate uncertainty in score A minus score B by paired resampling.

    Samples are resampled together so each bootstrap replicate preserves the
    pairing between ground truth and both models' predictions. Predictions are
    fixed; this function does not retrain either estimator.
    """
    if type(n_bootstrap) is not int or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, Real):
        raise ValueError("confidence_level must be a finite number between zero and one")
    try:
        level = float(confidence_level)
    except (OverflowError, TypeTypeError, ValueError):
        raise ValueError("confidence_level must be a finite number between zero and one") from None
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be a finite number between zero and one")
    _validate_random_state(random_state)
    actual, first, second = _paired_inputs(y_true, predictions_a, predictions_b)
    scorer, scoring_name = _resolve_scorer(scoring)
    observed = _difference(scorer, actual, first, second)
    generator = random.Random(random_state)
    count = len(actual)
    differences = []
    for _ in range(n_bootstrap):
        indices = [generator.randrange(count) for _ in range(count)]
        sampled_actual = [actual[index] for index in indices]
        sampled_first = [first[index] for index in indices]
        sampled_second = [second[index] for index in indices]
        differences.append(_difference(scorer, sampled_actual, sampled_first, sampled_second))
    ordered = sorted(differences)
    tail = (1.0 - level) / 2.0
    return PairedBootstrapResult(
        observed_difference=observed,
        mean_difference=_stable_mean(differences),
        lower_bound=_percentile(ordered, tail),
        upper_bound=_percentile(ordered, 1.0 - tail),
        confidence_level=level,
        n_bootstrap=n_bootstrap,
        bootstrap_differences=differences,
        scoring=scoring_name,
    )


def paired_permutation_test(
    y_true: Iterable[Any],
    predictions_a: Iterable[Any],
    predictions_b: Iterable[Any],
    *,
    scoring: str | Scorer = "accuracy",
    n_permutations: int = 1000,
    random_state: int | None = None,
) -> PairedPermutationResult:
    """Compare two fixed models using paired prediction swaps.

    For every permutation, independently swap the two models' predictions for
    each sample with probability one-half. The two-sided p-value uses the
    absolute score difference and the plus-one correction. This test assumes
    exchangeability of the models' predictions under the null hypothesis.
    """
    if type(n_permutations) is not int or n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer")
    _validate_random_state(random_state)
    actual, first, second = _paired_inputs(y_true, predictions_a, predictions_b)
    scorer, scoring_name = _resolve_scorer(scoring)
    observed = _difference(scorer, actual, first, second)
    generator = random.Random(random_state)
    differences = []
    for _ in range(n_permutations):
        permuted_first = []
        permuted_second = []
        for a, b in zip(first, second):
            if generator.random() < 0.5:
                permuted_first.append(b)
                permuted_second.append(a)
            else:
                permuted_first.append(a)
                permuted_second.append(b)
        differences.append(_difference(scorer, actual, permuted_first, permuted_second))
    extreme = sum(abs(value) >= abs(observed) for value in differences)
    p_value = (extreme + 1) / (n_permutations + 1)
    return PairedPermutationResult(
        observed_difference=observed,
        p_value=p_value,
        n_permutations=n_permutations,
        permutation_differences=differences,
        scoring=scoring_name,
    )
