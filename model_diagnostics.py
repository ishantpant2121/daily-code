"""Learning- and validation-curve diagnostics for small estimators."""

import math
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable

from data_splitting import k_fold_indices, stratified_k_fold_indices
from model_selection import _feature_rows, _mean_and_std, _score_estimator, cross_validate


def _train_size_counts(
    train_sizes: Iterable[int | float],
    maximum: int,
) -> list[int]:
    try:
        requested = list(train_sizes)
    except TypeError:
        raise ValueError("train_sizes must be an iterable of positive sizes or fractions") from None
    if not requested:
        raise ValueError("train_sizes must contain at least one size")
    counts = []
    for value in requested:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("train sizes must be positive integers or fractions in (0, 1]")
        if isinstance(value, int):
            count = value
        else:
            try:
                fraction = float(value)
            except (OverflowError, TypeError, ValueError):
                raise ValueError("fractional train sizes must be finite and in (0, 1]") from None
            if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
                raise ValueError("fractional train sizes must be finite and in (0, 1]")
            count = math.ceil(fraction * maximum)
        if not 1 <= count <= maximum:
            raise ValueError("each train size must fit within the smallest training fold")
        counts.append(count)
    if counts != sorted(set(counts)):
        raise ValueError("train sizes must be strictly increasing after conversion")
    return counts


def learning_curve(
    estimator: Any,
    features: Iterable[Iterable[Any]],
    labels: Iterable[Any],
    *,
    train_sizes: Iterable[int | float] = (0.25, 0.5, 0.75, 1.0),
    cv: int = 5,
    scoring: str | Callable[..., float] | None = None,
    stratified: bool = False,
    shuffle: bool = False,
    random_state: int | None = None,
) -> dict[str, list[Any]]:
    """Evaluate training and validation scores at increasing training sizes.

    Every fold uses the same absolute training sizes, bounded by the smallest
    training partition. Smaller subsets are prefixes of each fold's training
    indices. When the input rows are ordered, use shuffled folds or pre-shuffle
    the rows to avoid systematically selecting an unrepresentative prefix.
    """
    matrix = _feature_rows(features)
    try:
        target = list(labels)
    except TypeError:
        raise ValueError("labels must be an iterable of targets") from None
    sample_count = len(matrix)
    if len(target) != sample_count:
        raise ValueError("features and labels must have equal sample counts")
    if type(cv) is not int or not 2 <= cv <= sample_count:
        raise ValueError("cv must be an integer between 2 and the sample count")
    if type(stratified) is not bool or type(shuffle) is not bool:
        raise ValueError("stratified and shuffle must be booleans")
    if random_state is not None and type(random_state) is not int:
        raise ValueError("random_state must be an integer or None")
    if not callable(getattr(estimator, "fit", None)):
        raise ValueError("estimator must implement fit")
    folds = (
        stratified_k_fold_indices(target, n_splits=cv, shuffle=shuffle, random_state=random_state)
        if stratified
        else k_fold_indices(sample_count, n_splits=cv, shuffle=shuffle, random_state=random_state)
    )
    sizes = _train_size_counts(train_sizes, min(len(train) for train, _ in folds))
    train_scores: list[list[float]] = [[] for _ in sizes]
    test_scores: list[list[float]] = [[] for _ in sizes]

    for train_indices, validation_indices in folds:
        validation_x = [matrix[index] for index in validation_indices]
        validation_y = [target[index] for index in validation_indices]
        for size_index, size in enumerate(sizes):
            selected = train_indices[:size]
            train_x = [matrix[index] for index in selected]
            train_y = [target[index] for index in selected]
            model = deepcopy(estimator)
            model.fit(train_x, train_y)
            train_scores[size_index].append(_score_estimator(model, train_x, train_y, scoring))
            test_scores[size_index].append(
                _score_estimator(model, validation_x, validation_y, scoring)
            )
    return {
        "train_sizes": sizes,
        "train_scores": train_scores,
        "test_scores": test_scores,
    }


def _set_parameter(estimator: Any, name: str, value: Any) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("param_name must be a non-empty string")
    if "__" in name:
        step_name, attribute = name.split("__", 1)
        named_steps = getattr(estimator, "named_steps", None)
        if not isinstance(named_steps, Mapping) or not step_name or step_name not in named_steps:
            raise ValueError(f"unknown pipeline step in parameter {name!r}")
        target = named_steps[step_name]
    else:
        attribute = name
        target = estimator
    if not attribute or not hasattr(target, attribute):
        raise ValueError(f"unknown estimator parameter {name!r}")
    setattr(target, attribute, value)


def validation_curve(
    estimator: Any,
    features: Iterable[Iterable[Any]],
    labels: Iterable[Any],
    param_name: str,
    param_range: Iterable[Any],
    *,
    cv: int = 5,
    scoring: str | Callable[..., float] | None = None,
    stratified: bool = False,
    shuffle: bool = False,
    random_state: int | None = None,
) -> dict[str, list[Any]]:
    """Compare train and validation scores across values of one parameter.

    Each parameter value receives a fresh estimator copy and identical
    cross-validation settings. Pipeline parameters use ``step__attribute``
    notation. The original estimator is never mutated.
    """
    try:
        values = list(param_range)
        matrix = _feature_rows(features)
        target = list(labels)
    except TypeError:
        raise ValueError("parameter values, features, and labels must be iterable") from None
    if not values:
        raise ValueError("param_range must contain at least one value")
    if len(matrix) != len(target):
        raise ValueError("features and labels must have equal sample counts")
    if not callable(getattr(estimator, "fit", None)):
        raise ValueError("estimator must implement fit")

    train_scores = []
    test_scores = []
    for value in values:
        candidate = deepcopy(estimator)
        _set_parameter(candidate, param_name, value)
        result = cross_validate(
            candidate,
            matrix,
            target,
            cv=cv,
            scoring=scoring,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
            return_train_score=True,
        )
        train_scores.append(result["train_score"])
        test_scores.append(result["test_score"])
    return {
        "param_range": values,
        "train_scores": train_scores,
        "test_scores": test_scores,
    }


@dataclass(frozen=True)
class BootstrapScore:
    """Bootstrap score distribution and percentile confidence interval."""

    scores: list[float]
    mean_score: float
    lower_bound: float
    upper_bound: float
    confidence_level: float


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_score(
    estimator: Any,
    features: Iterable[Iterable[Any]],
    labels: Iterable[Any],
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    scoring: str | Callable[..., float] | None = None,
    random_state: int | None = None,
) -> BootstrapScore:
    """Estimate uncertainty in a fitted model's score by paired resampling.

    Each replicate samples rows with replacement from the supplied evaluation
    dataset and scores the already-fitted estimator. The estimator is not
    refitted. Use a held-out dataset for a generalization estimate; this
    interval quantifies sampling variability conditional on that fitted model.
    """
    import random

    if type(n_bootstrap) is not int or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, Real):
        raise ValueError("confidence_level must be a finite number between zero and one")
    try:
        level = float(confidence_level)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("confidence_level must be a finite number between zero and one") from None
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be a finite number between zero and one")
    if random_state is not None and type(random_state) is not int:
        raise ValueError("random_state must be an integer or None")
    if scoring is not None and not callable(scoring) and scoring not in (
        "accuracy",
        "balanced_accuracy",
    ):
        raise ValueError("scoring must be None, a supported name, or a callable")
    matrix = _feature_rows(features)
    try:
        target = list(labels)
    except TypeError:
        raise ValueError("labels must be an iterable of targets") from None
    if len(matrix) != len(target):
        raise ValueError("features and labels must have equal sample counts")

    generator = random.Random(random_state)
    scores = []
    sample_count = len(matrix)
    for _ in range(n_bootstrap):
        indices = [generator.randrange(sample_count) for _ in range(sample_count)]
        sample_x = [matrix[index] for index in indices]
        sample_y = [target[index] for index in indices]
        scores.append(_score_estimator(estimator, sample_x, sample_y, scoring))

    ordered = sorted(scores)
    tail = (1.0 - level) / 2.0
    return BootstrapScore(
        scores=scores,
        mean_score=_mean_and_std(scores)[0],
        lower_bound=_percentile(ordered, tail),
        upper_bound=_percentile(ordered, 1.0 - tail),
        confidence_level=level,
    )
