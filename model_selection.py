"""Small, deterministic cross-validation helpers for estimator objects."""

import math
from copy import deepcopy
from numbers import Integral, Real
from typing import Any, Callable, Iterable

from classification_metrics import balanced_accuracy_score
from regression_metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
from data_splitting import k_fold_indices, stratified_k_fold_indices


Scorer = Callable[[Any, list[list[Any]], list[Any]], Real]


def _feature_rows(values: Iterable[Iterable[Any]]) -> list[list[Any]]:
    try:
        rows = [list(row) for row in values]
    except TypeError:
        raise ValueError("features must be an iterable of rows") from None
    if not rows or not rows[0]:
        raise ValueError("features must contain at least one row and one column")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all feature rows must have equal lengths")
    return rows


def _cv_splits(
    cv: int | Iterable[Any],
    sample_count: int,
    labels: list[Any],
    *,
    stratified: bool = False,
    shuffle: bool = False,
    random_state: int | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Build or validate reusable train/validation index pairs.

    Integer cv uses the built-in K-fold splitters. Other iterables are treated
    as explicit folds and must contain non-empty, disjoint, in-range indices.
    Explicit folds cannot be combined with the integer-split options.
    """
    if type(stratified) is not bool or type(shuffle) is not bool:
        raise ValueError("stratified and shuffle must be booleans")
    if random_state is not None and type(random_state) is not int:
        raise ValueError("random_state must be an integer or None")
    if type(cv) is int:
        if not 2 <= cv <= sample_count:
            raise ValueError("cv must be an integer between 2 and the sample count")
        raw_folds = (
            stratified_k_fold_indices(
                labels, n_splits=cv, shuffle=shuffle, random_state=random_state
            )
            if stratified
            else k_fold_indices(
                sample_count, n_splits=cv, shuffle=shuffle, random_state=random_state
            )
        )
    else:
        if stratified or shuffle or random_state is not None:
            raise ValueError("split options apply only when cv is an integer")
        try:
            raw_folds = list(cv)
        except TypeError:
            raise ValueError("cv must be an integer or iterable of train/test splits") from None
        if not raw_folds:
            raise ValueError("cv must contain at least one split")

    normalized: list[tuple[list[int], list[int]]] = []
    for fold_number, fold in enumerate(raw_folds):
        try:
            pair = list(fold)
        except TypeError:
            raise ValueError(f"cv split {fold_number} must contain train and test indices") from None
        if len(pair) != 2:
            raise ValueError(f"cv split {fold_number} must contain exactly two index lists")
        current = []
        for name, indices in zip(("train", "test"), pair):
            try:
                values = list(indices)
            except TypeError:
                raise ValueError(f"cv split {fold_number} {name} indices must be iterable") from None
            if not values:
                raise ValueError(f"cv split {fold_number} {name} indices must not be empty")
            checked = []
            for value in values:
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise ValueError("cv indices must be integers")
                index = int(value)
                if not 0 <= index < sample_count:
                    raise ValueError("cv indices must be within the sample range")
                checked.append(index)
            if len(set(checked)) != len(checked):
                raise ValueError(f"cv split {fold_number} contains duplicate {name} indices")
            current.append(checked)
        train_indices, test_indices = current
        if set(train_indices) & set(test_indices):
            raise ValueError(f"cv split {fold_number} train and test indices must be disjoint")
        normalized.append((train_indices, test_indices))
    if not normalized:
        raise ValueError("cv must contain at least one split")
    return normalized


def _score_predictions(scoring: str, actual: list[Any], predicted: list[Any]) -> float:
    """Apply one supported built-in scorer to already aligned predictions."""
    if scoring == "accuracy":
        return sum(a == p for a, p in zip(actual, predicted)) / len(actual)
    if scoring == "balanced_accuracy":
        return balanced_accuracy_score(actual, predicted)
    if scoring == "r2":
        return r2_score(actual, predicted)
    if scoring == "neg_mean_squared_error":
        return -mean_squared_error(actual, predicted)
    if scoring == "neg_mean_absolute_error":
        return -mean_absolute_error(actual, predicted)
    return -root_mean_squared_error(actual, predicted)


def _mean_and_std(values: Iterable[float]) -> tuple[float, float]:
    """Return population mean and standard deviation without overflow.

    Scores are finite floats. Scaling them into [-1, 1] before aggregation
    avoids overflowing both the sum and squared deviations for large values.
    """
    scores = list(values)
    if not scores:
        raise ValueError("at least one score is required")
    scale = max(abs(score) for score in scores)
    if scale == 0.0:
        return 0.0, 0.0

    normalized = [score / scale for score in scores]
    mean_normalized = math.fsum(normalized) / len(normalized)
    mean_normalized = min(1.0, max(-1.0, mean_normalized))
    variance_normalized = math.fsum(
        (score - mean_normalized) ** 2 for score in normalized
    ) / len(normalized)
    std_normalized = min(1.0, math.sqrt(variance_normalized))
    return scale * mean_normalized, scale * std_normalized


def cross_val_score(
    estimator: Any,
    features: Iterable[Iterable[Any]],
    labels: Iterable[Any],
    cv: int | Iterable[Any] = 5,
    *,
    scoring: str | Scorer | None = None,
    stratified: bool = False,
    shuffle: bool = False,
    random_state: int | None = None,
) -> list[float]:
    """Evaluate a fresh copy of an estimator on each validation fold.

    Supported built-in scorers are accuracy, balanced_accuracy, r2,
    neg_mean_squared_error, neg_mean_absolute_error, and
    neg_root_mean_squared_error. A custom scorer receives
    (fitted_estimator, X_validation, y_validation). When scoring is None,
    the estimator's score method is used. Integer cv generates folds; an
    iterable cv supplies explicit (train, validation) index pairs. With integer
    cv, set stratified=True for classification targets; each class must contain
    at least cv samples. Explicit folds must be non-empty, disjoint, and in range.
    """
    matrix = _feature_rows(features)
    target = list(labels)
    sample_count = len(matrix)
    if len(target) != sample_count:
        raise ValueError("features and labels must have equal sample counts")
    if type(stratified) is not bool or type(shuffle) is not bool:
        raise ValueError("stratified and shuffle must be booleans")
    if random_state is not None and type(random_state) is not int:
        raise ValueError("random_state must be an integer or None")
    if scoring is not None and not callable(scoring) and scoring not in (
        "accuracy",
        "balanced_accuracy",
        "r2",
        "neg_mean_squared_error",
        "neg_mean_absolute_error",
        "neg_root_mean_squared_error",
    ):
        raise ValueError("scoring must be None, a supported name, or a callable")
    if not callable(getattr(estimator, "fit", None)):
        raise ValueError("estimator must implement fit")

    folds = _cv_splits(
        cv, sample_count, target, stratified=stratified, shuffle=shuffle,
        random_state=random_state,
    )

    scores = []
    for train_indices, validation_indices in folds:
        model = deepcopy(estimator)
        x_train = [matrix[index] for index in train_indices]
        y_train = [target[index] for index in train_indices]
        x_validation = [matrix[index] for index in validation_indices]
        y_validation = [target[index] for index in validation_indices]
        model.fit(x_train, y_train)

        if scoring is None:
            method = getattr(model, "score", None)
            if not callable(method):
                raise ValueError("estimator must implement score when scoring is None")
            value = method(x_validation, y_validation)
        elif callable(scoring):
            value = scoring(model, x_validation, y_validation)
        else:
            predict = getattr(model, "predict", None)
            if not callable(predict):
                raise ValueError("estimator must implement predict for built-in scoring")
            predictions = list(predict(x_validation))
            if len(predictions) != len(y_validation):
                raise ValueError("estimator predictions must match validation labels")
            value = _score_predictions(scoring, y_validation, predictions)

        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("scoring function must return a finite real number")
        try:
            score = float(value)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("scoring function must return a finite real number") from None
        if not math.isfinite(score):
            raise ValueError("scoring function must return a finite real number")
        scores.append(score)
    return scores


def _score_estimator(model: Any, features: list[list[Any]], labels: list[Any], scoring: str | Scorer | None) -> float:
    if scoring is None:
        method = getattr(model, "score", None)
        if not callable(method):
            raise ValueError("estimator must implement score when scoring is None")
        value = method(features, labels)
    elif callable(scoring):
        value = scoring(model, features, labels)
    else:
        predict = getattr(model, "predict", None)
        if not callable(predict):
            raise ValueError("estimator must implement predict for built-in scoring")
        predictions = list(predict(features))
        if len(predictions) != len(labels):
            raise ValueError("estimator predictions must match validation labels")
        value = _score_predictions(scoring, labels, predictions)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("scoring function must return a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("scoring function must return a finite real number") from None
    if not math.isfinite(result):
        raise ValueError("scoring function must return a finite real number")
    return result


def cross_validate(
    estimator: Any,
    features: Iterable[Iterable[Any]],
    labels: Iterable[Any],
    cv: int | Iterable[Any] = 5,
    *,
    scoring: str | Scorer | None = None,
    stratified: bool = False,
    shuffle: bool = False,
    random_state: int | None = None,
    return_train_score: bool = False,
    return_estimator: bool = False,
) -> dict[str, list[Any]]:
    """Evaluate fresh estimator copies and return fold scores and timings.

    The result always contains fit_time, score_time and test_score.
    Optional flags add training scores and the fitted estimator from each fold.
    Integer cv generates folds; an iterable supplies explicit train/validation
    index pairs. score_time measures held-out scoring only; optional
    training-score work is excluded. Timing values are descriptive and should
    not be compared across machines.
    """
    import time

    matrix = _feature_rows(features)
    target = list(labels)
    sample_count = len(matrix)
    if len(target) != sample_count:
        raise ValueError("features and labels must have equal sample counts")
    if type(stratified) is not bool or type(shuffle) is not bool:
        raise ValueError("stratified and shuffle must be booleans")
    if type(return_train_score) is not bool or type(return_estimator) is not bool:
        raise ValueError("return_train_score and return_estimator must be booleans")
    if random_state is not None and type(random_state) is not int:
        raise ValueError("random_state must be an integer or None")
    if scoring is not None and not callable(scoring) and scoring not in (
        "accuracy",
        "balanced_accuracy",
        "r2",
        "neg_mean_squared_error",
        "neg_mean_absolute_error",
        "neg_root_mean_squared_error",
    ):
        raise ValueError("scoring must be None, a supported name, or a callable")
    if not callable(getattr(estimator, "fit", None)):
        raise ValueError("estimator must implement fit")

    folds = _cv_splits(
        cv, sample_count, target, stratified=stratified, shuffle=shuffle,
        random_state=random_state,
    )
    results: dict[str, list[Any]] = {"fit_time": [], "score_time": [], "test_score": []}
    if return_train_score:
        results["train_score"] = []
    if return_estimator:
        results["estimator"] = []

    for train_indices, validation_indices in folds:
        model = deepcopy(estimator)
        x_train = [matrix[index] for index in train_indices]
        y_train = [target[index] for index in train_indices]
        x_validation = [matrix[index] for index in validation_indices]
        y_validation = [target[index] for index in validation_indices]

        started = time.perf_counter()
        model.fit(x_train, y_train)
        results["fit_time"].append(time.perf_counter() - started)

        started = time.perf_counter()
        test_score = _score_estimator(model, x_validation, y_validation, scoring)
        results["score_time"].append(time.perf_counter() - started)
        results["test_score"].append(test_score)
        if return_train_score:
            results["train_score"].append(_score_estimator(model, x_train, y_train, scoring))
        if return_estimator:
            results["estimator"].append(model)
    return results


def cross_val_predict(
    estimator: Any,
    features: Iterable[Iterable[Any]],
    labels: Iterable[Any],
    cv: int | Iterable[Any] = 5,
    *,
    method: str = "predict",
    stratified: bool = False,
    shuffle: bool = False,
    random_state: int | None = None,
) -> list[Any]:
    """Return one out-of-fold prediction for every sample in original order.

    Integer cv generates folds, or an iterable can provide explicit
    train/validation index pairs. Every input row must appear in exactly one
    validation fold. This helper is useful for diagnostic plots and stacking;
    its pooled predictions are not a substitute for the mean of fold-level
    cross-validation scores when fold sizes differ or the metric is
    non-decomposable.
    """
    matrix = _feature_rows(features)
    target = list(labels)
    sample_count = len(matrix)
    if len(target) != sample_count:
        raise ValueError("features and labels must have equal sample counts")
    if not isinstance(method, str) or not method or method.startswith("_"):
        raise ValueError("method must be a public estimator method name")
    if type(stratified) is not bool or type(shuffle) is not bool:
        raise ValueError("stratified and shuffle must be booleans")
    if random_state is not None and type(random_state) is not int:
        raise ValueError("random_state must be an integer or None")
    if not callable(getattr(estimator, "fit", None)):
        raise ValueError("estimator must implement fit")

    folds = _cv_splits(
        cv, sample_count, target, stratified=stratified, shuffle=shuffle,
        random_state=random_state,
    )
    output: list[Any] = [None] * sample_count
    assigned = [False] * sample_count
    for train_indices, validation_indices in folds:
        model = deepcopy(estimator)
        x_train = [matrix[index] for index in train_indices]
        y_train = [target[index] for index in train_indices]
        x_validation = [matrix[index] for index in validation_indices]
        model.fit(x_train, y_train)
        predictor = getattr(model, method, None)
        if not callable(predictor):
            raise ValueError(f"estimator does not implement method {method!r}")
        predictions = list(predictor(x_validation))
        if len(predictions) != len(validation_indices):
            raise ValueError("prediction method must return one result per validation row")
        for index, prediction in zip(validation_indices, predictions):
            if assigned[index]:
                raise ValueError("cross-validation folds must validate each sample exactly once")
            output[index] = prediction
            assigned[index] = True
    if not all(assigned):
        raise ValueError("cross-validation folds must cover every sample exactly once")
    return output
