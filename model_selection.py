"""Small, deterministic cross-validation helpers for estimator objects."""

import math
from copy import deepcopy
from numbers import Real
from typing import Any, Callable, Iterable

from classification_metrics import balanced_accuracy_score
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


def cross_val_score(
    estimator: Any,
    features: Iterable[Iterable[Any]],
    labels: Iterable[Any],
    cv: int = 5,
    *,
    scoring: str | Scorer | None = None,
    stratified: bool = False,
    shuffle: bool = False,
    random_state: int | None = None,
) -> list[float]:
    """Evaluate a fresh copy of an estimator on each validation fold.

    Supported built-in scorers are accuracy and balanced_accuracy. A custom
    scorer receives (fitted_estimator, X_validation, y_validation). When
    scoring is None, the estimator's score method is used. Set stratified=True
    for classification targets; every class must contain at least cv samples.
    """
    matrix = _feature_rows(features)
    target = list(labels)
    sample_count = len(matrix)
    if len(target) != sample_count:
        raise ValueError("features and labels must have equal sample counts")
    if type(cv) is not int or not 2 <= cv <= sample_count:
        raise ValueError("cv must be an integer between 2 and the sample count")
    if type(stratified) is not bool or type(shuffle) is not bool:
        raise ValueError("stratified and shuffle must be booleans")
    if random_state is not None and type(random_state) is not int:
        raise ValueError("random_state must be an integer or None")
    if scoring is not None and not callable(scoring) and scoring not in (
        "accuracy",
        "balanced_accuracy",
    ):
        raise ValueError("scoring must be None, a supported name, or a callable")
    if not callable(getattr(estimator, "fit", None)):
        raise ValueError("estimator must implement fit")

    if stratified:
        folds = stratified_k_fold_indices(
            target, n_splits=cv, shuffle=shuffle, random_state=random_state
        )
    else:
        folds = k_fold_indices(
            sample_count, n_splits=cv, shuffle=shuffle, random_state=random_state
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
            if scoring == "accuracy":
                value = sum(
                    actual == predicted
                    for actual, predicted in zip(y_validation, predictions)
                ) / len(y_validation)
            else:
                value = balanced_accuracy_score(y_validation, predictions)

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
        if scoring == "accuracy":
            value = sum(actual == predicted for actual, predicted in zip(labels, predictions)) / len(labels)
        else:
            value = balanced_accuracy_score(labels, predictions)
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
    cv: int = 5,
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
    Timing values are descriptive and should not be compared across machines.
    """
    import time

    matrix = _feature_rows(features)
    target = list(labels)
    sample_count = len(matrix)
    if len(target) != sample_count:
        raise ValueError("features and labels must have equal sample counts")
    if type(cv) is not int or not 2 <= cv <= sample_count:
        raise ValueError("cv must be an integer between 2 and the sample count")
    if type(stratified) is not bool or type(shuffle) is not bool:
        raise ValueError("stratified and shuffle must be booleans")
    if type(return_train_score) is not bool or type(return_estimator) is not bool:
        raise ValueError("return_train_score and return_estimator must be booleans")
    if random_state is not None and type(random_state) is not int:
        raise ValueError("random_state must be an integer or None")
    if scoring is not None and not callable(scoring) and scoring not in ("accuracy", "balanced_accuracy"):
        raise ValueError("scoring must be None, a supported name, or a callable")
    if not callable(getattr(estimator, "fit", None)):
        raise ValueError("estimator must implement fit")

    folds = (
        stratified_k_fold_indices(target, n_splits=cv, shuffle=shuffle, random_state=random_state)
        if stratified
        else k_fold_indices(sample_count, n_splits=cv, shuffle=shuffle, random_state=random_state)
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
        results["test_score"].append(_score_estimator(model, x_validation, y_validation, scoring))
        if return_train_score:
            results["train_score"].append(_score_estimator(model, x_train, y_train, scoring))
        results["score_time"].append(time.perf_counter() - started)
        if return_estimator:
            results["estimator"].append(model)
    return results


def cross_val_predict(
    estimator: Any,
    features: Iterable[Iterable[Any]],
    labels: Iterable[Any],
    cv: int = 5,
    *,
    method: str = "predict",
    stratified: bool = False,
    shuffle: bool = False,
    random_state: int | None = None,
) -> list[Any]:
    """Return one out-of-fold prediction for every sample in original order.

    Every input row must appear in exactly one validation fold. This helper is
    useful for diagnostic plots and stacking; its pooled predictions are not
    a substitute for the mean of fold-level cross-validation scores when fold
    sizes differ or the metric is non-decomposable.
    """
    matrix = _feature_rows(features)
    target = list(labels)
    sample_count = len(matrix)
    if len(target) != sample_count:
        raise ValueError("features and labels must have equal sample counts")
    if type(cv) is not int or not 2 <= cv <= sample_count:
        raise ValueError("cv must be an integer between 2 and the sample count")
    if not isinstance(method, str) or not method or method.startswith("_"):
        raise ValueError("method must be a public estimator method name")
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
