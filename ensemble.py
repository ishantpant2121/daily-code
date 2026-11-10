"""Dependency-free voting ensembles for compatible classifier objects."""

import math
from collections.abc import Iterable, Sequence
from copy import deepcopy
from numbers import Real
from typing import Any


class VotingClassifier:
    """Combine fitted classifier copies with hard or probability voting.

    Estimators are supplied as unique (name, estimator) pairs. Hard voting
    chooses the class with the greatest total estimator weight; ties follow
    first-seen class order. Soft voting averages class-aligned probabilities.
    The original estimators are never fitted or mutated.
    """

    def __init__(
        self,
        estimators: Sequence[tuple[str, Any]],
        *,
        voting: str = "hard",
        weights: Sequence[Real] | None = None,
    ) -> None:
        materialized = list(estimators)
        if len(materialized) < 2:
            raise ValueError("at least two estimators are required")
        names = []
        normalized = []
        for item in materialized:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("each estimator must be a (name, estimator) pair")
            name, estimator = item
            if not isinstance(name, str) or not name or "__" in name:
                raise ValueError("estimator names must be non-empty and cannot contain double underscores")
            if name in names:
                raise ValueError("estimator names must be unique")
            if not callable(getattr(estimator, "fit", None)) or not callable(
                getattr(estimator, "predict", None)
            ):
                raise ValueError("each estimator must implement fit and predict")
            names.append(name)
            normalized.append((name, estimator))
        if voting not in ("hard", "soft"):
            raise ValueError("voting must be 'hard' or 'soft'")

        if weights is None:
            normalized_weights = [1.0] * len(normalized)
        else:
            normalized_weights = list(weights)
            if len(normalized_weights) != len(normalized):
                raise ValueError("weights must contain one value per estimator")
            converted = []
            for weight in normalized_weights:
                if isinstance(weight, bool) or not isinstance(weight, Real):
                    raise ValueError("weights must be finite non-negative real numbers")
                try:
                    value = float(weight)
                except (OverflowError, TypeError, ValueError):
                    raise ValueError("weights must be finite non-negative real numbers") from None
                if not math.isfinite(value) or value < 0:
                    raise ValueError("weights must be finite non-negative real numbers")
                converted.append(value)
            normalized_weights = converted
            try:
                total_weight = math.fsum(normalized_weights)
            except OverflowError:
                raise ValueError("weights must have a finite positive sum") from None
            if not math.isfinite(total_weight) or total_weight <= 0:
                raise ValueError("weights must have a finite positive sum")

        self.estimators = tuple(normalized)
        self.voting = voting
        self.weights = tuple(normalized_weights)
        self._fitted = False

    def fit(self, features: Iterable[Iterable[Any]], labels: Iterable[Any]) -> "VotingClassifier":
        from model_selection import _feature_rows

        matrix = _feature_rows(features)
        target = list(labels)
        if len(matrix) != len(target):
            raise ValueError("features and labels must have equal sample counts")
        classes = []
        seen = set()
        try:
            for label in target:
                if label not in seen:
                    seen.add(label)
                    classes.append(label)
        except TypeError:
            raise ValueError("class labels must be hashable") from None

        fitted = []
        for name, estimator in self.estimators:
            model = deepcopy(estimator)
            model.fit(matrix, target)
            if self.voting == "soft" and not callable(getattr(model, "predict_proba", None)):
                raise ValueError(f"estimator {name!r} must implement predict_proba for soft voting")
            if self.voting == "soft" and not hasattr(model, "classes_"):
                raise ValueError(f"estimator {name!r} must expose classes_ for soft voting")
            fitted.append(model)

        self.classes_ = tuple(classes)
        self.n_features_in_ = len(matrix[0])
        self.estimators_ = fitted
        self.named_estimators_ = {
            name: model for (name, _), model in zip(self.estimators, fitted)
        }
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before prediction")

    def predict_proba(self, features: Iterable[Iterable[Any]]) -> list[list[float]]:
        self._check_fitted()
        from model_selection import _feature_rows

        matrix = _feature_rows(features)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        total_weight = math.fsum(self.weights)
        probabilities = [[0.0] * len(self.classes_) for _ in matrix]

        for model, weight in zip(self.estimators_, self.weights):
            if weight == 0:
                continue
            if self.voting == "soft":
                method = getattr(model, "predict_proba", None)
                if not callable(method):
                    raise ValueError("all soft-voting estimators must implement predict_proba")
                model_classes = list(model.classes_)
                try:
                    class_positions = [self.classes_.index(label) for label in model_classes]
                except ValueError:
                    raise ValueError("estimator classes do not match the fitted ensemble classes") from None
                rows = list(method(matrix))
                if len(rows) != len(matrix):
                    raise ValueError("predict_proba must return one row per sample")
                for row_index, row in enumerate(rows):
                    values = list(row)
                    if len(values) != len(class_positions):
                        raise ValueError("probability columns must match estimator classes")
                    converted = []
                    for value in values:
                        if isinstance(value, bool) or not isinstance(value, Real):
                            raise ValueError("probabilities must be finite non-negative numbers")
                        try:
                            number = float(value)
                        except (OverflowError, TypeError, ValueError):
                            raise ValueError("probabilities must be finite non-negative numbers") from None
                        if not math.isfinite(number) or number < 0:
                            raise ValueError("probabilities must be finite non-negative numbers")
                        converted.append(number)
                    if not math.isclose(math.fsum(converted), 1.0, rel_tol=1e-6, abs_tol=1e-8):
                        raise ValueError("each probability row must sum to one")
                    for position, value in zip(class_positions, converted):
                        probabilities[row_index][position] += weight * value
            else:
                predictions = list(model.predict(matrix))
                if len(predictions) != len(matrix):
                    raise ValueError("predict must return one label per sample")
                for row_index, prediction in enumerate(predictions):
                    try:
                        position = self.classes_.index(prediction)
                    except ValueError:
                        raise ValueError("estimator predicted an unknown class label") from None
                    probabilities[row_index][position] += weight

        return [[value / total_weight for value in row] for row in probabilities]

    def predict(self, features: Iterable[Iterable[Any]]) -> list[Any]:
        probabilities = self.predict_proba(features)
        return [
            self.classes_[max(range(len(row)), key=lambda index: row[index])]
            for row in probabilities
        ]

    def score(self, features: Iterable[Iterable[Any]], labels: Iterable[Any]) -> float:
        actual = list(labels)
        predicted = self.predict(features)
        if len(actual) != len(predicted):
            raise ValueError("features and labels must have equal sample counts")
        if not actual:
            raise ValueError("features and labels must not be empty")
        return sum(left == right for left, right in zip(actual, predicted)) / len(actual)
