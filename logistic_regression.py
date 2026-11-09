"""Binary logistic regression trained with batch gradient descent."""

import math
from numbers import Real
from typing import Any, Iterable


def _feature_matrix(values: Iterable[Iterable[Real]]) -> list[list[float]]:
    rows = [list(row) for row in values]
    if not rows or not rows[0]:
        raise ValueError("features must contain at least one sample and one column")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all feature rows must have equal lengths")
    result = []
    for row in rows:
        converted = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("features must be finite real numbers")
            try:
                number = float(value)
            except (OverflowError, TypeError, ValueError):
                raise ValueError("features must be finite real numbers") from None
            if not math.isfinite(number):
                raise ValueError("features must be finite real numbers")
            converted.append(number)
        result.append(converted)
    return result


def _sigmoid(value: float) -> float:
    """Evaluate the logistic sigmoid without overflowing exponentials."""
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


class LogisticRegression:
    """Fit a binary logistic classifier using full-batch gradient descent.

    The first class encountered in the training labels is encoded as 0 and
    the second as 1. L2 regularization applies to coefficients, not the
    intercept. Feature scaling is recommended for faster convergence.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 1000,
        *,
        tol: float = 1e-6,
        l2: float = 0.0,
    ) -> None:
        for name, value, allow_zero in (
            ("learning_rate", learning_rate, False),
            ("tol", tol, True),
            ("l2", l2, True),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite real number")
            try:
                numeric = float(value)
            except (OverflowError, TypeError, ValueError):
                raise ValueError(f"{name} must be a finite real number") from None
            if not math.isfinite(numeric) or (numeric < 0 if allow_zero else numeric <= 0):
                raise ValueError(f"{name} is outside its valid range")
            setattr(self, name, numeric)
        if type(max_iter) is not int or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        self.max_iter = max_iter
        self._fitted = False

    def fit(
        self,
        features: Iterable[Iterable[Real]],
        labels: Iterable[Any],
    ) -> "LogisticRegression":
        matrix = _feature_matrix(features)
        target = list(labels)
        if len(target) != len(matrix):
            raise ValueError("features and labels must have equal sample counts")
        mapping = {}
        try:
            for label in target:
                if label not in mapping:
                    mapping[label] = len(mapping)
        except TypeError:
            raise ValueError("class labels must be hashable") from None
        if len(mapping) != 2:
            raise ValueError("binary logistic regression requires exactly two classes")
        encoded = [mapping[label] for label in target]

        sample_count = len(matrix)
        feature_count = len(matrix[0])
        weights = [0.0] * feature_count
        intercept = 0.0

        for iteration in range(1, self.max_iter + 1):
            errors = []
            for row, label in zip(matrix, encoded):
                try:
                    linear = math.fsum(
                        weight * value for weight, value in zip(weights, row)
                    ) + intercept
                except OverflowError:
                    raise ValueError("linear predictor exceeds finite float range") from None
                if not math.isfinite(linear):
                    raise ValueError("linear predictor exceeds finite float range")
                errors.append(_sigmoid(linear) - label)

            try:
                gradients = [
                    math.fsum(
                        error * row[column]
                        for error, row in zip(errors, matrix)
                    ) / sample_count + self.l2 * weights[column]
                    for column in range(feature_count)
                ]
                intercept_gradient = math.fsum(errors) / sample_count
            except OverflowError:
                raise ValueError("gradient exceeds finite float range") from None
            if any(
                not math.isfinite(value)
                for value in gradients + [intercept_gradient]
            ):
                raise ValueError("gradient exceeds finite float range")

            weights = [
                weight - self.learning_rate * gradient
                for weight, gradient in zip(weights, gradients)
            ]
            intercept -= self.learning_rate * intercept_gradient
            if any(not math.isfinite(value) for value in weights + [intercept]):
                raise ValueError("model parameters exceed finite float range")
            if max(
                [abs(value) for value in gradients] + [abs(intercept_gradient)]
            ) <= self.tol:
                break

        self.classes_ = tuple(mapping)
        self.coef_ = weights
        self.intercept_ = intercept
        self.n_features_in_ = feature_count
        self.n_iter_ = iteration
        self._fitted = True
        return self

    def decision_function(self, features: Iterable[Iterable[Real]]) -> list[float]:
        self._check_fitted()
        matrix = _feature_matrix(features)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        scores = []
        for row in matrix:
            try:
                score = math.fsum(
                    weight * value for weight, value in zip(self.coef_, row)
                ) + self.intercept_
            except OverflowError:
                raise ValueError("linear predictor exceeds finite float range") from None
            if not math.isfinite(score):
                raise ValueError("linear predictor exceeds finite float range")
            scores.append(score)
        return scores

    def predict_proba(self, features: Iterable[Iterable[Real]]) -> list[list[float]]:
        scores = self.decision_function(features)
        probabilities = []
        for score in scores:
            positive = _sigmoid(score)
            probabilities.append([1.0 - positive, positive])
        return probabilities

    def predict(self, features: Iterable[Iterable[Real]]) -> list[Any]:
        probabilities = self.predict_proba(features)
        return [
            self.classes_[1] if row[1] >= 0.5 else self.classes_[0]
            for row in probabilities
        ]

    def score(
        self,
        features: Iterable[Iterable[Real]],
        labels: Iterable[Any],
    ) -> float:
        actual = list(labels)
        predicted = self.predict(features)
        if not actual:
            raise ValueError("labels must not be empty")
        if len(actual) != len(predicted):
            raise ValueError("labels and predictions must have equal lengths")
        return sum(left == right for left, right in zip(actual, predicted)) / len(actual)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before prediction")
