"""Gaussian Naive Bayes classification without third-party dependencies."""

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
    matrix = []
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
        matrix.append(converted)
    return matrix


class GaussianNB:
    """Classify observations with class-conditional Gaussian feature models.

    Feature variances receive additive smoothing based on the largest global
    feature variance. If all features are constant, a unit-scale fallback
    keeps likelihoods finite. Class order follows first appearance in labels.
    """

    def __init__(self, *, var_smoothing: float = 1e-9) -> None:
        if isinstance(var_smoothing, bool) or not isinstance(var_smoothing, Real):
            raise ValueError("var_smoothing must be a positive finite real number")
        try:
            smoothing = float(var_smoothing)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("var_smoothing must be a positive finite real number") from None
        if not math.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("var_smoothing must be a positive finite real number")
        self.var_smoothing = smoothing
        self._fitted = False

    def fit(
        self,
        features: Iterable[Iterable[Real]],
        labels: Iterable[Any],
    ) -> "GaussianNB":
        matrix = _feature_matrix(features)
        target = list(labels)
        if len(target) != len(matrix):
            raise ValueError("features and labels must have equal sample counts")
        class_indices = {}
        try:
            for label in target:
                if label not in class_indices:
                    class_indices[label] = len(class_indices)
        except TypeError:
            raise ValueError("class labels must be hashable") from None
        if len(class_indices) < 2:
            raise ValueError("GaussianNB requires at least two classes")

        sample_count = len(matrix)
        feature_count = len(matrix[0])
        global_means = [
            math.fsum(row[column] for row in matrix) / sample_count
            for column in range(feature_count)
        ]
        global_variances = [
            math.fsum(
                (row[column] - global_means[column]) ** 2 for row in matrix
            ) / sample_count
            for column in range(feature_count)
        ]
        largest_variance = max(global_variances)
        scale = largest_variance if largest_variance > 0 else 1.0
        epsilon = self.var_smoothing * scale
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("variance smoothing exceeds finite float range")

        class_rows = [[] for _ in class_indices]
        for row, label in zip(matrix, target):
            class_rows[class_indices[label]].append(row)

        means = []
        variances = []
        counts = []
        priors = []
        for rows in class_rows:
            count = len(rows)
            counts.append(count)
            priors.append(count / sample_count)
            class_mean = [
                math.fsum(row[column] for row in rows) / count
                for column in range(feature_count)
            ]
            class_variance = [
                math.fsum(
                    (row[column] - class_mean[column]) ** 2 for row in rows
                ) / count + epsilon
                for column in range(feature_count)
            ]
            if any(not math.isfinite(value) or value <= 0 for value in class_variance):
                raise ValueError("class variance exceeds finite float range")
            means.append(class_mean)
            variances.append(class_variance)

        self.classes_ = tuple(class_indices)
        self.class_count_ = counts
        self.class_prior_ = priors
        self.theta_ = means
        self.var_ = variances
        self.epsilon_ = epsilon
        self.n_features_in_ = feature_count
        self._fitted = True
        return self

    def predict_log_proba(
        self,
        features: Iterable[Iterable[Real]],
    ) -> list[list[float]]:
        self._check_fitted()
        matrix = _feature_matrix(features)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        results = []
        for row in matrix:
            joint = []
            for class_index, prior in enumerate(self.class_prior_):
                terms = []
                for column, value in enumerate(row):
                    variance = self.var_[class_index][column]
                    difference = value - self.theta_[class_index][column]
                    try:
                        terms.append(
                            math.log(2.0 * math.pi * variance)
                            + difference * difference / variance
                        )
                    except (OverflowError, ValueError):
                        raise ValueError("Gaussian likelihood exceeds finite float range") from None
                try:
                    score = math.log(prior) - 0.5 * math.fsum(terms)
                except OverflowError:
                    raise ValueError("Gaussian likelihood exceeds finite float range") from None
                if not math.isfinite(score):
                    raise ValueError("Gaussian likelihood exceeds finite float range")
                joint.append(score)
            maximum = max(joint)
            normalizer = maximum + math.log(
                math.fsum(math.exp(value - maximum) for value in joint)
            )
            results.append([value - normalizer for value in joint])
        return results

    def predict_proba(self, features: Iterable[Iterable[Real]]) -> list[list[float]]:
        return [
            [math.exp(value) for value in row]
            for row in self.predict_log_proba(features)
        ]

    def predict(self, features: Iterable[Iterable[Real]]) -> list[Any]:
        probabilities = self.predict_proba(features)
        return [
            self.classes_[max(range(len(row)), key=row.__getitem__)]
            for row in probabilities
        ]

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before prediction")
