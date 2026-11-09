"""Dependency-free k-nearest-neighbors regression."""

import math
from numbers import Real
from typing import Iterable

from regression_metrics import r2_score


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


def _target_values(values: Iterable[Real]) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("targets must be finite real numbers")
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("targets must be finite real numbers") from None
        if not math.isfinite(number):
            raise ValueError("targets must be finite real numbers")
        result.append(number)
    return result


class KNNRegressor:
    """Predict numeric targets by averaging the nearest training samples.

    Euclidean and Manhattan distances are supported. Distance weighting uses
    inverse distance; exact matches share all weight to avoid division by zero.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        weights: str = "uniform",
        metric: str = "euclidean",
    ) -> None:
        if type(n_neighbors) is not int or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self._fitted = False

    def fit(
        self,
        features: Iterable[Iterable[Real]],
        targets: Iterable[Real],
    ) -> "KNNRegressor":
        matrix = _feature_matrix(features)
        values = _target_values(targets)
        if len(values) != len(matrix):
            raise ValueError("features and targets must have equal sample counts")
        if self.n_neighbors > len(matrix):
            raise ValueError("n_neighbors cannot exceed the training sample count")
        self.X_train_ = matrix
        self.y_train_ = values
        self.n_features_in_ = len(matrix[0])
        self._fitted = True
        return self

    def predict(self, features: Iterable[Iterable[Real]]) -> list[float]:
        self._check_fitted()
        matrix = _feature_matrix(features)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        predictions = []
        for sample in matrix:
            distances = []
            for index, training_sample in enumerate(self.X_train_):
                try:
                    if self.metric == "euclidean":
                        distance = math.dist(sample, training_sample)
                    else:
                        distance = math.fsum(
                            abs(left - right)
                            for left, right in zip(sample, training_sample)
                        )
                except OverflowError:
                    raise ValueError("distance exceeds finite float range") from None
                if not math.isfinite(distance):
                    raise ValueError("distance exceeds finite float range")
                distances.append((distance, index))
            neighbors = sorted(distances, key=lambda item: (item[0], item[1]))[
                :self.n_neighbors
            ]
            if self.weights == "uniform":
                weights = [1.0] * len(neighbors)
            else:
                zero_count = sum(distance == 0 for distance, _ in neighbors)
                if zero_count:
                    weights = [
                        1.0 if distance == 0 else 0.0
                        for distance, _ in neighbors
                    ]
                else:
                    nearest = neighbors[0][0]
                    weights = [nearest / distance for distance, _ in neighbors]
            numerator = math.fsum(
                self.y_train_[index] * weight
                for (_, index), weight in zip(neighbors, weights)
            )
            denominator = math.fsum(weights)
            prediction = numerator / denominator
            if not math.isfinite(prediction):
                raise ValueError("prediction exceeds finite float range")
            predictions.append(prediction)
        return predictions

    def score(
        self,
        features: Iterable[Iterable[Real]],
        targets: Iterable[Real],
    ) -> float:
        actual = _target_values(targets)
        predicted = self.predict(features)
        return r2_score(actual, predicted)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before prediction")
