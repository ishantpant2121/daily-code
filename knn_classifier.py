"""Dependency-free k-nearest-neighbors classification."""

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


class KNNClassifier:
    """Classify samples by voting among their nearest training observations.

    Supported metrics are Euclidean and Manhattan distance. Uniform voting
    gives every neighbor equal weight; distance voting uses inverse distance.
    Exact matches receive all distance weight, avoiding division by zero.
    Class order follows first appearance in the training labels and resolves
    probability ties deterministically.
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
        labels: Iterable[Any],
    ) -> "KNNClassifier":
        matrix = _feature_matrix(features)
        target = list(labels)
        if len(target) != len(matrix):
            raise ValueError("features and labels must have equal sample counts")
        class_indices: dict[Any, int] = {}
        try:
            for label in target:
                if label not in class_indices:
                    class_indices[label] = len(class_indices)
        except TypeError:
            raise ValueError("class labels must be hashable") from None
        if self.n_neighbors > len(matrix):
            raise ValueError("n_neighbors cannot exceed the training sample count")
        self.X_train_ = matrix
        self.y_train_ = target
        self.classes_ = tuple(class_indices)
        self._class_indices = class_indices
        self.n_features_in_ = len(matrix[0])
        self._fitted = True
        return self

    def predict_proba(self, features: Iterable[Iterable[Real]]) -> list[list[float]]:
        self._check_fitted()
        matrix = _feature_matrix(features)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")

        probabilities = []
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

            neighbors = sorted(distances, key=lambda item: (item[0], item[1]))[:self.n_neighbors]
            if self.weights == "uniform":
                neighbor_weights = [1.0] * len(neighbors)
            else:
                zero_count = sum(distance == 0 for distance, _ in neighbors)
                if zero_count:
                    neighbor_weights = [1.0 if distance == 0 else 0.0 for distance, _ in neighbors]
                else:
                    nearest_distance = neighbors[0][0]
                    neighbor_weights = [
                        nearest_distance / distance for distance, _ in neighbors
                    ]

            totals = [0.0] * len(self.classes_)
            for (_, training_index), weight in zip(neighbors, neighbor_weights):
                class_index = self._class_indices[self.y_train_[training_index]]
                totals[class_index] += weight
            denominator = math.fsum(totals)
            probabilities.append([total / denominator for total in totals])
        return probabilities

    def predict(self, features: Iterable[Iterable[Real]]) -> list[Any]:
        probabilities = self.predict_proba(features)
        return [
            self.classes_[max(range(len(row)), key=row.__getitem__)]
            for row in probabilities
        ]

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before prediction")
