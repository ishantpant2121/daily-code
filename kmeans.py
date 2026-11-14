"""K-means clustering with k-means++ initialization."""

import math
import random
from numbers import Real
from typing import Iterable

from statistics_utils import mean as numeric_mean


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


def _initialize_centers(
    matrix: list[list[float]],
    n_clusters: int,
    generator: random.Random,
) -> list[list[float]]:
    selected = [generator.randrange(len(matrix))]
    centers = [matrix[selected[0]][:]]
    while len(centers) < n_clusters:
        distances = []
        for row in matrix:
            distance = min(math.dist(row, center) for center in centers)
            squared = distance * distance
            if not math.isfinite(distance) or not math.isfinite(squared):
                raise ValueError("squared distance exceeds finite float range")
            distances.append(squared)
        total = math.fsum(distances)
        if not math.isfinite(total):
            raise ValueError("k-means initialization weight exceeds finite float range")
        if total == 0:
            candidates = [index for index in range(len(matrix)) if index not in selected]
            chosen = generator.choice(candidates)
        else:
            threshold = generator.random() * total
            cumulative = 0.0
            chosen = len(matrix) - 1
            for index, distance in enumerate(distances):
                cumulative += distance
                if cumulative >= threshold:
                    chosen = index
                    break
        selected.append(chosen)
        centers.append(matrix[chosen][:])
    return centers


def _assign(
    matrix: list[list[float]],
    centers: list[list[float]],
) -> tuple[list[int], list[float]]:
    labels = []
    distances = []
    for row in matrix:
        current = [math.dist(row, center) for center in centers]
        if any(not math.isfinite(distance) for distance in current):
            raise ValueError("distance exceeds finite float range")
        selected = min(range(len(current)), key=current.__getitem__)
        labels.append(selected)
        distances.append(current[selected])
    return labels, distances


class KMeans:
    """Partition samples into clusters by minimizing within-cluster squared distance.

    The implementation uses k-means++ seeding, multiple initializations,
    deterministic tie-breaking, and farthest-point reseeding for empty clusters.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        if type(n_clusters) is not int or n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer")
        if type(n_init) is not int or n_init < 1:
            raise ValueError("n_init must be a positive integer")
        if type(max_iter) is not int or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if isinstance(tol, bool) or not isinstance(tol, Real):
            raise ValueError("tol must be a finite non-negative real number")
        try:
            tolerance = float(tol)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("tol must be a finite non-negative real number") from None
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tol must be a finite non-negative real number")
        if random_state is not None and type(random_state) is not int:
            raise ValueError("random_state must be an integer or None")
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tolerance
        self.random_state = random_state
        self._fitted = False

    def fit(self, features: Iterable[Iterable[Real]]) -> "KMeans":
        matrix = _feature_matrix(features)
        if self.n_clusters > len(matrix):
            raise ValueError("n_clusters cannot exceed the sample count")
        generator = random.Random(self.random_state)
        best = None

        for _ in range(self.n_init):
            centers = _initialize_centers(matrix, self.n_clusters, generator)
            previous_labels = None
            for iteration in range(1, self.max_iter + 1):
                labels, distances = _assign(matrix, centers)
                groups = [[] for _ in range(self.n_clusters)]
                for index, label in enumerate(labels):
                    groups[label].append(index)

                updated = []
                reserved = set()
                for cluster, indices in enumerate(groups):
                    if indices:
                        updated.append([
                            numeric_mean(matrix[index][column] for index in indices)
                            for column in range(len(matrix[0]))
                        ])
                    else:
                        candidates = sorted(
                            (index for index in range(len(matrix)) if index not in reserved),
                            key=lambda index: (-distances[index], index),
                        )
                        if candidates:
                            chosen = candidates[0]
                            reserved.add(chosen)
                            updated.append(matrix[chosen][:])
                        else:
                            updated.append(centers[cluster][:])

                shifts = [
                    math.dist(old, new)
                    for old, new in zip(centers, updated)
                ]
                if any(not math.isfinite(shift) for shift in shifts):
                    raise ValueError("centroid shift exceeds finite float range")
                centers = updated
                if labels == previous_labels or max(shifts, default=0.0) <= self.tol:
                    break
                previous_labels = labels

            labels, distances = _assign(matrix, centers)
            try:
                inertia = math.fsum(distance * distance for distance in distances)
            except OverflowError:
                raise ValueError("cluster inertia exceeds finite float range") from None
            if not math.isfinite(inertia):
                raise ValueError("cluster inertia exceeds finite float range")
            candidate = (inertia, centers, labels, iteration)
            if best is None or candidate[0] < best[0]:
                best = candidate

        self.inertia_, self.cluster_centers_, self.labels_, self.n_iter_ = best
        self.n_features_in_ = len(matrix[0])
        self._fitted = True
        return self

    def fit_predict(self, features: Iterable[Iterable[Real]]) -> list[int]:
        matrix = _feature_matrix(features)
        self.fit(matrix)
        return self.labels_[:]

    def predict(self, features: Iterable[Iterable[Real]]) -> list[int]:
        self._check_fitted()
        matrix = _feature_matrix(features)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        labels, _ = _assign(matrix, self.cluster_centers_)
        return labels

    def transform(self, features: Iterable[Iterable[Real]]) -> list[list[float]]:
        self._check_fitted()
        matrix = _feature_matrix(features)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        return [
            [math.dist(row, center) for center in self.cluster_centers_]
            for row in matrix
        ]

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before prediction")
