"""Principal component analysis using a symmetric Jacobi eigensolver."""

import math
from numbers import Real
from typing import Iterable


def _feature_matrix(values: Iterable[Iterable[Real]]) -> list[list[float]]:
    rows = [list(row) for row in values]
    if not rows or not rows[0]:
        raise ValueError("input must contain at least one row and one feature")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all rows must contain the same number of features")
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


def _jacobi_eigen(
    matrix: list[list[float]],
    tolerance: float,
    max_iter: int,
) -> tuple[list[float], list[list[float]]]:
    """Diagonalize a real symmetric matrix and return eigenvalues/vectors."""
    size = len(matrix)
    values = [row[:] for row in matrix]
    vectors = [
        [1.0 if row == column else 0.0 for column in range(size)]
        for row in range(size)
    ]
    largest = 0.0
    for _ in range(max_iter):
        p = q = 0
        largest = 0.0
        for row in range(size):
            for column in range(row + 1, size):
                if abs(values[row][column]) > largest:
                    largest = abs(values[row][column])
                    p, q = row, column
        if largest <= tolerance:
            break

        app = values[p][p]
        aqq = values[q][q]
        apq = values[p][q]
        angle = 0.5 * math.atan2(2.0 * apq, aqq - app)
        cosine = math.cos(angle)
        sine = math.sin(angle)

        values[p][p] = (
            cosine * cosine * app
            - 2.0 * sine * cosine * apq
            + sine * sine * aqq
        )
        values[q][q] = (
            sine * sine * app
            + 2.0 * sine * cosine * apq
            + cosine * cosine * aqq
        )
        values[p][q] = values[q][p] = 0.0

        for index in range(size):
            if index != p and index != q:
                first = values[index][p]
                second = values[index][q]
                values[index][p] = values[p][index] = cosine * first - sine * second
                values[index][q] = values[q][index] = sine * first + cosine * second
            first_vector = vectors[index][p]
            second_vector = vectors[index][q]
            vectors[index][p] = cosine * first_vector - sine * second_vector
            vectors[index][q] = sine * first_vector + cosine * second_vector

    if largest > tolerance:
        raise RuntimeError("symmetric eigensolver did not converge")
    return [values[index][index] for index in range(size)], vectors


class PCA:
    """Project centered numeric data onto its leading principal components.

    Components are ordered by descending sample covariance eigenvalue.
    The implementation uses a Jacobi rotation method and the Python standard
    library, making it suitable for educational small and medium matrices.
    """

    def __init__(
        self,
        n_components: int | None = None,
        *,
        tol: float = 1e-12,
        max_iter: int | None = None,
    ) -> None:
        if n_components is not None and (
            type(n_components) is not int or n_components < 1
        ):
            raise ValueError("n_components must be a positive integer or None")
        if isinstance(tol, bool) or not isinstance(tol, Real):
            raise ValueError("tol must be a finite non-negative real number")
        try:
            tolerance = float(tol)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("tol must be a finite non-negative real number") from None
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tol must be a finite non-negative real number")
        if max_iter is not None and (type(max_iter) is not int or max_iter < 1):
            raise ValueError("max_iter must be a positive integer or None")
        self.n_components = n_components
        self.tol = tolerance
        self.max_iter = max_iter
        self._fitted = False

    def fit(self, values: Iterable[Iterable[Real]]) -> "PCA":
        matrix = _feature_matrix(values)
        sample_count = len(matrix)
        feature_count = len(matrix[0])
        if sample_count < 2:
            raise ValueError("PCA requires at least two samples")
        component_count = (
            min(sample_count, feature_count)
            if self.n_components is None
            else self.n_components
        )
        if component_count > feature_count:
            raise ValueError("n_components cannot exceed the feature count")

        try:
            mean = [
                math.fsum(row[column] for row in matrix) / sample_count
                for column in range(feature_count)
            ]
            centered = [
                [value - mean[column] for column, value in enumerate(row)]
                for row in matrix
            ]
            covariance = [
                [
                    math.fsum(
                        row[first] * row[second] for row in centered
                    ) / (sample_count - 1)
                    for second in range(feature_count)
                ]
                for first in range(feature_count)
            ]
        except OverflowError:
            raise ValueError("covariance exceeds finite float range") from None
        if any(
            not math.isfinite(value)
            for row in covariance
            for value in row
        ):
            raise ValueError("covariance exceeds finite float range")

        iterations = self.max_iter or max(1, 100 * feature_count * feature_count)
        eigenvalues, eigenvectors = _jacobi_eigen(
            covariance, self.tol, iterations
        )
        order = sorted(range(feature_count), key=lambda index: eigenvalues[index], reverse=True)
        ordered_values = []
        ordered_components = []
        for index in order:
            value = eigenvalues[index]
            if value < -self.tol:
                raise ValueError("covariance matrix produced a negative eigenvalue")
            ordered_values.append(max(0.0, value))
            ordered_components.append(
                [eigenvectors[row][index] for row in range(feature_count)]
            )

        selected_values = ordered_values[:component_count]
        selected_components = ordered_components[:component_count]
        total_variance = math.fsum(ordered_values)
        ratios = (
            [value / total_variance for value in selected_values]
            if total_variance > 0
            else [0.0] * component_count
        )

        self.mean_ = mean
        self.components_ = selected_components
        self.explained_variance_ = selected_values
        self.explained_variance_ratio_ = ratios
        self.n_features_in_ = feature_count
        self.n_components_ = component_count
        self._fitted = True
        return self

    def transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        self._check_fitted()
        matrix = _feature_matrix(values)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        return [
            [
                math.fsum(
                    (value - self.mean_[index]) * component[index]
                    for index, value in enumerate(row)
                )
                for component in self.components_
            ]
            for row in matrix
        ]

    def inverse_transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        self._check_fitted()
        scores = _feature_matrix(values)
        if len(scores[0]) != self.n_components_:
            raise ValueError("input component count does not match fitted data")
        return [
            [
                self.mean_[feature]
                + math.fsum(
                    row[component] * self.components_[component][feature]
                    for component in range(self.n_components_)
                )
                for feature in range(self.n_features_in_)
            ]
            for row in scores
        ]

    def fit_transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        matrix = _feature_matrix(values)
        self.fit(matrix)
        return self.transform(matrix)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before transforming data")
