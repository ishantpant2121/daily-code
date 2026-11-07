"""Small dependency-free feature scalers with fit/transform semantics."""

import math
from numbers import Real
from typing import Iterable


def _validated_matrix(values: Iterable[Iterable[Real]]) -> list[list[float]]:
    rows = [list(row) for row in values]
    if not rows or not rows[0]:
        raise ValueError("input must contain at least one row and one feature")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all rows must contain the same number of features")
    result: list[list[float]] = []
    for row in rows:
        converted = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("features must be real numbers")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("features must be finite")
            converted.append(number)
        result.append(converted)
    return result


class MinMaxScaler:
    """Scale each feature to a target range using training-set extrema.

    Constant training features map to the lower range bound and inverse
    transform back to their original constant value.
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        if (
            not isinstance(feature_range, tuple)
            or len(feature_range) != 2
            or any(isinstance(v, bool) or not isinstance(v, Real) for v in feature_range)
        ):
            raise ValueError("feature_range must be a pair of real numbers")
        lower, upper = map(float, feature_range)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("feature_range must contain finite increasing bounds")
        self.feature_range = (lower, upper)
        self._fitted = False

    def fit(self, values: Iterable[Iterable[Real]]) -> "MinMaxScaler":
        matrix = _validated_matrix(values)
        self.n_features_in_ = len(matrix[0])
        self.data_min_ = [min(row[column] for row in matrix) for column in range(self.n_features_in_)]
        self.data_max_ = [max(row[column] for row in matrix) for column in range(self.n_features_in_)]
        self._fitted = True
        return self

    def transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        self._check_fitted()
        matrix = _validated_matrix(values)
        self._check_width(matrix)
        lower, upper = self.feature_range
        width = upper - lower
        result = []
        for row in matrix:
            scaled = []
            for index, value in enumerate(row):
                span = self.data_max_[index] - self.data_min_[index]
                scaled.append(lower if span == 0 else lower + (value - self.data_min_[index]) * width / span)
            result.append(scaled)
        return result

    def inverse_transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        self._check_fitted()
        matrix = _validated_matrix(values)
        self._check_width(matrix)
        lower, upper = self.feature_range
        width = upper - lower
        result = []
        for row in matrix:
            restored = []
            for index, value in enumerate(row):
                span = self.data_max_[index] - self.data_min_[index]
                restored.append(self.data_min_[index] if span == 0 else self.data_min_[index] + (value - lower) * span / width)
            result.append(restored)
        return result

    def fit_transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        matrix = _validated_matrix(values)
        self.fit(matrix)
        return self.transform(matrix)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("scaler must be fitted before transforming data")

    def _check_width(self, matrix: list[list[float]]) -> None:
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")


class StandardScaler:
    """Standardize features using training means and population standard deviations."""

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, values: Iterable[Iterable[Real]]) -> "StandardScaler":
        matrix = _validated_matrix(values)
        self.n_features_in_ = len(matrix[0])
        count = len(matrix)
        self.mean_ = [math.fsum(row[column] for row in matrix) / count for column in range(self.n_features_in_)]
        self.scale_ = []
        for column, average in enumerate(self.mean_):
            variance = math.fsum((row[column] - average) ** 2 for row in matrix) / count
            deviation = math.sqrt(variance)
            self.scale_.append(deviation if deviation else 1.0)
        self._fitted = True
        return self

    def transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        self._check_fitted()
        matrix = _validated_matrix(values)
        self._check_width(matrix)
        return [[(value - self.mean_[i]) / self.scale_[i] for i, value in enumerate(row)] for row in matrix]

    def inverse_transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        self._check_fitted()
        matrix = _validated_matrix(values)
        self._check_width(matrix)
        return [[value * self.scale_[i] + self.mean_[i] for i, value in enumerate(row)] for row in matrix]

    def fit_transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        matrix = _validated_matrix(values)
        self.fit(matrix)
        return self.transform(matrix)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("scaler must be fitted before transforming data")

    def _check_width(self, matrix: list[list[float]]) -> None:
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
