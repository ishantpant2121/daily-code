"""Small dependency-free feature scalers and polynomial expansion."""

import math
from itertools import combinations_with_replacement
from numbers import Real
from typing import Any, Iterable

from statistics_utils import _scaled_centered, variance as numeric_variance


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
            try:
                number = float(value)
            except (OverflowError, TypeError, ValueError):
                raise ValueError("features must be finite real numbers") from None
            if not math.isfinite(number):
                raise ValueError("features must be finite real numbers")
            converted.append(number)
        result.append(converted)
    return result


class MinMaxScaler:
    """Scale each feature to a target range using training-set extrema."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        if (
            not isinstance(feature_range, tuple)
            or len(feature_range) != 2
            or any(isinstance(v, bool) or not isinstance(v, Real) for v in feature_range)
        ):
            raise ValueError("feature_range must be a pair of real numbers")
        try:
            lower, upper = map(float, feature_range)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("feature_range must contain finite increasing bounds") from None
        width = upper - lower
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or not math.isfinite(width)
            or lower >= upper
        ):
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
                minimum = self.data_min_[index]
                maximum = self.data_max_[index]
                if maximum == minimum:
                    scaled.append(lower)
                    continue
                span = maximum - minimum
                offset = value - minimum
                if math.isfinite(span) and math.isfinite(offset):
                    fraction = offset / span
                else:
                    scale = max(abs(value), abs(minimum), abs(maximum))
                    normalized_minimum = minimum / scale
                    normalized_span = maximum / scale - normalized_minimum
                    if normalized_span == 0.0:
                        raise ValueError("feature range is too small to scale safely")
                    fraction = (value / scale - normalized_minimum) / normalized_span
                transformed = lower + fraction * width
                if not math.isfinite(transformed):
                    raise ValueError("scaled feature exceeds finite float range")
                scaled.append(transformed)
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
                minimum = self.data_min_[index]
                maximum = self.data_max_[index]
                if maximum == minimum:
                    restored.append(minimum)
                    continue
                offset = value - lower
                if math.isfinite(offset):
                    fraction = offset / width
                else:
                    range_scale = max(abs(value), abs(lower), abs(upper))
                    normalized_width = upper / range_scale - lower / range_scale
                    if normalized_width == 0.0:
                        raise ValueError("feature range is too small to invert safely")
                    fraction = (value / range_scale - lower / range_scale) / normalized_width
                data_span = maximum - minimum
                if math.isfinite(data_span):
                    original = minimum + fraction * data_span
                else:
                    data_scale = max(abs(minimum), abs(maximum))
                    normalized_minimum = minimum / data_scale
                    normalized_span = maximum / data_scale - normalized_minimum
                    normalized_value = normalized_minimum + fraction * normalized_span
                    original = normalized_value * data_scale
                if not math.isfinite(original):
                    raise ValueError("inverse-scaled feature exceeds finite float range")
                restored.append(original)
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
        self.mean_ = []
        self.scale_ = []
        self._feature_scales_ = []
        self._normalized_means_ = []
        self._normalized_stds_ = []
        self._constant_features_ = []
        for column in range(self.n_features_in_):
            values_for_column = [row[column] for row in matrix]
            feature_scale, _, normalized_mean, sum_squares = _scaled_centered(
                values_for_column
            )
            average = feature_scale * normalized_mean
            normalized_std = math.sqrt(sum_squares / count)
            constant = normalized_std == 0.0
            if constant:
                actual_scale = 1.0
                normalized_std = 1.0
            else:
                actual_scale = feature_scale * normalized_std
                if not math.isfinite(actual_scale):
                    raise ValueError("feature standard deviation exceeds finite float range")
            self.mean_.append(average)
            self.scale_.append(actual_scale)
            self._feature_scales_.append(feature_scale if feature_scale else 1.0)
            self._normalized_means_.append(normalized_mean)
            self._normalized_stds_.append(normalized_std)
            self._constant_features_.append(constant)
        self._fitted = True
        return self

    def transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        self._check_fitted()
        matrix = _validated_matrix(values)
        self._check_width(matrix)
        result = []
        for row in matrix:
            transformed = []
            for index, value in enumerate(row):
                if self._constant_features_[index]:
                    scaled = value - self.mean_[index]
                    if not math.isfinite(scaled):
                        normalized = (
                            value / self._feature_scales_[index]
                            - self._normalized_means_[index]
                        )
                        scaled = normalized * self._feature_scales_[index]
                    if not math.isfinite(scaled):
                        raise ValueError("standardized feature exceeds finite float range")
                    transformed.append(scaled)
                    continue
                normalized = value / self._feature_scales_[index]
                scaled = (
                    normalized - self._normalized_means_[index]
                ) / self._normalized_stds_[index]
                if not math.isfinite(scaled):
                    raise ValueError("standardized feature exceeds finite float range")
                transformed.append(scaled)
            result.append(transformed)
        return result

    def inverse_transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        self._check_fitted()
        matrix = _validated_matrix(values)
        self._check_width(matrix)
        result = []
        for row in matrix:
            restored = []
            for index, value in enumerate(row):
                if self._constant_features_[index]:
                    original = value + self.mean_[index]
                    if not math.isfinite(original):
                        raise ValueError("inverse-standardized feature exceeds finite float range")
                    restored.append(original)
                    continue
                normalized = (
                    value * self._normalized_stds_[index]
                    + self._normalized_means_[index]
                )
                original = normalized * self._feature_scales_[index]
                if not math.isfinite(original):
                    raise ValueError("inverse-standardized feature exceeds finite float range")
                restored.append(original)
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


def _linear_quantile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return math.fsum((
        sorted_values[lower] * (1.0 - weight),
        sorted_values[upper] * weight,
    ))


class RobustScaler:
    """Scale features by median and inter-quantile range."""

    def __init__(self, quantile_range: tuple[float, float] = (25.0, 75.0)) -> None:
        if (
            not isinstance(quantile_range, tuple)
            or len(quantile_range) != 2
            or any(isinstance(value, bool) or not isinstance(value, Real) for value in quantile_range)
        ):
            raise ValueError("quantile_range must be a pair of real numbers")
        lower, upper = map(float, quantile_range)
        if not math.isfinite(lower) or not math.isfinite(upper) or not 0 <= lower < upper <= 100:
            raise ValueError("quantile_range must satisfy 0 <= lower < upper <= 100")
        self.quantile_range = (lower, upper)
        self._fitted = False

    def fit(self, values: Iterable[Iterable[Real]]) -> "RobustScaler":
        matrix = _validated_matrix(values)
        self.n_features_in_ = len(matrix[0])
        lower, upper = (value / 100.0 for value in self.quantile_range)
        columns = [sorted(row[index] for row in matrix) for index in range(self.n_features_in_)]
        self.center_ = [_linear_quantile(column, 0.5) for column in columns]
        self.scale_ = []
        for column in columns:
            spread = _linear_quantile(column, upper) - _linear_quantile(column, lower)
            if not math.isfinite(spread):
                raise ValueError("feature spread is too large to represent")
            self.scale_.append(spread if spread else 1.0)
        self._fitted = True
        return self

    def transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        self._check_fitted()
        matrix = _validated_matrix(values)
        self._check_width(matrix)
        return [[(value - self.center_[index]) / self.scale_[index] for index, value in enumerate(row)] for row in matrix]

    def inverse_transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        self._check_fitted()
        matrix = _validated_matrix(values)
        self._check_width(matrix)
        return [[value * self.scale_[index] + self.center_[index] for index, value in enumerate(row)] for row in matrix]

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


class PolynomialFeatures:
    """Expand rows into polynomial powers and interaction terms.

    Terms are ordered by increasing total degree, then combinations with
    replacement of feature indices. A leading constant-one term is included
    when include_bias is true.
    """

    def __init__(self, degree: int = 2, *, include_bias: bool = True) -> None:
        if type(degree) is not int or degree < 1:
            raise ValueError("degree must be a positive integer")
        if type(include_bias) is not bool:
            raise ValueError("include_bias must be a boolean")
        self.degree = degree
        self.include_bias = include_bias
        self._fitted = False

    def fit(self, values: Iterable[Iterable[Real]]) -> "PolynomialFeatures":
        matrix = _validated_matrix(values)
        self.n_features_in_ = len(matrix[0])
        powers = [()] if self.include_bias else []
        for degree in range(1, self.degree + 1):
            powers.extend(combinations_with_replacement(range(self.n_features_in_), degree))
        self.powers_ = tuple(powers)
        self._fitted = True
        return self

    def transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        self._check_fitted()
        matrix = _validated_matrix(values)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        transformed = []
        for row in matrix:
            expanded = []
            for powers in self.powers_:
                product = 1.0
                for index in powers:
                    product *= row[index]
                if not math.isfinite(product):
                    raise ValueError("polynomial feature exceeds finite float range")
                expanded.append(product)
            transformed.append(expanded)
        return transformed

    def fit_transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        matrix = _validated_matrix(values)
        self.fit(matrix)
        return self.transform(matrix)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("transform requires a fitted PolynomialFeatures instance")


class VarianceThreshold:
    """Remove features whose training-set population variance is <= threshold."""

    def __init__(self, threshold: Real = 0.0) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise ValueError("threshold must be a finite non-negative real number")
        try:
            value = float(threshold)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("threshold must be a finite non-negative real number") from None
        if not math.isfinite(value) or value < 0:
            raise ValueError("threshold must be a finite non-negative real number")
        self.threshold = value
        self._fitted = False

    def fit(self, values: Iterable[Iterable[Real]]) -> "VarianceThreshold":
        matrix = _validated_matrix(values)
        count = len(matrix)
        self.n_features_in_ = len(matrix[0])
        self.variances_ = []
        for column in range(self.n_features_in_):
            values_for_column = [row[column] for row in matrix]
            feature_variance = numeric_variance(
                values_for_column, sample=False
            )
            self.variances_.append(feature_variance)
        self.support_ = tuple(value > self.threshold for value in self.variances_)
        if not any(self.support_):
            raise ValueError("no features meet the variance threshold")
        self._fitted = True
        return self

    def transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        self._check_fitted()
        matrix = _validated_matrix(values)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        return [
            [value for index, value in enumerate(row) if self.support_[index]]
            for row in matrix
        ]

    def fit_transform(self, values: Iterable[Iterable[Real]]) -> list[list[float]]:
        matrix = _validated_matrix(values)
        self.fit(matrix)
        return self.transform(matrix)

    def get_support(self, *, indices: bool = False) -> list[bool] | list[int]:
        self._check_fitted()
        if type(indices) is not bool:
            raise ValueError("indices must be a boolean")
        if indices:
            return [index for index, selected in enumerate(self.support_) if selected]
        return list(self.support_)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before transforming data")



def _categorical_matrix(values: Iterable[Iterable[Any]]) -> list[list[Any]]:
    rows = [list(row) for row in values]
    if not rows or not rows[0]:
        raise ValueError("input must contain at least one row and one feature")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all rows must contain the same number of features")
    try:
        for row in rows:
            for value in row:
                hash(value)
    except TypeError:
        raise ValueError("categorical values must be hashable") from None
    return rows


class OneHotEncoder:
    """Encode categorical columns as indicator features.

    Category order follows first appearance in the fitted data. Unknown
    categories either raise ValueError or produce an all-zero feature block
    when handle_unknown="ignore".
    """

    def __init__(self, *, handle_unknown: str = "error") -> None:
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'")
        self.handle_unknown = handle_unknown
        self._fitted = False

    def fit(self, values: Iterable[Iterable[Any]]) -> "OneHotEncoder":
        matrix = _categorical_matrix(values)
        self.n_features_in_ = len(matrix[0])
        categories = []
        indices = []
        for column in range(self.n_features_in_):
            mapping = {}
            for row in matrix:
                value = row[column]
                if value not in mapping:
                    mapping[value] = len(mapping)
            categories.append(tuple(mapping))
            indices.append(mapping)
        self.categories_ = tuple(categories)
        self._category_indices_ = tuple(indices)
        self._fitted = True
        return self

    def transform(self, values: Iterable[Iterable[Any]]) -> list[list[int]]:
        self._check_fitted()
        matrix = _categorical_matrix(values)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        transformed = []
        for row in matrix:
            encoded = []
            for column, value in enumerate(row):
                mapping = self._category_indices_[column]
                if value not in mapping:
                    if self.handle_unknown == "error":
                        raise ValueError(f"unknown category {value!r} in feature {column}")
                    encoded.extend([0] * len(self.categories_[column]))
                    continue
                block = [0] * len(self.categories_[column])
                block[mapping[value]] = 1
                encoded.extend(block)
            transformed.append(encoded)
        return transformed

    def fit_transform(self, values: Iterable[Iterable[Any]]) -> list[list[int]]:
        matrix = _categorical_matrix(values)
        self.fit(matrix)
        return self.transform(matrix)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before transforming data")



class LabelEncoder:
    """Map hashable class labels to stable integer indices and back.

    Class order follows first appearance in the fitted target sequence.
    Unknown labels and invalid inverse indices raise ValueError.
    """

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, labels: Iterable[Any]) -> "LabelEncoder":
        values = list(labels)
        if not values:
            raise ValueError("labels must not be empty")
        mapping = {}
        try:
            for value in values:
                if value not in mapping:
                    mapping[value] = len(mapping)
        except TypeError:
            raise ValueError("class labels must be hashable") from None
        self.classes_ = tuple(mapping)
        self._class_indices = mapping
        self._fitted = True
        return self

    def transform(self, labels: Iterable[Any]) -> list[int]:
        self._check_fitted()
        result = []
        for value in labels:
            try:
                result.append(self._class_indices[value])
            except (KeyError, TypeError):
                raise ValueError(f"unknown or unhashable class label: {value!r}") from None
        return result

    def inverse_transform(self, indices: Iterable[int]) -> list[Any]:
        self._check_fitted()
        result = []
        for index in indices:
            if type(index) is not int or not 0 <= index < len(self.classes_):
                raise ValueError("encoded labels must be valid integer class indices")
            result.append(self.classes_[index])
        return result

    def fit_transform(self, labels: Iterable[Any]) -> list[int]:
        values = list(labels)
        self.fit(values)
        return self.transform(values)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before transforming labels")
