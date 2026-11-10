"""Simple, dependency-free imputation for rectangular feature matrices."""

import math
from numbers import Real
from typing import Any, Iterable


def _matrix(values: Iterable[Iterable[Any]]) -> list[list[Any]]:
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


class SimpleImputer:
    """Replace missing values using per-column training statistics.

    None and NaN are always considered missing. missing_values adds one
    optional equality-based sentinel. For mean and median strategies, observed
    values must be finite real numbers. Most-frequent imputation supports
    hashable categorical values and resolves ties by first appearance.

    Empty columns are dropped by default for statistical strategies. Set
    keep_empty_features=True to retain them with a zero fill value.
    Constant strategy always retains empty columns.
    """

    def __init__(
        self,
        *,
        strategy: str = "mean",
        missing_values: Any = float("nan"),
        fill_value: Any = 0,
        keep_empty_features: bool = False,
    ) -> None:
        if strategy not in ("mean", "median", "most_frequent", "constant"):
            raise ValueError("strategy must be mean, median, most_frequent, or constant")
        if type(keep_empty_features) is not bool:
            raise ValueError("keep_empty_features must be a boolean")
        self.strategy = strategy
        self.missing_values = missing_values
        self.fill_value = fill_value
        self.keep_empty_features = keep_empty_features
        self._fitted = False

    def _is_missing(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, Real) and not isinstance(value, bool):
            try:
                if math.isnan(float(value)):
                    return True
            except (OverflowError, TypeError, ValueError):
                pass
        try:
            return bool(value == self.missing_values)
        except Exception:
            return False

    @staticmethod
    def _numeric(values: list[Any]) -> list[float]:
        converted = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("mean and median strategies require real numeric values")
            try:
                number = float(value)
            except (OverflowError, TypeError, ValueError):
                raise ValueError("numeric values must be finite") from None
            if not math.isfinite(number):
                raise ValueError("numeric values must be finite")
            converted.append(number)
        return converted

    def _statistic(self, observed: list[Any]) -> Any:
        if self.strategy == "constant":
            return self.fill_value
        if self.strategy == "most_frequent":
            counts: dict[Any, int] = {}
            try:
                for value in observed:
                    counts[value] = counts.get(value, 0) + 1
            except TypeError:
                raise ValueError("most-frequent imputation requires hashable values") from None
            return max(counts, key=counts.__getitem__)

        numeric = self._numeric(observed)
        if self.strategy == "mean":
            try:
                result = math.fsum(value / len(numeric) for value in numeric)
            except OverflowError:
                raise ValueError("mean exceeds finite float range") from None
        else:
            ordered = sorted(numeric)
            middle = len(ordered) // 2
            if len(ordered) % 2:
                result = ordered[middle]
            else:
                result = ordered[middle - 1] / 2.0 + ordered[middle] / 2.0
        if not math.isfinite(result):
            raise ValueError("imputation statistic exceeds finite float range")
        return result

    def fit(self, features: Iterable[Iterable[Any]]) -> "SimpleImputer":
        rows = _matrix(features)
        self.n_features_in_ = len(rows[0])
        statistics = []
        support = []
        for column in range(self.n_features_in_):
            observed = [row[column] for row in rows if not self._is_missing(row[column])]
            if not observed:
                retain = self.keep_empty_features or self.strategy == "constant"
                support.append(retain)
                statistics.append(
                    self.fill_value if self.strategy == "constant" else 0.0
                )
                continue
            support.append(True)
            statistics.append(self._statistic(observed))

        self.statistics_ = tuple(statistics)
        self.support_ = tuple(support)
        self.feature_indices_ = tuple(
            index for index, retained in enumerate(support) if retained
        )
        self.n_features_out_ = len(self.feature_indices_)
        self._fitted = True
        return self

    def transform(self, features: Iterable[Iterable[Any]]) -> list[list[Any]]:
        self._check_fitted()
        rows = _matrix(features)
        if len(rows[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        transformed = []
        for row in rows:
            transformed.append([
                self.statistics_[index] if self._is_missing(row[index]) else row[index]
                for index in self.feature_indices_
            ])
        return transformed

    def fit_transform(self, features: Iterable[Iterable[Any]]) -> list[list[Any]]:
        rows = _matrix(features)
        self.fit(rows)
        return self.transform(rows)

    def get_support(self, *, indices: bool = False) -> list[bool] | list[int]:
        self._check_fitted()
        if type(indices) is not bool:
            raise ValueError("indices must be a boolean")
        if indices:
            return list(self.feature_indices_)
        return list(self.support_)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before transforming data")
