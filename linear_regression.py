"""Simple univariate ordinary least-squares regression without dependencies."""

import math
from numbers import Real
from typing import Iterable

from regression_metrics import r2_score


def _numeric_values(values: Iterable[Real], name: str) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must contain only real numbers")
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError):
            raise ValueError(f"{name} must contain only finite numbers") from None
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite numbers")
        result.append(number)
    return result


class SimpleLinearRegression:
    """Fit y = slope * x + intercept by minimizing squared residuals."""

    def fit(self, x_values: Iterable[Real], y_values: Iterable[Real]) -> "SimpleLinearRegression":
        x = _numeric_values(x_values, "x_values")
        y = _numeric_values(y_values, "y_values")
        if len(x) != len(y):
            raise ValueError("x_values and y_values must have equal lengths")
        if len(x) < 2:
            raise ValueError("at least two samples are required")
        x_mean = math.fsum(x) / len(x)
        y_mean = math.fsum(y) / len(y)
        denominator = math.fsum((value - x_mean) ** 2 for value in x)
        if denominator == 0:
            raise ValueError("x_values must contain at least two distinct values")
        numerator = math.fsum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x, y))
        self.coef_ = numerator / denominator
        self.intercept_ = y_mean - self.coef_ * x_mean
        self.n_features_in_ = 1
        self._fitted = True
        return self

    def predict(self, x_values: Iterable[Real]) -> list[float]:
        self._check_fitted()
        x = _numeric_values(x_values, "x_values")
        return [self.coef_ * value + self.intercept_ for value in x]

    def score(self, x_values: Iterable[Real], y_values: Iterable[Real]) -> float:
        """Return R-squared on the supplied evaluation samples."""
        actual = _numeric_values(y_values, "y_values")
        predicted = self.predict(x_values)
        return r2_score(actual, predicted)

    def _check_fitted(self) -> None:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("model must be fitted before prediction")
