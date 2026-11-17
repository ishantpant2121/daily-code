# Regression Metrics

The dependency-free `regression_metrics.py` module provides mean absolute
error (MAE), mean squared error (MSE), root mean squared error (RMSE), median
absolute error, maximum error, R-squared, explained variance, and logarithmic
error metrics.

## Numerical behavior

All metrics require non-empty, equal-length inputs containing finite real
numbers. Boolean values are not accepted as numeric targets.

MAE, MSE, and RMSE normalize residuals before aggregation. This avoids
intermediate overflow when the final result is representable. MAE and MSE
raise `ValueError` when a required residual or final result exceeds the finite
floating-point range. RMSE is computed directly from scaled residuals, so it
can remain representable even when squaring a residual would underflow or the
corresponding MSE would exceed the finite range.

R-squared and explained variance normalize their variance calculations to avoid
overflow from large finite targets. Explained variance centers targets and
predictions separately; a constant prediction offset therefore does not
reduce explained variance, as required by its definition. These metrics retain
the `force_finite` policy for constant targets: perfect/constant predictions
return 1.0, imperfect/non-constant predictions return 0.0, or NaN and negative
infinity respectively when `force_finite=False`.

A score can still be negative infinity when its mathematical magnitude is
larger than the representable floating-point range. This is distinct from an
intermediate overflow and reflects an unrepresentable result.

## Example

```python
from regression_metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

actual = [3.0, -0.5, 2.0, 7.0]
predicted = [2.5, 0.0, 2.0, 8.0]

print(mean_absolute_error(actual, predicted))
print(mean_squared_error(actual, predicted))
print(root_mean_squared_error(actual, predicted))
print(r2_score(actual, predicted))
print(explained_variance_score(actual, predicted))
```

The metrics are intended for small educational workflows. For large-scale
model evaluation, use a maintained scientific-computing library with an API
and numerical implementation appropriate to the workload.
