# Binary probability calibration

`calibration_metrics.py` summarizes how predicted probabilities compare with
observed positive-class frequencies.

## Reliability curve

`calibration_curve(y_true, y_prob, n_bins=10, positive_label=1)` uses
equal-width bins over the interval [0, 1]. It omits empty bins and returns
`prob_true` (observed positive frequency), `prob_pred` (mean predicted
probability), and `bin_counts` (number of samples in each retained bin).

The final bin includes probability 1.0. A perfectly calibrated group has
observed frequency close to its mean predicted probability.

## Scalar errors

- `expected_calibration_error` computes the bin-count-weighted mean absolute
  gap between observed frequency and mean confidence.
- `maximum_calibration_error` returns the largest absolute gap among
  non-empty bins.

Both functions return values between 0 and 1 for valid probabilities. Smaller
values indicate closer agreement. ECE and MCE depend on the chosen bin count;
MCE can be especially sensitive to small bins. Report the binning choice and
sample counts alongside a score.

```python
from calibration_metrics import (
    calibration_curve,
    expected_calibration_error,
    maximum_calibration_error,
)

curve = calibration_curve(y_test, positive_class_probabilities, n_bins=10)
print(curve["bin_counts"])
print(expected_calibration_error(y_test, positive_class_probabilities, n_bins=10))
print(maximum_calibration_error(y_test, positive_class_probabilities, n_bins=10))
```

These helpers are for binary probability outputs. Map a multiclass probability
matrix to the chosen one-vs-rest positive class before calling them. They do
not calibrate or modify the estimator; they only evaluate its probabilities.
