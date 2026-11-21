# Split conformal prediction for regression

Split conformal regression wraps an existing point predictor to create symmetric
prediction intervals. It is dependency-free and does not train the underlying
model.

## Workflow and guarantee

1. Split the data into training, calibration, and final test portions.
2. Fit the predictor only on the training portion.
3. Predict the calibration rows and call
   SplitConformalRegressor(alpha).fit(y_calibration, calibration_predictions).
4. Predict test rows with the fitted model, then pass those predictions to the
   conformal object's predict method.
5. Report test coverage and mean interval width with the provided helpers.

With absolute-residual scores, the threshold is the
ceil((n_cal + 1) * (1 - alpha))-th smallest calibration residual. If the
rank is greater than the calibration sample count, the threshold is infinity;
small calibration sets can therefore produce unbounded intervals.

The finite-sample marginal coverage guarantee is at least 1 - alpha under
exchangeability of calibration and future test examples. This is marginal
coverage, not a guarantee conditional on every feature value. Calibration rows
must not be used to fit or tune the predictor.

## API

- SplitConformalRegressor(alpha=0.1): alpha is a miscoverage rate strictly
  between zero and one.
- fit(calibration_targets, calibration_predictions): consumes paired, finite,
  non-empty iterable values. A failed fit clears previous calibration results.
- predict(predictions): returns inclusive (lower, upper) tuples. Input
  predictions must be finite and non-empty.
- empirical_coverage(targets, intervals): fraction of targets inside their
  corresponding intervals; endpoints are inclusive.
- mean_interval_width(intervals): average width. Returns infinity if any
  interval is unbounded or its finite endpoints produce an unrepresentable width.

Calibration and prediction input may be one-shot iterables. Booleans, strings,
NaN, infinity in target/prediction data, empty inputs, and mismatched lengths
are rejected. Interval endpoints may be infinite but never NaN or reversed.

## Example

Run python examples/split_conformal_regression.py. The example uses a training
split, a separate calibration split, and a test split. It reports test coverage
and mean interval width; those metrics are descriptive and do not themselves
establish the theoretical coverage guarantee.

## References

- Lei, G'Sell, Rinaldo, Tibshirani, and Wasserman, Distribution-Free
  Predictive Inference For Regression (2016):
  https://arxiv.org/abs/1604.04173
- Algorithm 2, split conformal prediction:
  https://www.stat.berkeley.edu/~ryantibs/papers/conformal.pdf
