# Probability calibration

Probability calibration compares predicted confidence with observed outcomes. These
helpers evaluate probabilities; they do not fit a calibrator or modify a model.
Evaluate on held-out data rather than the samples used to fit the estimator.

## Binary reliability curves

`calibration_curve(y_true, y_prob, n_bins=10, positive_label=1, strategy="uniform")`
returns observed positive frequency (`prob_true`), mean predicted probability
(`prob_pred`), and sample counts (`bin_counts`) for non-empty bins.

- `strategy="uniform"` divides [0, 1] into equal-width bins. The final bin includes 1.0.
- `strategy="quantile"` uses observed probability boundaries to create
  approximately equal-frequency bins. Equal probabilities stay together, so
  ties can make bin sizes uneven or leave bins empty.

`expected_calibration_error` is the count-weighted mean absolute gap.
`maximum_calibration_error` is the largest absolute gap in any non-empty bin.
Both accept the same `strategy` option. ECE and MCE depend on the bin count and
binning strategy; report those settings and bin counts alongside the score.

## Multiclass top-label calibration

For multiclass probabilities, pass a matrix with one row per sample and columns
in the exact order of the supplied `labels`. Every row must contain finite
probabilities in [0, 1] and sum to one.

`multiclass_calibration_curve` bins the largest class probability for each
sample. `prob_pred` is the mean top-class confidence in a bin, while
`prob_true` is the fraction of those top-class predictions that are correct.
The scalar helpers provide three norms of the same top-label gaps:

- `multiclass_expected_calibration_error`: count-weighted L1 gap.
- `multiclass_maximum_calibration_error`: largest absolute gap.
- `multiclass_root_mean_squared_calibration_error`: count-weighted RMS gap.

All accept `strategy="uniform"` or `strategy="quantile"`. These are top-label
metrics: they assess confidence in the winning class, not every class column
independently.

For class-specific one-vs-rest diagnostics, `classwise_calibration_curves`
returns a reliability curve for each class, preserving the supplied label order.
Each curve compares that class's probability column with whether the true label
matches that class.

## Multiclass probability losses

`multiclass_brier_score_loss` computes the mean, per sample, of the sum of
squared errors across all class probability columns (the unhalved multiclass
definition). `multiclass_log_loss` computes the mean negative log probability
of the true class. It clips probabilities below `eps`, then renormalizes each
row before scoring. Both require an explicit `labels` sequence that defines
the probability-column order.

## Example

```python
from calibration_metrics import (
    classwise_calibration_curves,
    multiclass_calibration_curve,
    multiclass_expected_calibration_error,
    multiclass_maximum_calibration_error,
    multiclass_root_mean_squared_calibration_error,
)
from classification_metrics import multiclass_brier_score_loss, multiclass_log_loss

curve = multiclass_calibration_curve(
    y_test, probabilities, labels=model.classes_, n_bins=10, strategy="quantile"
)
print(curve["bin_counts"])
print(multiclass_expected_calibration_error(
    y_test, probabilities, labels=model.classes_, n_bins=10
))
print(multiclass_maximum_calibration_error(
    y_test, probabilities, labels=model.classes_, n_bins=10
))
print(multiclass_root_mean_squared_calibration_error(
    y_test, probabilities, labels=model.classes_, n_bins=10
))
print(classwise_calibration_curves(
    y_test, probabilities, labels=model.classes_, n_bins=10
))
print(multiclass_brier_score_loss(
    y_test, probabilities, labels=model.classes_
))
print(multiclass_log_loss(
    y_test, probabilities, labels=model.classes_
))
```

A lower Brier or log loss is better, but these proper scoring rules reflect
calibration and discrimination together; they are not pure calibration-only
measures. Small datasets can produce unstable bins and estimates. The utilities
are dependency-free educational tools, not substitutes for a dedicated
calibration workflow on production-scale data.
