# Model inspection and learning diagnostics

The `feature_importance` and `model_diagnostics` modules add model-agnostic
inspection helpers for estimators that implement the small fit/predict/score
interface used in this repository.

## Permutation importance

`permutation_importance` requires an already-fitted estimator. It measures the
decrease in a selected score after shuffling one feature column at a time.
Results include the baseline score, raw score drops for every repeat, the mean
and population standard deviation per feature, and n_samples_used. Built-in
scorers cover classification (accuracy, balanced_accuracy) and regression
(r2, neg_mean_squared_error, neg_mean_absolute_error, and
neg_root_mean_squared_error); a custom scorer can be supplied as a callable.

```python
from feature_importance import permutation_importance

result = permutation_importance(
    fitted_model,
    x_validation,
    y_validation,
    scoring="accuracy",
    n_repeats=30,
    random_state=7,
)
print(result.importances_mean)
```

For larger evaluation datasets, max_samples accepts either an integer row
count or a fraction in (0, 1]. Fractions are rounded up to at least one row.
One deterministic subset is selected and reused for the baseline and all
permutations, and its size is returned as n_samples_used. The same API works
with regression estimators:

```python
from feature_importance import permutation_importance

result = permutation_importance(
    fitted_regressor,
    x_validation,
    y_validation,
    scoring="r2",
    n_repeats=20,
    max_samples=0.75,
    random_state=7,
)
print(result.importances_mean, result.n_samples_used)
```

Use held-out data when the goal is to understand reliance that generalizes.
Correlated features can mask one another: shuffling one may have little effect
when another feature carries similar information. Negative importance is
possible when a permutation happens to improve the score.

## Learning curves

`learning_curve` reports train and validation scores for increasing absolute
training sizes. Fractional sizes are converted to counts using the smallest
training partition, so every fold uses the same number of training rows.
Smaller subsets are prefixes of the fold's training indices. When rows are
ordered, shuffle the folds or pre-shuffle the data to avoid biased prefixes.

The result contains `train_sizes`, `train_scores`, and `test_scores`; score
arrays are arranged by training size and then by fold.

## Validation curves

`validation_curve` varies one estimator parameter and evaluates every value
using identical cross-validation settings. Direct estimator attributes use
their attribute name; pipeline step parameters use `step__attribute` notation.
It returns `param_range`, `train_scores`, and `test_scores`. The original
estimator is not mutated.

## Bootstrap score interval

`bootstrap_score` resamples paired evaluation rows with replacement and scores
the already-fitted estimator. It returns the replicate scores, their mean, and
a percentile confidence interval. The estimator is not refitted: the interval
describes evaluation-sample variability conditional on the fitted model. Use a
separate held-out dataset, and do not interpret this interval as uncertainty
from retraining the full modeling process.
