# Paired model comparison

The `paired_model_comparison` module compares two sets of predictions on the
**same held-out samples**. Positive score differences mean model A scores better
than model B for the selected metric. The utilities resample fixed predictions;
they do not retrain either model.

## Paired bootstrap interval

The bootstrap draws sample indices with replacement and applies the same indices
to ground truth and both prediction arrays. It returns the observed score
difference, the mean bootstrap difference, and a percentile confidence interval.

```python
from paired_model_comparison import paired_bootstrap_score_difference

result = paired_bootstrap_score_difference(
    y_true,
    predictions_a,
    predictions_b,
    scoring="accuracy",
    n_bootstrap=1000,
    confidence_level=0.95,
    random_state=42,
)
print(result.observed_difference)
print(result.lower_bound, result.upper_bound)
```

## Paired permutation test

For each permutation, model predictions are swapped independently for each sample
with probability one-half. The two-sided p-value compares the absolute
permuted score differences with the observed absolute difference and applies
the plus-one correction. The test assumes the two models' predictions are
exchangeable under the null hypothesis.

```python
from paired_model_comparison import paired_permutation_test

result = paired_permutation_test(
    y_true,
    predictions_a,
    predictions_b,
    scoring="accuracy",
    n_permutations=1000,
    random_state=42,
)
print(result.observed_difference, result.p_value)
```

## Scoring

Built-in scorers are `accuracy`, `balanced_accuracy`, `r2`,
`neg_mean_squared_error`, `neg_mean_absolute_error`, and
`neg_root_mean_squared_error`. A custom callable receives
`(y_true, y_pred)` and must return a finite real score; larger scores are
better. The negative regression-error names follow the common convention that
higher scores are better.

## Interpretation and limitations

- Use predictions from the same test rows for both models; pairing is essential.
- Keep the test set separate from training and model selection.
- The bootstrap interval is a percentile interval over paired resamples and is
  approximate, especially for small test sets.
- The permutation test's exchangeability assumption is not guaranteed merely
  because both models were evaluated on the same data.
- Neither procedure retrains models, so these results quantify uncertainty or
  evidence conditional on the supplied predictions, not the full training
  pipeline.
- A small p-value does not establish practical importance; inspect the observed
  difference and its interval as well.
- Results are reproducible when `random_state` is an integer.
