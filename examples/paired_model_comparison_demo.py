"""Compare two fixed prediction arrays using paired resampling tests."""

from paired_model_comparison import (
    paired_bootstrap_score_difference,
    paired_permutation_test,
)

actual = [0, 0, 1, 1, 1, 0, 1, 0, 1, 0]
model_a = [0, 0, 1, 1, 0, 0, 1, 0, 1, 0]
model_b = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

bootstrap = paired_bootstrap_score_difference(
    actual,
    model_a,
    model_b,
    scoring="accuracy",
    n_bootstrap=1000,
    confidence_level=0.95,
    random_state=42,
)
permutation = paired_permutation_test(
    actual,
    model_a,
    model_b,
    scoring="accuracy",
    n_permutations=1000,
    random_state=42,
)

print("Observed accuracy difference (A - B):", bootstrap.observed_difference)
print("Bootstrap mean difference:", bootstrap.mean_difference)
print("95% paired bootstrap interval:", bootstrap.lower_bound, bootstrap.upper_bound)
print("Paired permutation p-value:", permutation.p_value)
