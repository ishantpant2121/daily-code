"""Report empirical prediction-interval coverage with Wilson uncertainty."""
from interval_diagnostics import interval_diagnostics
from proportion_intervals import wilson_score_interval

targets = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
intervals = [
    (0.5, 1.5), (1.5, 2.5), (2.5, 3.5),
    (3.5, 4.5), (4.5, 5.5), (5.5, 6.5),
]
summary = interval_diagnostics(targets, intervals).overall
lower, upper = wilson_score_interval(summary.covered, summary.count)

print(f"Empirical coverage: {summary.coverage:.1%}")
print(f"95% Wilson interval: [{lower:.1%}, {upper:.1%}]")
print("Interpretation assumes independent, identically distributed test rows.")
