# Regression prediction-interval diagnostics

The dependency-free `interval_diagnostics` module evaluates prediction
intervals on a held-out dataset. It reports empirical coverage, mean and median
width, counts below/inside/above the interval, and mean Winkler interval score.

```python
from interval_diagnostics import interval_diagnostics

report = interval_diagnostics(
    targets=[1.2, 2.1, 3.8, 4.2],
    intervals=[(0.5, 1.5), (1.5, 2.5), (3.0, 4.0), (3.5, 4.5)],
    alpha=0.1,
    groups=["north", "north", "south", "south"],
)
print(report.overall.coverage)
print(report.by_group["north"].mean_width)
```

The Winkler score is interval width plus a penalty of 2/alpha for each unit
outside either endpoint. Lower scores are better when comparing intervals at
the same nominal alpha. Unbounded intervals are accepted and have infinite
width and score.

Group summaries are descriptive diagnostics, not conditional coverage
guarantees. Small groups produce noisy estimates. Do not tune a model or
interval procedure on the final test set used for reporting. The nominal
miscoverage alpha must be strictly between zero and one.

Run the focused tests with:

```bash
python -m unittest tests.test_interval_diagnostics -v
```
