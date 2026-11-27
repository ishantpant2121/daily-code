# Wilson score intervals for proportions

Use `wilson_score_interval(successes, trials)` to report uncertainty around an
observed binomial proportion. It returns a two-sided, uncorrected Wilson score
interval and uses Python's standard-library normal quantile implementation.

```python
from proportion_intervals import wilson_score_interval

covered = 87
evaluated = 100
lower, upper = wilson_score_interval(covered, evaluated, confidence_level=0.95)
print(f"Observed coverage: {covered / evaluated:.1%}")
print(f"95% Wilson interval: [{lower:.1%}, {upper:.1%}]")
```

The interval is asymmetric and remains informative when the observed count is
zero or equals the number of trials. This avoids the zero-width endpoint
behavior of the simple normal/Wald interval.

## Assumptions and interpretation

- Each observation is a Bernoulli trial with the same success probability.
- Trials are independent; grouped, clustered, weighted, or repeated-measures
  observations need a method that accounts for their sampling design.
- The returned interval is an approximate score interval, not an exact
  Clopper-Pearson interval.
- A nominal 95% confidence procedure describes long-run coverage under its
  model; it does not mean there is a 95% probability that a fixed parameter
  lies inside this already-computed interval.

For prediction-interval evaluation, pass the number of covered held-out
targets as successes and the number evaluated as trials. The result quantifies
sampling uncertainty in empirical coverage; it does not establish conditional
coverage or repair distribution shift.
