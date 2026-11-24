# Finite hyperparameter search

The search utilities in `hyperparameter_search.py` work with finite mappings
from parameter names to non-string sequences.

## ParameterGrid

`ParameterGrid` sorts parameter names so candidate order is deterministic.
Iteration follows the same Cartesian-product order as
[`itertools.product`](https://docs.python.org/3/library/itertools.html#itertools.product):
the rightmost sorted parameter changes fastest.

Indexed access uses mixed-radix decomposition rather than walking through all
preceding candidates. If a grid has (p) parameter names, retrieving one
candidate takes (O(p)) indexing and integer-division operations after the
grid size is known. Negative indices are supported.

## ParameterSampler

`ParameterSampler` samples distinct integer positions from
`range(len(grid))`, then resolves only the selected positions through
`ParameterGrid.__getitem__`. It does not first build a list containing every
candidate.

For (N) combinations, (p) parameter names, and (k) requested samples:

- The parameter value sequences are retained as tuples, proportional to the
  supplied choices.
- Sampling stores (k) integer positions rather than (N) candidate
  dictionaries.
- Iterating the sampler materializes only the (k) selected dictionaries.

The candidate-related memory therefore scales with the requested sample count,
approximately (O(kp)), instead of (O(Np)) for materializing the full
Cartesian product. The search still evaluates each selected candidate across
the requested cross-validation folds; this optimization reduces candidate
construction and storage, not model-fitting cost.

Python documents that `random.sample(range(...), k)` is especially fast and
space-efficient for large populations:
[Python random.sample documentation](https://docs.python.org/3/library/random.html#random.sample).

## Example: sample from one million combinations

```python
from hyperparameter_search import ParameterSampler

parameters = {
    "max_depth": range(100),
    "min_samples_leaf": range(100),
    "max_features": range(100),
}

sampler = ParameterSampler(parameters, n_iter=8, random_state=42)
for candidate in sampler:
    print(candidate)
```

This grid describes (100^3 = 1{,}000{,}000) possible combinations, while
the sampler retains only eight sampled positions and materializes eight
candidate dictionaries as it is iterated.

## Constraints

- Parameter values must be finite, non-string sequences.
- Sampling is without replacement.
- `n_iter` must be positive and cannot exceed the grid size.
- The random seed makes the selected positions reproducible for a given
  Python implementation and version.
- This API does not yet accept continuous distributions or objects exposing
  an `rvs` method.

## Reusing one-shot cross-validation iterators

`GridSearchCV` and `RandomizedSearchCV` accept explicit fold
iterables. A generator or other one-shot iterator is consumed when its folds
are first validated. The search object snapshots those validated index pairs
and reuses the same folds on later `fit()` calls, rather than trying to read an
exhausted iterator. This makes repeated fitting reproducible and keeps the
validation partitions identical across refits.

If you replace the search object's `cv` attribute with a different object, the
cached snapshot is discarded and the new fold source is validated. Reusable
splitter objects and ordinary lists are read again on each fit.

## Refit after failure

Both `GridSearchCV` and `RandomizedSearchCV` clear results from their previous
successful fit at the start of a new `fit()` attempt. Candidate scores and the
selected parameters are kept local while cross-validation runs. The selected
estimator is then refitted on the full input data; search-result attributes are
published only after this final refit succeeds.

If input validation, candidate evaluation, parameter assignment, or the final
full-data refit raises an exception, the search object remains unfitted and
does not expose results from either the previous fit or the incomplete attempt.
`RandomizedSearchCV` likewise exposes `sampled_params_` only after a successful
fit. A complete successful fit is required before calling `predict()`,
`score()`, or `predict_proba()` again.
