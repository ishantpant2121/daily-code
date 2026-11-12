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
