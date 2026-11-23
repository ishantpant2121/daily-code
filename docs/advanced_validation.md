# Advanced validation splitters

The `advanced_splitting` module complements the basic train/test and K-fold
helpers in `data_splitting.py`. It returns integer indices, so the same split
can be applied to features, labels, timestamps, and other aligned arrays.

## Time-ordered observations

`TimeSeriesSplit` uses earlier observations for training and later observations
for testing. Its training window expands by default; `max_train_size` bounds it
for a rolling-window evaluation. `gap` excludes observations immediately
before each test window, which can reduce leakage when feature or target
windows overlap.

```python
from advanced_splitting import TimeSeriesSplit

splitter = TimeSeriesSplit(n_splits=3, test_size=4, gap=1)
for train_indices, test_indices in splitter.split(40):
    assert max(train_indices) < min(test_indices)
```

Samples should be in chronological order before splitting. For comparable
fold durations, observations should be sampled at regular intervals. The
splitter cannot infer timestamps or verify temporal spacing.

## Groups and repeated folds

- `GroupKFold` assigns each group to exactly one validation fold and greedily
  balances the number of samples in each fold.
- `GroupShuffleSplit` creates randomized holdouts at the group level. Its
  `test_size` counts distinct groups when an integer is supplied and represents
  a fraction of distinct groups when a float is supplied.
- `StratifiedShuffleSplit` creates repeated randomized holdouts while keeping
  every class in both partitions. Class counts are allocated proportionally.
- `RepeatedKFold` repeats shuffled K-fold validation.
- `RepeatedStratifiedKFold` repeats shuffled, class-balanced K-fold validation.
- `LeaveOneOut` creates one fold per sample and is mainly useful for very small
  datasets because it requires one model fit per observation.

All splitters are deterministic when given the same integer `random_state`.
Group labels and class labels must be hashable. The splitters return indices;
they do not fit estimators.

## Choosing a splitter

Use a time-series splitter when future observations must never enter a training
fold. Use a group splitter when observations from the same subject, customer,
device, or experiment must stay together. Use stratification for classification
when each validation partition needs representation from every class. Repeated
splitters help assess sensitivity to a particular random partition, at the
cost of additional model fits.


## Reusing explicit folds in evaluation and model search

Every cross-validation helper accepts either an integer number of folds or a
reusable iterable of `(train_indices, validation_indices)` pairs. Explicit
folds are validated for non-empty partitions, integer indices, bounds,
duplicates, and train/validation overlap. This lets the splitters in this module
be used without changing the model-selection APIs.

`PredefinedSplit` stores fixed assignments: each non-negative value marks a
validation fold, while `-1` keeps that row in training for every fold.

```python
from advanced_splitting import PredefinedSplit
from decision_tree import DecisionTreeClassifier
from model_selection import cross_val_score

features = [[0], [1], [2], [3], [10], [11]]
labels = [0, 0, 0, 1, 1, 1]
folds = PredefinedSplit([0, 0, 1, 1, 2, 2])

scores = cross_val_score(
    DecisionTreeClassifier(max_depth=2),
    features,
    labels,
    cv=folds,
    scoring="accuracy",
)
print(scores)
```

A splitter that returns index pairs can also be passed directly by supplying its
generated folds. This is especially useful for temporal and group-aware splits:

```python
from advanced_splitting import TimeSeriesSplit, GroupKFold

features = [[float(index)] for index in range(12)]
groups = [index // 2 for index in range(12)]
time_folds = TimeSeriesSplit(n_splits=3, test_size=2, gap=1).split(len(features))
group_folds = GroupKFold(n_splits=3).split(len(features), groups)
```

Explicit folds are reused unchanged by `cross_val_score`, `cross_validate`,
`learning_curve`, `validation_curve`, `GridSearchCV`, and
`RandomizedSearchCV`. For search, each candidate is scored on the same folds.
When passing explicit folds, leave `stratified`, `shuffle`, and
`random_state` at their defaults because the fold assignments already encode
the splitting strategy.

`cross_val_predict` is stricter: each sample must appear in exactly one
validation fold. It rejects repeated validation assignments, missing samples,
and predefined assignments of `-1` that are never validated.
