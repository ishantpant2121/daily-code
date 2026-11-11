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
