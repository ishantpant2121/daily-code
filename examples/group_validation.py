"""Demonstrate group-aware and stratified validation strategies."""

from advanced_splitting import (
    GroupKFold,
    GroupShuffleSplit,
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
)

groups = ["customer-a"] * 3 + ["customer-b"] * 2 + ["customer-c"] * 3 + ["customer-d"] * 2
labels = [0, 0, 1, 0, 1, 1, 1, 0, 0, 1]

for train, test in GroupKFold(n_splits=3).split(len(groups), groups):
    assert not ({groups[index] for index in train} & {groups[index] for index in test})

holdout = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
train_indices, test_indices = holdout.split(len(groups), groups)[0]
print("Group-disjoint holdout:", train_indices, test_indices)

stratified = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=7)
print("Stratified holdouts:", stratified.split(labels))

repeated = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=7)
print("Repeated stratified folds:", len(repeated.split(labels)))
