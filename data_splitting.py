"""Deterministic train/test and K-fold index splitting utilities."""

import math
import random
from typing import Sequence, TypeVar

T = TypeVar("T")


def _test_count(sample_count: int, test_size: int | float) -> int:
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
        raise ValueError("test_size must be an integer or a fraction")
    if isinstance(test_size, int):
        test_count = test_size
    else:
        if not math.isfinite(test_size) or not 0 < test_size < 1:
            raise ValueError("fractional test_size must be between 0 and 1")
        test_count = math.ceil(sample_count * test_size)
    if not 1 <= test_count < sample_count:
        raise ValueError("test_size must leave at least one sample in each split")
    return test_count


def train_test_split(
    *arrays: Sequence[T],
    test_size: int | float = 0.25,
    random_state: int | None = None,
    shuffle: bool = True,
) -> tuple[list[T], ...]:
    """Return alternating train/test lists for equally sized input sequences."""
    if not arrays:
        raise ValueError("at least one array is required")
    materialized = [list(array) for array in arrays]
    sample_count = len(materialized[0])
    if sample_count < 2:
        raise ValueError("at least two samples are required")
    if any(len(array) != sample_count for array in materialized):
        raise ValueError("all arrays must have the same number of samples")
    if type(shuffle) is not bool:
        raise ValueError("shuffle must be a boolean")
    if random_state is not None and type(random_state) is not int:
        raise ValueError("random_state must be an integer or None")
    test_count = _test_count(sample_count, test_size)

    indices = list(range(sample_count))
    if shuffle:
        random.Random(random_state).shuffle(indices)
    split_at = sample_count - test_count
    train_indices, test_indices = indices[:split_at], indices[split_at:]
    result: list[list[T]] = []
    for array in materialized:
        result.extend(([array[i] for i in train_indices], [array[i] for i in test_indices]))
    return tuple(result)


def stratified_train_test_split(
    *arrays: Sequence[T],
    labels: Sequence[object],
    test_size: int | float = 0.25,
    random_state: int | None = None,
    shuffle: bool = True,
) -> tuple[list[T], ...]:
    """Split aligned arrays while retaining each class in both partitions.

    Per-class test counts are allocated proportionally using largest
    remainders, subject to at least one item from every class in each split.
    Labels must be hashable and aligned with the input arrays.
    """
    if not arrays:
        raise ValueError("at least one array is required")
    materialized = [list(array) for array in arrays]
    target = list(labels)
    sample_count = len(materialized[0])
    if sample_count < 2:
        raise ValueError("at least two samples are required")
    if any(len(array) != sample_count for array in materialized):
        raise ValueError("all arrays must have the same number of samples")
    if len(target) != sample_count:
        raise ValueError("labels must have the same number of samples as arrays")
    if type(shuffle) is not bool:
        raise ValueError("shuffle must be a boolean")
    if random_state is not None and type(random_state) is not int:
        raise ValueError("random_state must be an integer or None")
    test_count = _test_count(sample_count, test_size)

    groups: dict[object, list[int]] = {}
    try:
        for index, label in enumerate(target):
            groups.setdefault(label, []).append(index)
    except TypeError:
        raise ValueError("labels must be hashable") from None
    if len(groups) > min(test_count, sample_count - test_count):
        raise ValueError("both splits must be large enough to contain every class")
    if any(len(indices) < 2 for indices in groups.values()):
        raise ValueError("each class must contain at least two samples")

    group_indices = list(groups.values())
    desired = [len(indices) * test_count / sample_count for indices in group_indices]
    allocations = [
        min(len(indices) - 1, max(1, math.floor(count)))
        for indices, count in zip(group_indices, desired)
    ]
    while sum(allocations) < test_count:
        candidates = [
            index for index, group in enumerate(group_indices)
            if allocations[index] < len(group) - 1
        ]
        chosen = max(candidates, key=lambda index: (desired[index] - allocations[index], -index))
        allocations[chosen] += 1
    while sum(allocations) > test_count:
        candidates = [index for index, count in enumerate(allocations) if count > 1]
        chosen = min(candidates, key=lambda index: (desired[index] - allocations[index], index))
        allocations[chosen] -= 1

    generator = random.Random(random_state)
    train_indices = []
    test_indices = []
    for indices, count in zip(group_indices, allocations):
        current = indices.copy()
        if shuffle:
            generator.shuffle(current)
        test_indices.extend(current[:count])
        train_indices.extend(current[count:])
    if shuffle:
        generator.shuffle(train_indices)
        generator.shuffle(test_indices)
    else:
        train_indices.sort()
        test_indices.sort()

    result: list[list[T]] = []
    for array in materialized:
        result.extend(([array[index] for index in train_indices], [array[index] for index in test_indices]))
    return tuple(result)


def k_fold_indices(
    n_samples: int,
    n_splits: int = 5,
    *,
    shuffle: bool = False,
    random_state: int | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Return train/validation index pairs for each balanced K-fold split."""
    if type(n_samples) is not int or n_samples < 2:
        raise ValueError("n_samples must be an integer of at least two")
    if type(n_splits) is not int or not 2 <= n_splits <= n_samples:
        raise ValueError("n_splits must be between 2 and n_samples")
    if type(shuffle) is not bool:
        raise ValueError("shuffle must be a boolean")
    if random_state is not None and type(random_state) is not int:
        raise ValueError("random_state must be an integer or None")
    indices = list(range(n_samples))
    if shuffle:
        random.Random(random_state).shuffle(indices)
    base_size, remainder = divmod(n_samples, n_splits)
    folds: list[tuple[list[int], list[int]]] = []
    offset = 0
    for fold_index in range(n_splits):
        fold_size = base_size + (1 if fold_index < remainder else 0)
        validation = indices[offset:offset + fold_size]
        validation_set = set(validation)
        training = [index for index in indices if index not in validation_set]
        folds.append((training, validation))
        offset += fold_size
    return folds


def stratified_k_fold_indices(
    labels: Sequence[T],
    n_splits: int = 5,
    *,
    shuffle: bool = False,
    random_state: int | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Return K-fold splits that preserve each class's proportions."""
    materialized = list(labels)
    sample_count = len(materialized)
    if sample_count < 2:
        raise ValueError("at least two labels are required")
    if type(n_splits) is not int or not 2 <= n_splits <= sample_count:
        raise ValueError("n_splits must be between 2 and the number of labels")
    if type(shuffle) is not bool:
        raise ValueError("shuffle must be a boolean")
    if random_state is not None and type(random_state) is not int:
        raise ValueError("random_state must be an integer or None")
    groups: dict[T, list[int]] = {}
    try:
        for index, label in enumerate(materialized):
            groups.setdefault(label, []).append(index)
    except TypeError:
        raise ValueError("labels must be hashable") from None
    if any(len(indices) < n_splits for indices in groups.values()):
        raise ValueError("each class must contain at least n_splits samples")

    generator = random.Random(random_state)
    validation_folds: list[list[int]] = [[] for _ in range(n_splits)]
    fold_cursor = 0
    for class_indices in groups.values():
        current = class_indices.copy()
        if shuffle:
            generator.shuffle(current)
        for index in current:
            validation_folds[fold_cursor].append(index)
            fold_cursor = (fold_cursor + 1) % n_splits
    if shuffle:
        for fold in validation_folds:
            generator.shuffle(fold)

    all_indices = list(range(sample_count))
    result = []
    for validation in validation_folds:
        validation_set = set(validation)
        training = [index for index in all_indices if index not in validation_set]
        result.append((training, validation))
    return result
