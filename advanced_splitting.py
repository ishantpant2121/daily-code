"""Advanced cross-validation splitters for ordered and grouped samples."""

import math
import random
from collections.abc import Sequence
from typing import Any

from data_splitting import k_fold_indices, stratified_k_fold_indices


class TimeSeriesSplit:
    """Create expanding or bounded rolling splits without training on the future.

    ``split(n_samples)`` returns ``(train_indices, test_indices)`` pairs. A
    configurable ``gap`` excludes observations immediately before each test
    window, which is useful when labels or features overlap in time.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        max_train_size: int | None = None,
        test_size: int | None = None,
        gap: int = 0,
    ) -> None:
        if type(n_splits) is not int or n_splits < 2:
            raise ValueError("n_splits must be an integer of at least two")
        if max_train_size is not None and (
            type(max_train_size) is not int or max_train_size < 1
        ):
            raise ValueError("max_train_size must be a positive integer or None")
        if test_size is not None and (type(test_size) is not int or test_size < 1):
            raise ValueError("test_size must be a positive integer or None")
        if type(gap) is not int or gap < 0:
            raise ValueError("gap must be a non-negative integer")
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, n_samples: int) -> list[tuple[list[int], list[int]]]:
        """Return chronological train/test index pairs for ``n_samples`` rows."""
        if type(n_samples) is not int or n_samples < 2:
            raise ValueError("n_samples must be an integer of at least two")
        test_size = self.test_size or n_samples // (self.n_splits + 1)
        if test_size < 1:
            raise ValueError("not enough samples for the requested number of splits")
        first_test_start = n_samples - self.n_splits * test_size
        if first_test_start - self.gap < 1:
            raise ValueError("each split must leave at least one training sample")
        result = []
        for fold in range(self.n_splits):
            test_start = first_test_start + fold * test_size
            train_end = test_start - self.gap
            train_start = (
                0
                if self.max_train_size is None
                else max(0, train_end - self.max_train_size)
            )
            train = list(range(train_start, train_end))
            test = list(range(test_start, test_start + test_size))
            if not train or not test:
                raise ValueError("each split must contain training and test samples")
            result.append((train, test))
        return result


class GroupKFold:
    """Split samples so every group appears in exactly one validation fold.

    Groups are assigned largest-first to the currently smallest fold. This
    greedy strategy balances sample counts while preserving group integrity.
    Ties are resolved deterministically by original group order and fold index.
    """

    def __init__(self, n_splits: int = 5) -> None:
        if type(n_splits) is not int or n_splits < 2:
            raise ValueError("n_splits must be an integer of at least two")
        self.n_splits = n_splits

    def split(
        self,
        n_samples: int,
        groups: Sequence[Any],
    ) -> list[tuple[list[int], list[int]]]:
        """Return train/test indices for aligned group labels."""
        if type(n_samples) is not int or n_samples < 2:
            raise ValueError("n_samples must be an integer of at least two")
        try:
            group_values = list(groups)
        except TypeError:
            raise ValueError("groups must be an iterable of hashable labels") from None
        if len(group_values) != n_samples:
            raise ValueError("groups must have the same number of samples")
        group_indices: dict[Any, list[int]] = {}
        try:
            for index, group in enumerate(group_values):
                group_indices.setdefault(group, []).append(index)
        except TypeError:
            raise ValueError("group labels must be hashable") from None
        if len(group_indices) < self.n_splits:
            raise ValueError("the number of distinct groups must be at least n_splits")

        ordered_groups = sorted(
            enumerate(group_indices.items()),
            key=lambda item: (-len(item[1][1]), item[0]),
        )
        fold_indices: list[list[int]] = [[] for _ in range(self.n_splits)]
        fold_sizes = [0] * self.n_splits
        for _, (_, indices) in ordered_groups:
            fold = min(range(self.n_splits), key=lambda index: (fold_sizes[index], index))
            fold_indices[fold].extend(indices)
            fold_sizes[fold] += len(indices)

        all_indices = list(range(n_samples))
        result = []
        for validation in fold_indices:
            validation.sort()
            validation_set = set(validation)
            training = [index for index in all_indices if index not in validation_set]
            result.append((training, validation))
        return result


class RepeatedKFold:
    """Repeat shuffled K-fold validation with independently derived seeds."""

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 10,
        *,
        random_state: int | None = None,
    ) -> None:
        if type(n_splits) is not int or n_splits < 2:
            raise ValueError("n_splits must be an integer of at least two")
        if type(n_repeats) is not int or n_repeats < 1:
            raise ValueError("n_repeats must be a positive integer")
        if random_state is not None and type(random_state) is not int:
            raise ValueError("random_state must be an integer or None")
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, n_samples: int) -> list[tuple[list[int], list[int]]]:
        """Return all shuffled folds, grouped in repeat order."""
        if type(n_samples) is not int or n_samples < self.n_splits:
            raise ValueError("n_samples must be at least n_splits")
        generator = random.Random(self.random_state)
        result = []
        for _ in range(self.n_repeats):
            seed = generator.randrange(0, 2**32)
            result.extend(
                k_fold_indices(
                    n_samples,
                    self.n_splits,
                    shuffle=True,
                    random_state=seed,
                )
            )
        return result


class RepeatedStratifiedKFold:
    """Repeat shuffled stratified K-fold splits with reproducible seeds."""

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 10,
        *,
        random_state: int | None = None,
    ) -> None:
        if type(n_splits) is not int or n_splits < 2:
            raise ValueError("n_splits must be an integer of at least two")
        if type(n_repeats) is not int or n_repeats < 1:
            raise ValueError("n_repeats must be a positive integer")
        if random_state is not None and type(random_state) is not int:
            raise ValueError("random_state must be an integer or None")
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, labels: Sequence[Any]) -> list[tuple[list[int], list[int]]]:
        """Return all stratified folds in repeat order."""
        values = list(labels)
        if len(values) < self.n_splits:
            raise ValueError("the sample count must be at least n_splits")
        generator = random.Random(self.random_state)
        result = []
        for _ in range(self.n_repeats):
            seed = generator.randrange(0, 2**32)
            result.extend(
                stratified_k_fold_indices(
                    values,
                    self.n_splits,
                    shuffle=True,
                    random_state=seed,
                )
            )
        return result


class GroupShuffleSplit:
    """Create randomized train/test partitions without splitting groups."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        test_size: int | float = 0.2,
        random_state: int | None = None,
    ) -> None:
        if type(n_splits) is not int or n_splits < 1:
            raise ValueError("n_splits must be a positive integer")
        if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
            raise ValueError("test_size must be an integer group count or a fraction")
        if isinstance(test_size, float) and (
            not 0.0 < test_size < 1.0 or not math.isfinite(test_size)
        ):
            raise ValueError("fractional test_size must be finite and between zero and one")
        if isinstance(test_size, int) and test_size < 1:
            raise ValueError("integer test_size must be positive")
        if random_state is not None and type(random_state) is not int:
            raise ValueError("random_state must be an integer or None")
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def split(
        self,
        n_samples: int,
        groups: Sequence[Any],
    ) -> list[tuple[list[int], list[int]]]:
        """Return randomized group-disjoint train/test indices."""
        if type(n_samples) is not int or n_samples < 2:
            raise ValueError("n_samples must be an integer of at least two")
        try:
            group_values = list(groups)
        except TypeError:
            raise ValueError("groups must be an iterable of hashable labels") from None
        if len(group_values) != n_samples:
            raise ValueError("groups must have the same number of samples")
        group_indices: dict[Any, list[int]] = {}
        try:
            for index, group in enumerate(group_values):
                group_indices.setdefault(group, []).append(index)
        except TypeError:
            raise ValueError("group labels must be hashable") from None
        unique_groups = list(group_indices)
        if len(unique_groups) < 2:
            raise ValueError("at least two distinct groups are required")
        if isinstance(self.test_size, float):
            test_group_count = math.ceil(len(unique_groups) * self.test_size)
        else:
            test_group_count = self.test_size
        if not 1 <= test_group_count < len(unique_groups):
            raise ValueError("test_size must leave at least one group in each partition")

        generator = random.Random(self.random_state)
        all_indices = list(range(n_samples))
        result = []
        for _ in range(self.n_splits):
            shuffled = unique_groups[:]
            generator.shuffle(shuffled)
            selected = set(shuffled[:test_group_count])
            test = [
                index for group in selected for index in group_indices[group]
            ]
            test_set = set(test)
            train = [index for index in all_indices if index not in test_set]
            test.sort()
            result.append((train, test))
        return result


class LeaveOneOut:
    """Generate one validation fold per sample."""

    def split(self, n_samples: int) -> list[tuple[list[int], list[int]]]:
        if type(n_samples) is not int or n_samples < 2:
            raise ValueError("n_samples must be an integer of at least two")
        all_indices = list(range(n_samples))
        return [
            ([index for index in all_indices if index != held_out], [held_out])
            for held_out in all_indices
        ]


class StratifiedShuffleSplit:
    """Generate randomized holdout splits with approximately preserved classes."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        test_size: int | float = 0.25,
        random_state: int | None = None,
    ) -> None:
        if type(n_splits) is not int or n_splits < 1:
            raise ValueError("n_splits must be a positive integer")
        if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
            raise ValueError("test_size must be an integer sample count or a fraction")
        if isinstance(test_size, float) and (
            not math.isfinite(test_size) or not 0.0 < test_size < 1.0
        ):
            raise ValueError("fractional test_size must be finite and between zero and one")
        if isinstance(test_size, int) and test_size < 1:
            raise ValueError("integer test_size must be positive")
        if random_state is not None and type(random_state) is not int:
            raise ValueError("random_state must be an integer or None")
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def split(self, labels: Sequence[Any]) -> list[tuple[list[int], list[int]]]:
        """Return stratified train/test indices for every randomized split."""
        values = list(labels)
        sample_count = len(values)
        if sample_count < 2:
            raise ValueError("at least two labels are required")
        groups: dict[Any, list[int]] = {}
        try:
            for index, label in enumerate(values):
                groups.setdefault(label, []).append(index)
        except TypeError:
            raise ValueError("class labels must be hashable") from None
        if len(groups) < 2 or any(len(indices) < 2 for indices in groups.values()):
            raise ValueError("each of at least two classes must contain two samples")
        test_count = (
            math.ceil(sample_count * self.test_size)
            if isinstance(self.test_size, float)
            else self.test_size
        )
        if not 1 <= test_count < sample_count or test_count < len(groups) or (
            sample_count - test_count < len(groups)
        ):
            raise ValueError("both partitions must contain every class")

        desired = {
            label: len(indices) * test_count / sample_count
            for label, indices in groups.items()
        }
        allocations = {
            label: min(len(indices) - 1, max(1, math.floor(desired[label])))
            for label, indices in groups.items()
        }
        while sum(allocations.values()) < test_count:
            candidates = [
                label for label, indices in groups.items()
                if allocations[label] < len(indices) - 1
            ]
            chosen = max(
                candidates,
                key=lambda label: (desired[label] - allocations[label], -list(groups).index(label)),
            )
            allocations[chosen] += 1
        while sum(allocations.values()) > test_count:
            candidates = [label for label, count in allocations.items() if count > 1]
            chosen = min(
                candidates,
                key=lambda label: (desired[label] - allocations[label], list(groups).index(label)),
            )
            allocations[chosen] -= 1

        generator = random.Random(self.random_state)
        result = []
        for _ in range(self.n_splits):
            train = []
            test = []
            for label, indices in groups.items():
                current = indices[:]
                generator.shuffle(current)
                count = allocations[label]
                test.extend(current[:count])
                train.extend(current[count:])
            generator.shuffle(train)
            generator.shuffle(test)
            result.append((train, test))
        return result
