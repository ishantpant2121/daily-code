"""A small, deterministic CART-style decision-tree classifier."""

import math
import random
from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable


def _feature_matrix(values: Iterable[Iterable[Real]]) -> list[list[float]]:
    try:
        rows = [list(row) for row in values]
    except TypeError:
        raise ValueError("features must be an iterable of rows") from None
    if not rows or not rows[0]:
        raise ValueError("features must contain at least one row and one column")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all feature rows must have equal lengths")
    matrix = []
    for row in rows:
        converted = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("features must be finite real numbers")
            try:
                number = float(value)
            except (OverflowError, TypeError, ValueError):
                raise ValueError("features must be finite real numbers") from None
            if not math.isfinite(number):
                raise ValueError("features must be finite real numbers")
            converted.append(number)
        matrix.append(converted)
    return matrix


@dataclass
class _Node:
    class_counts: tuple[int, ...]
    feature_index: int | None = None
    threshold: float | None = None
    left: int | None = None
    right: int | None = None


class DecisionTreeClassifier:
    """Fit a binary-split classification tree using Gini or entropy impurity.

    Candidate thresholds are evaluated between adjacent distinct feature
    values. Optional max_features sampling selects a random feature subset at
    each node; additional features are checked only when the sampled subset
    cannot form a valid split. A fixed random_state makes sampling reproducible.
    Tree construction is iterative, avoiding Python recursion limits on deep
    trees. Class probabilities are empirical leaf frequencies.
    """

    def __init__(
        self,
        *,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: Real = 0.0,
        max_features: int | float | str | None = None,
        random_state: int | None = None,
    ) -> None:
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if max_depth is not None and (type(max_depth) is not int or max_depth < 1):
            raise ValueError("max_depth must be a positive integer or None")
        if type(min_samples_split) is not int or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer of at least two")
        if type(min_samples_leaf) is not int or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer")
        if isinstance(min_impurity_decrease, bool) or not isinstance(
            min_impurity_decrease, Real
        ):
            raise ValueError("min_impurity_decrease must be a finite non-negative number")
        try:
            decrease = float(min_impurity_decrease)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("min_impurity_decrease must be a finite non-negative number") from None
        if not math.isfinite(decrease) or decrease < 0:
            raise ValueError("min_impurity_decrease must be a finite non-negative number")
        if isinstance(max_features, bool):
            raise ValueError("max_features must be None, a positive integer, a fraction in (0, 1], 'sqrt', or 'log2'")
        if type(max_features) is int:
            if max_features < 1:
                raise ValueError("integer max_features must be positive")
        elif isinstance(max_features, Real):
            try:
                max_features = float(max_features)
            except (OverflowError, TypeError, ValueError):
                raise ValueError("fractional max_features must be finite and in (0, 1]") from None
            if not math.isfinite(max_features) or not 0 < max_features <= 1:
                raise ValueError("fractional max_features must be finite and in (0, 1]")
        elif max_features not in (None, "sqrt", "log2"):
            raise ValueError("max_features must be None, a positive integer, a fraction in (0, 1], 'sqrt', or 'log2'")
        if random_state is not None and type(random_state) is not int:
            raise ValueError("random_state must be an integer or None")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = decrease
        self.max_features = max_features
        self.random_state = random_state
        self._fitted = False

    def _impurity(self, counts: list[int]) -> float:
        total = sum(counts)
        if total == 0:
            return 0.0
        probabilities = [count / total for count in counts if count]
        if self.criterion == "gini":
            return max(0.0, 1.0 - math.fsum(value * value for value in probabilities))
        return -math.fsum(value * math.log2(value) for value in probabilities)

    @staticmethod
    def _counts(indices: list[int], encoded_labels: list[int], class_count: int) -> tuple[int, ...]:
        counts = [0] * class_count
        for index in indices:
            counts[encoded_labels[index]] += 1
        return tuple(counts)

    def _max_feature_count(self) -> int:
        """Resolve max_features against the fitted input width."""
        if self.max_features is None:
            return self.n_features_in_
        if type(self.max_features) is int:
            return min(self.max_features, self.n_features_in_)
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * self.n_features_in_))
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(self.n_features_in_)))
        return max(1, int(math.log2(self.n_features_in_)))

    def _best_split(
        self,
        matrix: list[list[float]],
        encoded_labels: list[int],
        indices: list[int],
        parent_counts: tuple[int, ...],
    ) -> tuple[int, float, float, list[int], list[int]] | None:
        parent_impurity = self._impurity(list(parent_counts))
        sample_count = len(indices)
        best_gain = 0.0
        best_split = None
        class_count = len(parent_counts)

        feature_count = self.max_features_
        if feature_count == self.n_features_in_:
            selected_features = list(range(self.n_features_in_))
            remaining_features = []
        else:
            selected_features = sorted(
                self._feature_rng.sample(range(self.n_features_in_), feature_count)
            )
            selected_set = set(selected_features)
            remaining_features = [
                feature for feature in range(self.n_features_in_)
                if feature not in selected_set
            ]

        def evaluate_feature(feature: int) -> None:
            nonlocal best_gain, best_split
            ordered = sorted(indices, key=lambda index: (matrix[index][feature], index))
            left_counts = [0] * class_count
            right_counts = list(parent_counts)
            for position in range(sample_count - 1):
                sample_index = ordered[position]
                class_index = encoded_labels[sample_index]
                left_counts[class_index] += 1
                right_counts[class_index] -= 1
                left_size = position + 1
                right_size = sample_count - left_size
                current_value = matrix[sample_index][feature]
                next_value = matrix[ordered[position + 1]][feature]

                if current_value == next_value:
                    continue
                if left_size < self.min_samples_leaf or right_size < self.min_samples_leaf:
                    continue

                weighted_impurity = (
                    left_size * self._impurity(left_counts)
                    + right_size * self._impurity(right_counts)
                ) / sample_count
                gain = parent_impurity - weighted_impurity
                if gain > best_gain + 1e-15:
                    best_gain = gain
                    midpoint = current_value / 2.0 + next_value / 2.0
                    if midpoint <= current_value or midpoint >= next_value:
                        midpoint = current_value
                    best_split = (
                        feature,
                        midpoint,
                        gain,
                        ordered[:left_size],
                        ordered[left_size:],
                    )

        for feature in selected_features:
            evaluate_feature(feature)

        # If the sampled features cannot produce a valid split, inspect
        # additional features until one can, matching the usual tree-search
        # safeguard for nodes where sampled columns are constant.
        while best_split is None and remaining_features:
            offset = self._feature_rng.randrange(len(remaining_features))
            feature = remaining_features.pop(offset)
            evaluate_feature(feature)
        return best_split

    def fit(
        self,
        features: Iterable[Iterable[Real]],
        labels: Iterable[Any],
    ) -> "DecisionTreeClassifier":
        matrix = _feature_matrix(features)
        target = list(labels)
        if len(target) != len(matrix):
            raise ValueError("features and labels must have equal sample counts")

        class_indices: dict[Any, int] = {}
        try:
            for label in target:
                if label not in class_indices:
                    class_indices[label] = len(class_indices)
        except TypeError:
            raise ValueError("class labels must be hashable") from None

        encoded = [class_indices[label] for label in target]
        self.n_features_in_ = len(matrix[0])
        self.max_features_ = self._max_feature_count()
        self._feature_rng = random.Random(self.random_state)
        self.classes_ = tuple(class_indices)
        self._nodes = [
            _Node(self._counts(list(range(len(matrix))), encoded, len(self.classes_)))
        ]
        importances = [0.0] * self.n_features_in_
        max_depth_seen = 0
        stack = [(0, list(range(len(matrix))), 0)]

        while stack:
            node_index, sample_indices, depth = stack.pop()
            max_depth_seen = max(max_depth_seen, depth)
            node = self._nodes[node_index]
            counts = list(node.class_counts)
            if (
                len(sample_indices) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)
            ):
                continue
            if sum(count > 0 for count in counts) <= 1:
                continue

            split = self._best_split(matrix, encoded, sample_indices, node.class_counts)
            if split is None:
                continue
            feature, threshold, gain, left_indices, right_indices = split
            if gain <= self.min_impurity_decrease:
                continue

            left_index = len(self._nodes)
            self._nodes.append(
                _Node(self._counts(left_indices, encoded, len(self.classes_)))
            )
            right_index = len(self._nodes)
            self._nodes.append(
                _Node(self._counts(right_indices, encoded, len(self.classes_)))
            )
            node.feature_index = feature
            node.threshold = threshold
            node.left = left_index
            node.right = right_index
            importances[feature] += len(sample_indices) / len(matrix) * gain
            stack.append((right_index, right_indices, depth + 1))
            stack.append((left_index, left_indices, depth + 1))

        total_importance = math.fsum(importances)
        self.feature_importances_ = (
            [value / total_importance for value in importances]
            if total_importance > 0
            else importances
        )
        self.node_count_ = len(self._nodes)
        self.n_leaves_ = sum(node.feature_index is None for node in self._nodes)
        self.max_depth_ = max_depth_seen
        self._fitted = True
        return self

    def _leaf_index(self, row: list[float]) -> int:
        node_index = 0
        while self._nodes[node_index].feature_index is not None:
            node = self._nodes[node_index]
            feature = node.feature_index
            if row[feature] <= node.threshold:
                node_index = node.left
            else:
                node_index = node.right
        return node_index

    def apply(self, features: Iterable[Iterable[Real]]) -> list[int]:
        self._check_fitted()
        matrix = _feature_matrix(features)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        return [self._leaf_index(row) for row in matrix]

    def predict_proba(self, features: Iterable[Iterable[Real]]) -> list[list[float]]:
        leaf_indices = self.apply(features)
        probabilities = []
        for index in leaf_indices:
            counts = self._nodes[index].class_counts
            total = sum(counts)
            probabilities.append([count / total for count in counts])
        return probabilities

    def predict(self, features: Iterable[Iterable[Real]]) -> list[Any]:
        return [
            self.classes_[max(range(len(row)), key=row.__getitem__)]
            for row in self.predict_proba(features)
        ]

    def score(
        self,
        features: Iterable[Iterable[Real]],
        labels: Iterable[Any],
    ) -> float:
        target = list(labels)
        predicted = self.predict(features)
        if len(target) != len(predicted):
            raise ValueError("features and labels must have equal sample counts")
        return sum(actual == expected for actual, expected in zip(target, predicted)) / len(target)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before prediction")
