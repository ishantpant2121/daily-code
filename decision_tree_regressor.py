"""A dependency-free CART regression tree using squared-error splits."""

import math
import random
from dataclasses import dataclass
from numbers import Real
from typing import Iterable

from decision_tree import _feature_matrix
from regression_metrics import r2_score


def _target_values(values: Iterable[Real]) -> list[float]:
    try:
        raw = list(values)
    except TypeError:
        raise ValueError("targets must be an iterable of real numbers") from None
    if not raw:
        raise ValueError("targets must contain at least one value")
    result = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("targets must contain only finite real numbers")
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("targets must contain only finite real numbers") from None
        if not math.isfinite(number):
            raise ValueError("targets must contain only finite real numbers")
        result.append(number)
    return result


def _stable_mean(values: Iterable[float]) -> float:
    values = list(values)
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    mean = scale * (math.fsum(value / scale for value in values) / len(values))
    if not math.isfinite(mean):
        raise ValueError("regression prediction exceeds finite floating-point range")
    return mean


@dataclass
class _RegressionNode:
    value: float
    feature_index: int | None = None
    threshold: float | None = None
    left: int | None = None
    right: int | None = None


class DecisionTreeRegressor:
    """Fit a binary regression tree by minimizing squared-error impurity.

    Candidate thresholds lie between adjacent distinct feature values.
    Optional per-node feature sampling follows the classifier's conventions.
    Tree construction is iterative, so deep trees do not consume Python's
    call stack. Predictions in a leaf are the arithmetic mean of its targets.
    """

    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: Real = 0.0,
        max_features: int | float | str | None = None,
        random_state: int | None = None,
    ) -> None:
        if max_depth is not None and (type(max_depth) is not int or max_depth < 1):
            raise ValueError("max_depth must be a positive integer or None")
        if type(min_samples_split) is not int or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer of at least two")
        if type(min_samples_leaf) is not int or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer")
        if isinstance(min_impurity_decrease, bool) or not isinstance(min_impurity_decrease, Real):
            raise ValueError("min_impurity_decrease must be a finite non-negative number")
        try:
            decrease = float(min_impurity_decrease)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("min_impurity_decrease must be a finite non-negative number") from None
        if not math.isfinite(decrease) or decrease < 0:
            raise ValueError("min_impurity_decrease must be a finite non-negative number")
        if isinstance(max_features, bool):
            raise ValueError("max_features must be None, a positive integer, a fraction in (0, 1], sqrt, or log2")
        if type(max_features) is int:
            if max_features < 1:
                raise ValueError("max_features must be positive")
        elif isinstance(max_features, Real):
            try:
                max_features = float(max_features)
            except (OverflowError, TypeError, ValueError):
                raise ValueError("fractional max_features must be finite and in (0, 1]") from None
            if not math.isfinite(max_features) or not 0 < max_features <= 1:
                raise ValueError("fractional max_features must be finite and in (0, 1]")
        elif max_features not in (None, "sqrt", "log2"):
            raise ValueError("unsupported max_features value")
        if random_state is not None and type(random_state) is not int:
            raise ValueError("random_state must be an integer or None")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = decrease
        self.max_features = max_features
        self.random_state = random_state
        self._fitted = False

    def _max_feature_count(self) -> int:
        if self.max_features is None:
            return self.n_features_in_
        if type(self.max_features) is int:
            return min(self.max_features, self.n_features_in_)
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * self.n_features_in_))
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(self.n_features_in_)))
        return max(1, int(math.log2(self.n_features_in_)))

    @staticmethod
    def _m2(values: list[float]) -> float:
        count = 0
        mean = 0.0
        total = 0.0
        for value in values:
            count += 1
            delta = value - mean
            mean += delta / count
            total += delta * (value - mean)
        return max(0.0, total)

    def _best_split(
        self,
        matrix: list[list[float]],
        scaled_targets: list[float],
        indices: list[int],
        parent_m2: float,
        required_reduction: float,
    ):
        sample_count = len(indices)
        best_gain = required_reduction
        best_split = None
        if self.max_features_ == self.n_features_in_:
            selected = list(range(self.n_features_in_))
            remaining = []
        else:
            selected = sorted(self._feature_rng.sample(range(self.n_features_in_), self.max_features_))
            selected_set = set(selected)
            remaining = [feature for feature in range(self.n_features_in_) if feature not in selected_set]

        def evaluate(feature: int) -> None:
            nonlocal best_gain, best_split
            ordered = sorted(indices, key=lambda index: (matrix[index][feature], index))
            suffix_m2 = [0.0] * (sample_count + 1)
            count = 0
            mean = 0.0
            m2 = 0.0
            for position in range(sample_count - 1, -1, -1):
                value = scaled_targets[ordered[position]]
                count += 1
                delta = value - mean
                mean += delta / count
                m2 += delta * (value - mean)
                suffix_m2[position] = max(0.0, m2)

            left_count = 0
            left_mean = 0.0
            left_m2 = 0.0
            for position in range(sample_count - 1):
                sample_index = ordered[position]
                value = scaled_targets[sample_index]
                left_count += 1
                delta = value - left_mean
                left_mean += delta / left_count
                left_m2 += delta * (value - left_mean)

                right_count = sample_count - left_count
                current = matrix[sample_index][feature]
                following = matrix[ordered[position + 1]][feature]
                if current == following:
                    continue
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                gain = parent_m2 - max(0.0, left_m2) - suffix_m2[position + 1]
                if gain > best_gain:
                    midpoint = current / 2.0 + following / 2.0
                    if midpoint <= current or midpoint >= following:
                        midpoint = current
                    best_gain = gain
                    best_split = (
                        feature,
                        midpoint,
                        gain,
                        ordered[:left_count],
                        ordered[left_count:],
                    )

        for feature in selected:
            evaluate(feature)
        while best_split is None and remaining:
            feature = remaining.pop(self._feature_rng.randrange(len(remaining)))
            evaluate(feature)
        return best_split

    def fit(
        self,
        features: Iterable[Iterable[Real]],
        targets: Iterable[Real],
    ) -> "DecisionTreeRegressor":
        matrix = _feature_matrix(features)
        target = _target_values(targets)
        if len(matrix) != len(target):
            raise ValueError("features and targets must have equal sample counts")

        self.n_features_in_ = len(matrix[0])
        self.n_samples_fit_ = len(matrix)
        self._target_scale = max(abs(value) for value in target)
        scaled_targets = (
            [value / self._target_scale for value in target]
            if self._target_scale else [0.0] * len(target)
        )
        self.max_features_ = self._max_feature_count()
        self._feature_rng = random.Random(self.random_state)
        if self._target_scale == 0.0:
            required_reduction = 0.0 if self.min_impurity_decrease == 0 else math.inf
        else:
            normalized_decrease = (self.min_impurity_decrease / self._target_scale) / self._target_scale
            required_reduction = normalized_decrease * len(target)
            if not math.isfinite(required_reduction):
                required_reduction = math.inf

        root_indices = list(range(len(matrix)))
        root_mean = _stable_mean(target)
        self._nodes = [_RegressionNode(root_mean)]
        stack = [(0, root_indices, 0)]
        gains = [0.0] * self.n_features_in_
        processed_depths = []

        while stack:
            node_index, indices, depth = stack.pop()
            node = self._nodes[node_index]
            node.value = _stable_mean(target[index] for index in indices)
            processed_depths.append(depth)
            if self.max_depth is not None and depth >= self.max_depth:
                continue
            if len(indices) < self.min_samples_split:
                continue
            parent_m2 = self._m2([scaled_targets[index] for index in indices])
            if parent_m2 == 0.0:
                continue
            split = self._best_split(
                matrix, scaled_targets, indices, parent_m2, required_reduction
            )
            if split is None:
                continue
            feature, threshold, gain, left_indices, right_indices = split
            gains[feature] += gain
            left_node = len(self._nodes)
            self._nodes.append(_RegressionNode(0.0))
            right_node = len(self._nodes)
            self._nodes.append(_RegressionNode(0.0))
            node.feature_index = feature
            node.threshold = threshold
            node.left = left_node
            node.right = right_node
            stack.append((right_node, right_indices, depth + 1))
            stack.append((left_node, left_indices, depth + 1))

        total_gain = math.fsum(gains)
        self.feature_importances_ = (
            [value / total_gain for value in gains]
            if total_gain > 0.0 else [0.0] * self.n_features_in_
        )
        self.node_count_ = len(self._nodes)
        self.n_leaves_ = sum(node.feature_index is None for node in self._nodes)
        self.max_depth_ = max(processed_depths, default=0)
        self._fitted = True
        return self

    def _leaf(self, row: list[float]) -> int:
        index = 0
        while self._nodes[index].feature_index is not None:
            node = self._nodes[index]
            if row[node.feature_index] <= node.threshold:
                index = node.left
            else:
                index = node.right
        return index

    def apply(self, features: Iterable[Iterable[Real]]) -> list[int]:
        """Return the terminal node index reached by each input sample."""
        self._check_fitted()
        matrix = _feature_matrix(features)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        return [self._leaf(row) for row in matrix]

    def predict(self, features: Iterable[Iterable[Real]]) -> list[float]:
        self._check_fitted()
        matrix = _feature_matrix(features)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        return [self._nodes[self._leaf(row)].value for row in matrix]

    def score(self, features: Iterable[Iterable[Real]], targets: Iterable[Real]) -> float:
        return r2_score(_target_values(targets), self.predict(features))

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before prediction")
