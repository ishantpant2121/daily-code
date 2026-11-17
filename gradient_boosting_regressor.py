"""A small, dependency-free gradient-boosting regressor for squared error."""

import math
import random
from collections.abc import Iterable, Iterator
from numbers import Real

from decision_tree import _feature_matrix
from decision_tree_regressor import DecisionTreeRegressor, _stable_mean, _target_values
from regression_metrics import r2_score


class GradientBoostingRegressor:
    """Build an additive regression model from shallow trees.

    Each stage fits a regression tree to the current residuals and adds its
    predictions multiplied by learning_rate. Targets are scaled during fitting
    so large but finite values do not overflow while residuals are calculated.
    subsample below one enables stochastic boosting by selecting rows without
    replacement at each stage.

    This implementation supports squared-error loss only. It is intended for
    education and small datasets rather than high-performance workloads.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        learning_rate: Real = 0.1,
        subsample: Real = 1.0,
        max_depth: int | None = 2,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = None,
        random_state: int | None = None,
    ) -> None:
        if type(n_estimators) is not int or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, Real):
            raise ValueError("learning_rate must be a finite positive number")
        try:
            normalized_rate = float(learning_rate)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("learning_rate must be a finite positive number") from None
        if not math.isfinite(normalized_rate) or normalized_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number")
        if isinstance(subsample, bool) or not isinstance(subsample, Real):
            raise ValueError("subsample must be a finite number in (0, 1]")
        try:
            normalized_subsample = float(subsample)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("subsample must be a finite number in (0, 1]") from None
        if not math.isfinite(normalized_subsample) or not 0 < normalized_subsample <= 1:
            raise ValueError("subsample must be a finite number in (0, 1]")
        if random_state is not None and type(random_state) is not int:
            raise ValueError("random_state must be an integer or None")

        # Delegate tree-specific parameter validation to the base estimator.
        DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=0,
        )

        self.n_estimators = n_estimators
        self.learning_rate = normalized_rate
        self.subsample = normalized_subsample
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self._fitted = False

    def fit(
        self,
        features: Iterable[Iterable[Real]],
        targets: Iterable[Real],
    ) -> "GradientBoostingRegressor":
        """Fit the stage-wise additive model and return self."""
        matrix = _feature_matrix(features)
        target = _target_values(targets)
        if len(matrix) != len(target):
            raise ValueError("features and targets must have equal sample counts")

        sample_count = len(matrix)
        target_scale = max(abs(value) for value in target)
        scaled_target = (
            [value / target_scale for value in target]
            if target_scale else [0.0] * sample_count
        )
        initial = _stable_mean(scaled_target)
        current = [initial] * sample_count
        generator = random.Random(self.random_state)
        estimators = []
        estimator_samples = []

        draw_count = max(1, int(self.subsample * sample_count))
        for _ in range(self.n_estimators):
            indices = (
                generator.sample(range(sample_count), draw_count)
                if draw_count < sample_count
                else list(range(sample_count))
            )
            residuals = [scaled_target[index] - current[index] for index in indices]
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=generator.randrange(0, 2**32),
            )
            tree.fit(
                [matrix[index] for index in indices],
                residuals,
            )
            updates = tree.predict(matrix)
            next_values = [
                value + self.learning_rate * update
                for value, update in zip(current, updates)
            ]
            if any(not math.isfinite(value) for value in next_values):
                raise ValueError("boosted prediction exceeds finite floating-point range")
            current = next_values
            estimators.append(tree)
            estimator_samples.append(tuple(indices))

        feature_count = len(matrix[0])
        importances = [
            math.fsum(tree.feature_importances_[feature] for tree in estimators)
            / self.n_estimators
            for feature in range(feature_count)
        ]
        total_importance = math.fsum(importances)
        if total_importance > 0:
            importances = [value / total_importance for value in importances]

        initial_value = target_scale * initial
        if not math.isfinite(initial_value):
            raise ValueError("initial prediction exceeds finite floating-point range")

        self.n_features_in_ = feature_count
        self.n_samples_fit_ = sample_count
        self.target_scale_ = target_scale
        self.init_prediction_ = initial_value
        self._init_prediction_scaled = initial
        self.estimators_ = estimators
        self.estimators_samples_ = estimator_samples
        self.feature_importances_ = importances
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before prediction")

    def _validated_matrix(self, features: Iterable[Iterable[Real]]) -> list[list[float]]:
        matrix = _feature_matrix(features)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        return matrix

    def _restore_scale(self, values: list[float]) -> list[float]:
        restored = [self.target_scale_ * value for value in values]
        if any(not math.isfinite(value) for value in restored):
            raise ValueError("prediction exceeds finite floating-point range")
        return restored

    def staged_predict(
        self, features: Iterable[Iterable[Real]]
    ) -> Iterator[list[float]]:
        """Yield predictions after each boosting stage, in fitting order."""
        self._check_fitted()
        matrix = self._validated_matrix(features)
        current = [self._init_prediction_scaled] * len(matrix)
        for tree in self.estimators_:
            updates = tree.predict(matrix)
            current = [
                value + self.learning_rate * update
                for value, update in zip(current, updates)
            ]
            if any(not math.isfinite(value) for value in current):
                raise ValueError("prediction exceeds finite floating-point range")
            yield self._restore_scale(current)

    def predict(self, features: Iterable[Iterable[Real]]) -> list[float]:
        """Predict targets using all fitted boosting stages."""
        final_predictions = None
        for final_predictions in self.staged_predict(features):
            pass
        return final_predictions

    def apply(self, features: Iterable[Iterable[Real]]) -> list[list[int]]:
        """Return sample-major leaf indices, one column per boosting stage."""
        self._check_fitted()
        matrix = self._validated_matrix(features)
        per_stage = [tree.apply(matrix) for tree in self.estimators_]
        return [
            [stage[row_index] for stage in per_stage]
            for row_index in range(len(matrix))
        ]

    def score(
        self,
        features: Iterable[Iterable[Real]],
        targets: Iterable[Real],
    ) -> float:
        """Return the coefficient of determination on the supplied samples."""
        return r2_score(_target_values(targets), self.predict(features))
