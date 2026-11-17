"""Bootstrap-aggregated regression trees without third-party dependencies."""

import math
import random
from collections.abc import Iterable
from numbers import Real

from decision_tree_regressor import (
    DecisionTreeRegressor,
    _stable_mean,
    _target_values,
)
from decision_tree import _feature_matrix
from regression_metrics import r2_score


class RandomForestRegressor:
    """Predict by averaging bootstrap-fitted squared-error regression trees.

    Each tree is trained on a bootstrap sample by default and considers a
    random subset of features at each split. A fixed random_state makes the
    bootstrap draws and tree feature sampling reproducible.

    When oob_score=True, each training row is predicted only by trees whose
    bootstrap samples omitted that row. Rows without out-of-bag predictions
    contain NaN in oob_prediction_ and are excluded from the OOB R-squared
    calculation. This is an educational, sequential implementation.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: Real = 0.0,
        max_features: int | float | str | None = 1.0,
        bootstrap: bool = True,
        max_samples: int | float | None = None,
        oob_score: bool = False,
        random_state: int | None = None,
    ) -> None:
        if type(n_estimators) is not int or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if type(bootstrap) is not bool:
            raise ValueError("bootstrap must be a boolean")
        if type(oob_score) is not bool:
            raise ValueError("oob_score must be a boolean")
        if oob_score and not bootstrap:
            raise ValueError("oob_score requires bootstrap=True")
        if random_state is not None and type(random_state) is not int:
            raise ValueError("random_state must be an integer or None")

        DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            random_state=0,
        )

        normalized_max_samples = max_samples
        if max_samples is not None:
            if not bootstrap:
                raise ValueError("max_samples can only be set when bootstrap=True")
            if isinstance(max_samples, bool):
                raise ValueError("max_samples must be a positive integer or a fraction in (0, 1]")
            if type(max_samples) is int:
                if max_samples < 1:
                    raise ValueError("integer max_samples must be positive")
            elif isinstance(max_samples, Real):
                try:
                    normalized_max_samples = float(max_samples)
                except (OverflowError, TypeError, ValueError):
                    raise ValueError("fractional max_samples must be finite and in (0, 1]") from None
                if not math.isfinite(normalized_max_samples) or not 0 < normalized_max_samples <= 1:
                    raise ValueError("fractional max_samples must be finite and in (0, 1]")
            else:
                raise ValueError("max_samples must be a positive integer or a fraction in (0, 1]")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = normalized_max_samples
        self.oob_score = oob_score
        self.random_state = random_state
        self._fitted = False

    def fit(
        self,
        features: Iterable[Iterable[Real]],
        targets: Iterable[Real],
    ) -> "RandomForestRegressor":
        matrix = _feature_matrix(features)
        target = _target_values(targets)
        if len(matrix) != len(target):
            raise ValueError("features and targets must have equal sample counts")

        sample_count = len(matrix)
        if self.max_samples is None:
            draw_count = sample_count
        elif type(self.max_samples) is int:
            if self.max_samples > sample_count:
                raise ValueError("integer max_samples cannot exceed the training sample count")
            draw_count = self.max_samples
        else:
            draw_count = max(1, int(self.max_samples * sample_count))

        generator = random.Random(self.random_state)
        estimators = []
        estimators_samples = []
        target_scale = max(abs(value) for value in target)
        oob_scaled_sums = [0.0] * sample_count if self.oob_score else None
        oob_counts = [0] * sample_count if self.oob_score else None

        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = [generator.randrange(sample_count) for _ in range(draw_count)]
            else:
                indices = list(range(sample_count))
            tree_seed = generator.randrange(0, 2**32)
            estimator = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                random_state=tree_seed,
            )
            estimator.fit(
                [matrix[index] for index in indices],
                [target[index] for index in indices],
            )
            estimators.append(estimator)
            estimators_samples.append(tuple(indices))

            if self.oob_score:
                in_bag = set(indices)
                oob_indices = [
                    index for index in range(sample_count) if index not in in_bag
                ]
                if oob_indices:
                    predictions = estimator.predict([matrix[index] for index in oob_indices])
                    for sample_index, prediction in zip(oob_indices, predictions):
                        if target_scale:
                            oob_scaled_sums[sample_index] += prediction / target_scale
                        oob_counts[sample_index] += 1

        tree_importances = [estimator.feature_importances_ for estimator in estimators]
        importances = [
            math.fsum(values[feature] for values in tree_importances) / self.n_estimators
            for feature in range(len(matrix[0]))
        ]
        importance_std = [
            math.sqrt(
                math.fsum(
                    (values[feature] - importances[feature]) ** 2
                    for values in tree_importances
                ) / self.n_estimators
            )
            for feature in range(len(matrix[0]))
        ]
        total_importance = math.fsum(importances)
        if total_importance > 0.0:
            importances = [value / total_importance for value in importances]

        self.n_features_in_ = len(matrix[0])
        self.n_samples_fit_ = sample_count
        self.max_samples_ = draw_count
        self.estimators_ = estimators
        self.estimators_samples_ = estimators_samples
        self.feature_importances_ = importances
        self.feature_importances_std_ = importance_std

        if self.oob_score:
            predictions = []
            covered_targets = []
            covered_predictions = []
            for index, count in enumerate(oob_counts):
                if count == 0:
                    predictions.append(math.nan)
                    continue
                normalized = oob_scaled_sums[index] / count
                prediction = target_scale * normalized
                if not math.isfinite(prediction):
                    raise ValueError("out-of-bag prediction exceeds finite floating-point range")
                predictions.append(prediction)
                covered_targets.append(target[index])
                covered_predictions.append(prediction)
            self.oob_counts_ = oob_counts
            self.oob_prediction_ = predictions
            self.oob_coverage_ = sum(count > 0 for count in oob_counts) / sample_count
            self.oob_score_ = (
                r2_score(covered_targets, covered_predictions)
                if len(covered_targets) >= 2 else math.nan
            )

        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before prediction")

    def predict(self, features: Iterable[Iterable[Real]]) -> list[float]:
        self._check_fitted()
        matrix = _feature_matrix(features)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        per_tree = [estimator.predict(matrix) for estimator in self.estimators_]
        return [
            _stable_mean(tree_predictions[row_index] for tree_predictions in per_tree)
            for row_index in range(len(matrix))
        ]

    def apply(self, features: Iterable[Iterable[Real]]) -> list[list[int]]:
        """Return sample-major terminal-node indices, one column per tree."""
        self._check_fitted()
        matrix = _feature_matrix(features)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        per_tree = [estimator.apply(matrix) for estimator in self.estimators_]
        return [
            [tree_leaves[row_index] for tree_leaves in per_tree]
            for row_index in range(len(matrix))
        ]

    def score(
        self,
        features: Iterable[Iterable[Real]],
        targets: Iterable[Real],
    ) -> float:
        return r2_score(_target_values(targets), self.predict(features))
