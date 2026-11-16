"""Bootstrap-aggregated decision-tree classification without dependencies."""

import math
import random
from collections.abc import Iterable
from numbers import Real
from typing import Any

from decision_tree import DecisionTreeClassifier, _feature_matrix


class RandomForestClassifier:
    """Classify by averaging bootstrap-fitted trees with random node features.

    Each estimator receives a sample drawn with replacement when bootstrap is
    enabled. At every split, its decision tree considers a random subset of
    features. Probability columns are aligned to the complete training class
    order, including classes absent from an individual bootstrap sample.

    When oob_score=True, samples omitted from each bootstrap draw contribute
    to out-of-bag class probabilities and an accuracy estimate. Rows that
    receive no out-of-bag votes contain NaN values in the diagnostic matrix.

    This compact implementation is sequential and intended for learning and
    small datasets; it does not parallelize tree fitting.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = "sqrt",
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

        # Reuse the tree's public parameter validation at construction time.
        DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
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
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = normalized_max_samples
        self.oob_score = oob_score
        self.random_state = random_state
        self._fitted = False

    def fit(
        self,
        features: Iterable[Iterable[Real]],
        labels: Iterable[Any],
    ) -> "RandomForestClassifier":
        matrix = _feature_matrix(features)
        target = list(labels)
        if len(matrix) != len(target):
            raise ValueError("features and labels must have equal sample counts")

        classes = []
        class_positions = {}
        try:
            for label in target:
                if label not in class_positions:
                    class_positions[label] = len(classes)
                    classes.append(label)
        except TypeError:
            raise ValueError("class labels must be hashable") from None

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
        oob_probability_sums = (
            [[0.0] * len(classes) for _ in range(sample_count)]
            if self.oob_score else None
        )
        oob_counts = [0] * sample_count if self.oob_score else None
        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = [generator.randrange(sample_count) for _ in range(draw_count)]
            else:
                indices = list(range(sample_count))
            tree_seed = generator.randrange(0, 2**32)
            estimator = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
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
                    oob_probabilities = estimator.predict_proba(
                        [matrix[index] for index in oob_indices]
                    )
                    for tree_column, label in enumerate(estimator.classes_):
                        forest_column = class_positions[label]
                        for sample_index, row in zip(oob_indices, oob_probabilities):
                            oob_probability_sums[sample_index][forest_column] += row[tree_column]
                    for sample_index in oob_indices:
                        oob_counts[sample_index] += 1

        tree_importances = [
            estimator.feature_importances_ for estimator in estimators
        ]
        importances = [
            math.fsum(values[feature] for values in tree_importances)
            / self.n_estimators
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
        if total_importance > 0:
            importances = [value / total_importance for value in importances]

        self.classes_ = tuple(classes)
        self._class_positions = class_positions
        self.n_features_in_ = len(matrix[0])
        self.n_samples_fit_ = sample_count
        self.max_samples_ = draw_count
        self.estimators_ = estimators
        self.estimators_samples_ = estimators_samples
        self.feature_importances_ = importances
        self.feature_importances_std_ = importance_std
        if self.oob_score:
            decision = []
            correct = 0
            covered = 0
            for sample_index, count in enumerate(oob_counts):
                if count == 0:
                    decision.append([math.nan] * len(classes))
                    continue
                row = [
                    value / count
                    for value in oob_probability_sums[sample_index]
                ]
                decision.append(row)
                predicted_index = max(range(len(row)), key=row.__getitem__)
                correct += classes[predicted_index] == target[sample_index]
                covered += 1
            self.oob_counts_ = oob_counts
            self.oob_decision_function_ = decision
            self.oob_coverage_ = covered / sample_count
            self.oob_score_ = correct / covered if covered else math.nan
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before prediction")

    def predict_proba(
        self,
        features: Iterable[Iterable[Real]],
    ) -> list[list[float]]:
        self._check_fitted()
        matrix = _feature_matrix(features)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")

        probabilities = [[0.0] * len(self.classes_) for _ in matrix]
        contribution = 1.0 / len(self.estimators_)
        for estimator in self.estimators_:
            estimator_probabilities = estimator.predict_proba(matrix)
            for tree_column, label in enumerate(estimator.classes_):
                forest_column = self._class_positions[label]
                for row_index, row in enumerate(estimator_probabilities):
                    probabilities[row_index][forest_column] += row[tree_column] * contribution
        return probabilities

    def apply(self, features: Iterable[Iterable[Real]]) -> list[list[int]]:
        """Return each sample's terminal node index in every fitted tree.

        The result is sample-major: one row per input sample and one column
        per fitted estimator.
        """
        self._check_fitted()
        matrix = _feature_matrix(features)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        per_tree = [estimator.apply(matrix) for estimator in self.estimators_]
        return [
            [tree_leaves[row_index] for tree_leaves in per_tree]
            for row_index in range(len(matrix))
        ]

    def predict_log_proba(
        self,
        features: Iterable[Iterable[Real]],
    ) -> list[list[float]]:
        """Return natural-log class probabilities; zero probabilities map to -inf."""
        return [
            [math.log(value) if value > 0.0 else -math.inf for value in row]
            for row in self.predict_proba(features)
        ]

    def predict_votes(
        self,
        features: Iterable[Iterable[Real]],
    ) -> list[list[int]]:
        """Count hard class votes from each tree for every input sample.

        Columns follow classes_ order. Unlike predict_proba, every tree gives
        exactly one vote regardless of the confidence of its leaf probabilities.
        """
        self._check_fitted()
        matrix = _feature_matrix(features)
        if len(matrix[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        votes = [[0] * len(self.classes_) for _ in matrix]
        for estimator in self.estimators_:
            for row_index, label in enumerate(estimator.predict(matrix)):
                votes[row_index][self._class_positions[label]] += 1
        return votes

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
