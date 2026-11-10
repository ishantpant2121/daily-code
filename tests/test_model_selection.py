"""Tests for deterministic, leakage-aware cross-validation."""

import unittest

from decision_tree import DecisionTreeClassifier
from logistic_regression import LogisticRegression
from model_selection import cross_val_score
from pipeline import Pipeline
from preprocessing import StandardScaler


class CrossValScoreTests(unittest.TestCase):
    def test_scores_stratified_classification_folds(self):
        features = [[-3], [-2], [-1], [1], [2], [3]]
        labels = [0, 0, 0, 1, 1, 1]
        scores = cross_val_score(
            LogisticRegression(max_iter=1000),
            features,
            labels,
            cv=3,
            scoring="accuracy",
            stratified=True,
            shuffle=True,
            random_state=7,
        )
        self.assertEqual(scores, [1.0, 1.0, 1.0])

    def test_uses_fresh_pipeline_copies_for_each_fold(self):
        estimator = Pipeline([
            ("scale", StandardScaler()),
            ("tree", DecisionTreeClassifier(max_depth=2)),
        ])
        features = [[0], [1], [2], [8], [9], [10]]
        labels = ["left", "left", "left", "right", "right", "right"]
        scores = cross_val_score(
            estimator, features, labels, cv=3, scoring="accuracy", stratified=True
        )
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))
        self.assertFalse(estimator._fitted)

    def test_supports_default_estimator_score_and_custom_scorers(self):
        features = [[0], [1], [2], [3], [4], [5], [10], [11], [12], [13], [14], [15]]
        labels = [0] * 6 + [1] * 6
        estimator = DecisionTreeClassifier(max_depth=2)
        self.assertEqual(
            cross_val_score(estimator, features, labels, cv=3, stratified=True),
            [1.0, 1.0, 1.0],
        )
        custom = cross_val_score(
            estimator,
            features,
            labels,
            cv=3,
            stratified=True,
            scoring=lambda fitted, x, y: fitted.score(x, y),
        )
        self.assertEqual(custom, [1.0, 1.0, 1.0])

    def test_balanced_accuracy_scorer(self):
        features = [[0], [1], [2], [3], [4], [5], [10], [11], [12], [13], [14], [15]]
        labels = [0] * 6 + [1] * 6
        scores = cross_val_score(
            DecisionTreeClassifier(max_depth=2),
            features,
            labels,
            cv=3,
            scoring="balanced_accuracy",
            stratified=True,
        )
        self.assertEqual(scores, [1.0, 1.0, 1.0])

    def test_is_reproducible_and_accepts_one_shot_iterables(self):
        features = ([value] for value in range(8))
        labels = (value // 4 for value in range(8))
        scores = cross_val_score(
            DecisionTreeClassifier(),
            features,
            labels,
            cv=4,
            scoring="accuracy",
            shuffle=True,
            random_state=12,
        )
        repeated = cross_val_score(
            DecisionTreeClassifier(),
            ([value] for value in range(8)),
            (value // 4 for value in range(8)),
            cv=4,
            scoring="accuracy",
            shuffle=True,
            random_state=12,
        )
        self.assertEqual(scores, repeated)

    def test_rejects_invalid_inputs_and_scorer_results(self):
        estimator = DecisionTreeClassifier()
        for kwargs in (
            {"cv": 1},
            {"cv": 7},
            {"cv": True},
            {"stratified": 1},
            {"shuffle": 1},
            {"random_state": True},
            {"scoring": "unknown"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cross_val_score(estimator, [[0], [1], [2]], [0, 1, 1], **kwargs)
        with self.assertRaises(ValueError):
            cross_val_score(estimator, [[0], [1]], [0], cv=2)
        with self.assertRaises(ValueError):
            cross_val_score(
                estimator,
                [[0], [1], [2], [3]],
                [0, 0, 0, 1],
                cv=2,
                stratified=True,
            )
        with self.assertRaises(ValueError):
            cross_val_score(
                estimator,
                [[0], [1], [2], [3]],
                [0, 0, 1, 1],
                cv=2,
                scoring=lambda fitted, x, y: float("nan"),
            )


if __name__ == "__main__":
    unittest.main()
