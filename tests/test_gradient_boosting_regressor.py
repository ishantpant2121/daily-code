"""Tests for the dependency-free gradient boosting regressor."""

import math
import unittest

from gradient_boosting_regressor import GradientBoostingRegressor


class GradientBoostingRegressorTests(unittest.TestCase):
    def setUp(self):
        self.features = [[float(index)] for index in range(6)]
        self.targets = [0.0, 0.0, 0.0, 10.0, 10.0, 10.0]

    def test_learns_piecewise_targets_and_staged_predictions(self):
        model = GradientBoostingRegressor(
            n_estimators=30, learning_rate=0.2, max_depth=1, random_state=7
        ).fit(self.features, self.targets)
        stages = list(model.staged_predict(self.features))
        self.assertEqual(len(stages), 30)
        self.assertEqual(stages[-1], model.predict(self.features))
        self.assertGreater(model.score(self.features, self.targets), 0.99)
        self.assertLess(
            sum((actual - predicted) ** 2 for actual, predicted in zip(self.targets, stages[-1])),
            sum((actual - 5.0) ** 2 for actual in self.targets),
        )

    def test_constant_targets_remain_constant(self):
        model = GradientBoostingRegressor(n_estimators=5, random_state=1).fit(
            [[0], [1], [2]], [3.5, 3.5, 3.5]
        )
        self.assertEqual(model.init_prediction_, 3.5)
        self.assertEqual(model.predict([[4], [5]]), [3.5, 3.5])
        self.assertEqual(model.feature_importances_, [0.0])

    def test_subsampling_is_without_replacement_and_reproducible(self):
        features = [[float(index)] for index in range(20)]
        targets = [float(index % 5) for index in range(20)]
        kwargs = {
            "n_estimators": 8,
            "subsample": 0.6,
            "max_depth": 2,
            "random_state": 17,
        }
        first = GradientBoostingRegressor(**kwargs).fit(features, targets)
        second = GradientBoostingRegressor(**kwargs).fit(features, targets)
        self.assertEqual(first.estimators_samples_, second.estimators_samples_)
        self.assertTrue(all(len(sample) == 12 for sample in first.estimators_samples_))
        self.assertTrue(all(len(set(sample)) == len(sample) for sample in first.estimators_samples_))

    def test_feature_importance_and_apply_are_sample_major(self):
        features = [[0, 5], [1, 5], [2, 5], [3, 5], [8, 5], [9, 5]]
        targets = [0, 0, 0, 10, 10, 10]
        model = GradientBoostingRegressor(
            n_estimators=6, max_depth=1, max_features=None, random_state=3
        ).fit(features, targets)
        self.assertEqual(len(model.feature_importances_), 2)
        self.assertAlmostEqual(sum(model.feature_importances_), 1.0)
        self.assertGreater(model.feature_importances_[0], 0.99)
        self.assertEqual(model.feature_importances_[1], 0.0)
        leaves = model.apply(features)
        self.assertEqual(len(leaves), len(features))
        self.assertTrue(all(len(row) == 6 for row in leaves))

    def test_accepts_one_shot_iterables(self):
        model = GradientBoostingRegressor(n_estimators=3, random_state=2).fit(
            ([value] for value in range(6)),
            (value for value in self.targets),
        )
        self.assertEqual(len(model.predict([[1], [4]])), 2)

    def test_large_finite_targets_are_scaled_during_fitting(self):
        model = GradientBoostingRegressor(n_estimators=4, random_state=1).fit(
            [[0], [1]], [1e308, 1e308]
        )
        self.assertEqual(model.predict([[0], [1]]), [1e308, 1e308])

    def test_rejects_invalid_configuration(self):
        for kwargs in (
            {"n_estimators": 0},
            {"n_estimators": True},
            {"learning_rate": 0},
            {"learning_rate": float("inf")},
            {"learning_rate": True},
            {"subsample": 0},
            {"subsample": 1.1},
            {"subsample": True},
            {"max_depth": 0},
            {"min_samples_split": 1},
            {"min_samples_leaf": 0},
            {"max_features": 0},
            {"max_features": "auto"},
            {"random_state": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GradientBoostingRegressor(**kwargs)

    def test_rejects_invalid_data_and_prediction_before_fit(self):
        model = GradientBoostingRegressor(n_estimators=2)
        with self.assertRaises(RuntimeError):
            model.predict([[1]])
        for features, targets in (
            ([], []),
            ([[0], [1]], [0]),
            ([[True], [1]], [0, 1]),
            ([[0], [1]], [0, float("nan")]),
            ([[0], [1]], [0, float("inf")]),
            ([[0], [1]], [0, True]),
            ([[0], [1]], [0, 10 ** 400]),
        ):
            with self.subTest(features=features, targets=targets), self.assertRaises(ValueError):
                GradientBoostingRegressor(n_estimators=2).fit(features, targets)

        model.fit([[0, 1], [1, 2]], [0, 1])
        with self.assertRaises(ValueError):
            model.predict([[1]])

    def test_rejects_predictions_outside_finite_float_range(self):
        model = GradientBoostingRegressor(
            n_estimators=1, learning_rate=1e308, max_depth=1, random_state=1
        ).fit([[0], [1]], [0, 10])
        with self.assertRaises(ValueError):
            model.predict([[0], [1]])


if __name__ == "__main__":
    unittest.main()
