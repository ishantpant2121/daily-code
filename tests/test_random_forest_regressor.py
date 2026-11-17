"""Tests for bootstrap regression-tree ensembles."""

import math
import unittest

from random_forest_regressor import RandomForestRegressor


class RandomForestRegressorTests(unittest.TestCase):
    def setUp(self):
        self.features = [[0.0], [1.0], [2.0], [3.0], [8.0], [9.0], [10.0], [11.0]]
        self.targets = [0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0]

    def test_fit_predict_and_scores_are_reproducible(self):
        kwargs = {"n_estimators": 31, "max_depth": 3, "random_state": 19}
        first = RandomForestRegressor(**kwargs).fit(self.features, self.targets)
        second = RandomForestRegressor(**kwargs).fit(self.features, self.targets)
        self.assertEqual(first.predict(self.features), second.predict(self.features))
        self.assertEqual(first.feature_importances_, second.feature_importances_)
        self.assertEqual(len(first.estimators_), 31)
        self.assertGreater(first.score(self.features, self.targets), 0.8)

    def test_bootstrap_sizes_and_bootstrap_free_mode(self):
        fractional = RandomForestRegressor(
            n_estimators=5, max_samples=0.5, random_state=2
        ).fit(self.features, self.targets)
        self.assertEqual(fractional.max_samples_, 4)
        self.assertTrue(all(len(sample) == 4 for sample in fractional.estimators_samples_))

        full = RandomForestRegressor(
            n_estimators=3, bootstrap=False, random_state=2
        ).fit(self.features, self.targets)
        self.assertEqual(full.max_samples_, len(self.features))
        self.assertTrue(all(sample == tuple(range(8)) for sample in full.estimators_samples_))

    def test_oob_predictions_and_coverage_are_reproducible(self):
        kwargs = {"n_estimators": 100, "oob_score": True, "random_state": 23}
        first = RandomForestRegressor(**kwargs).fit(self.features, self.targets)
        second = RandomForestRegressor(**kwargs).fit(self.features, self.targets)
        self.assertEqual(first.estimators_samples_, second.estimators_samples_)
        self.assertEqual(first.oob_counts_, second.oob_counts_)
        self.assertEqual(first.oob_prediction_, second.oob_prediction_)
        self.assertTrue(all(count > 0 for count in first.oob_counts_))
        self.assertTrue(all(math.isfinite(value) for value in first.oob_prediction_))
        self.assertAlmostEqual(first.oob_coverage_, 1.0)
        self.assertGreater(first.oob_score_, 0.5)
        self.assertLessEqual(first.oob_score_, 1.0)

    def test_oob_rows_without_votes_are_nan(self):
        model = RandomForestRegressor(
            n_estimators=3, oob_score=True, random_state=4
        ).fit([[1.0]], [5.0])
        self.assertEqual(model.oob_counts_, [0])
        self.assertTrue(math.isnan(model.oob_prediction_[0]))
        self.assertTrue(math.isnan(model.oob_score_))
        self.assertEqual(model.oob_coverage_, 0.0)

    def test_feature_importances_and_leaf_assignments(self):
        features = [[0, 5], [1, 5], [2, 5], [3, 5], [8, 5], [9, 5]]
        targets = [0, 0, 0, 10, 10, 10]
        model = RandomForestRegressor(
            n_estimators=20, max_features=None, random_state=11
        ).fit(features, targets)
        self.assertEqual(len(model.feature_importances_), 2)
        self.assertAlmostEqual(sum(model.feature_importances_), 1.0)
        self.assertEqual(model.feature_importances_[1], 0.0)
        self.assertEqual(len(model.feature_importances_std_), 2)
        leaves = model.apply(features)
        self.assertEqual(len(leaves), len(features))
        self.assertTrue(all(len(row) == 20 for row in leaves))

    def test_accepts_one_shot_iterables(self):
        model = RandomForestRegressor(n_estimators=5, random_state=1).fit(
            (row for row in self.features),
            (value for value in self.targets),
        )
        self.assertEqual(len(model.predict([[1], [9]])), 2)

    def test_stable_prediction_averaging_handles_large_finite_targets(self):
        model = RandomForestRegressor(
            n_estimators=5, bootstrap=False, random_state=1
        ).fit([[0], [1]], [1e308, 1e308])
        self.assertEqual(model.predict([[0], [1]]), [1e308, 1e308])

    def test_rejects_invalid_configuration_and_data(self):
        for kwargs in (
            {"n_estimators": 0},
            {"n_estimators": True},
            {"max_depth": 0},
            {"min_samples_split": 1},
            {"min_samples_leaf": 0},
            {"min_impurity_decrease": -1},
            {"max_features": 0},
            {"max_features": True},
            {"max_samples": 0},
            {"max_samples": 1.1},
            {"max_samples": True},
            {"oob_score": True, "bootstrap": False},
            {"max_samples": 2, "bootstrap": False},
            {"random_state": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RandomForestRegressor(**kwargs)

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
                RandomForestRegressor(n_estimators=2).fit(features, targets)

    def test_rejects_unfitted_prediction_and_feature_width_mismatch(self):
        model = RandomForestRegressor(n_estimators=2)
        with self.assertRaises(RuntimeError):
            model.predict([[1]])
        model.fit([[0, 1], [1, 2]], [0, 1])
        with self.assertRaises(ValueError):
            model.predict([[1]])


if __name__ == "__main__":
    unittest.main()
