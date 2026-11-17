"""Tests for dependency-free CART regression trees."""

import unittest

from decision_tree_regressor import DecisionTreeRegressor


class DecisionTreeRegressorTests(unittest.TestCase):
    def test_learns_a_piecewise_regression_problem(self):
        features = [[0], [1], [2], [3], [4], [5]]
        targets = [0, 1, 2, 8, 9, 10]
        model = DecisionTreeRegressor(max_depth=2).fit(features, targets)
        predictions = model.predict(features)
        self.assertLess(sum((a - b) ** 2 for a, b in zip(targets, predictions)), 2.1)
        self.assertEqual(model.max_depth_, 2)
        self.assertEqual(model.node_count_, 7)
        self.assertEqual(model.n_leaves_, 4)
        self.assertAlmostEqual(model.score(features, targets), 0.99)

    def test_thresholds_split_between_adjacent_training_values(self):
        model = DecisionTreeRegressor(max_depth=1).fit(
            [[0.0], [1.0], [3.0], [4.0]], [0.0, 0.0, 10.0, 10.0]
        )
        self.assertEqual(model.predict([[1.5], [2.0], [2.5]]), [0.0, 0.0, 10.0])

    def test_leaf_constraints_and_minimum_impurity_decrease(self):
        features = [[0], [1], [2], [3]]
        targets = [0, 0, 10, 10]
        constrained = DecisionTreeRegressor(min_samples_leaf=2).fit(features, targets)
        self.assertEqual(constrained.n_leaves_, 2)
        self.assertEqual(constrained.predict(features), targets)
        stopped = DecisionTreeRegressor(min_impurity_decrease=100.0).fit(features, targets)
        self.assertEqual(stopped.node_count_, 1)

    def test_constant_targets_produce_a_single_leaf(self):
        model = DecisionTreeRegressor().fit([[0, 1], [1, 0], [2, 1]], [5, 5, 5])
        self.assertEqual(model.predict([[3, 1]]), [5.0])
        self.assertEqual(model.feature_importances_, [0.0, 0.0])
        self.assertEqual(model.max_depth_, 0)

    def test_feature_importances_identify_the_predictive_column(self):
        features = [[0, 5], [1, 5], [2, 5], [3, 5], [8, 5], [9, 5]]
        targets = [0, 0, 0, 10, 10, 10]
        model = DecisionTreeRegressor().fit(features, targets)
        self.assertEqual(model.feature_importances_[1], 0.0)
        self.assertAlmostEqual(sum(model.feature_importances_), 1.0)
        self.assertGreater(model.feature_importances_[0], 0.99)

    def test_random_feature_selection_is_reproducible(self):
        features = [
            [0, 0, 1], [1, 0, 1], [2, 1, 0], [3, 1, 0],
            [8, 0, 1], [9, 0, 1], [10, 1, 0], [11, 1, 0],
        ]
        targets = [0, 0, 0, 0, 10, 10, 10, 10]
        kwargs = {"max_features": 2, "random_state": 19, "max_depth": 3}
        first = DecisionTreeRegressor(**kwargs).fit(features, targets)
        second = DecisionTreeRegressor(**kwargs).fit(features, targets)
        self.assertEqual(first.predict(features), second.predict(features))
        self.assertEqual(first.feature_importances_, second.feature_importances_)

    def test_accepts_one_shot_iterables_and_apply_returns_leaf_ids(self):
        model = DecisionTreeRegressor().fit(
            ([value] for value in [0, 1, 2, 3]),
            (value for value in [0, 0, 5, 5]),
        )
        leaves = model.apply([[0], [3]])
        self.assertEqual(len(leaves), 2)
        self.assertNotEqual(leaves[0], leaves[1])

    def test_stable_mean_handles_large_finite_targets(self):
        model = DecisionTreeRegressor(max_depth=1).fit(
            [[0], [1]], [1e308, 1e308]
        )
        self.assertEqual(model.predict([[0], [1]]), [1e308, 1e308])

    def test_rejects_invalid_configuration_and_data(self):
        for kwargs in (
            {"max_depth": 0},
            {"max_depth": True},
            {"min_samples_split": 1},
            {"min_samples_leaf": 0},
            {"min_impurity_decrease": -1},
            {"min_impurity_decrease": float("nan")},
            {"min_impurity_decrease": True},
            {"max_features": 0},
            {"max_features": 1.1},
            {"max_features": True},
            {"max_features": "auto"},
            {"random_state": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DecisionTreeRegressor(**kwargs)

        for features, targets in (
            ([], []),
            ([[1], [2]], [0]),
            ([[True], [2]], [0, 1]),
            ([[float("inf")], [2]], [0, 1]),
            ([[1], [2, 3]], [0, 1]),
            ([[1], [2]], [0, float("nan")]),
            ([[1], [2]], [0, float("inf")]),
            ([[1], [2]], [0, True]),
            ([[1], [2]], [0, 10 ** 400]),
        ):
            with self.subTest(features=features, targets=targets), self.assertRaises(ValueError):
                DecisionTreeRegressor().fit(features, targets)

    def test_rejects_unfitted_prediction_and_feature_width_mismatch(self):
        model = DecisionTreeRegressor()
        with self.assertRaises(RuntimeError):
            model.predict([[1]])
        model.fit([[0, 1], [1, 2]], [0, 1])
        with self.assertRaises(ValueError):
            model.predict([[1]])
        with self.assertRaises(ValueError):
            model.apply([[1]])


if __name__ == "__main__":
    unittest.main()
