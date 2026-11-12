"""Tests for deterministic finite hyperparameter candidate utilities."""

import unittest
from unittest.mock import patch

from hyperparameter_search import ParameterGrid, ParameterSampler


class ParameterGridTests(unittest.TestCase):
    def test_enumerates_cartesian_product_in_sorted_key_order(self):
        grid = ParameterGrid({"max_depth": [1, 2], "criterion": ["gini", "entropy"]})
        self.assertEqual(len(grid), 4)
        self.assertEqual(
            list(grid),
            [
                {"criterion": "gini", "max_depth": 1},
                {"criterion": "gini", "max_depth": 2},
                {"criterion": "entropy", "max_depth": 1},
                {"criterion": "entropy", "max_depth": 2},
            ],
        )

    def test_empty_mapping_and_empty_values(self):
        self.assertEqual(list(ParameterGrid({})), [{}])
        self.assertEqual(len(ParameterGrid({"depth": []})), 0)
        self.assertEqual(list(ParameterGrid({"depth": []})), [])

    def test_indexing_supports_negative_indices(self):
        grid = ParameterGrid({"depth": [1, 2, 3]})
        self.assertEqual(grid[0], {"depth": 1})
        self.assertEqual(grid[-1], {"depth": 3})
        with self.assertRaises(IndexError):
            _ = grid[3]
        with self.assertRaises(TypeError):
            _ = grid[1.5]

    def test_indexing_matches_product_order_without_iterating(self):
        grid = ParameterGrid({
            "z": [10, 20],
            "a": ["x", "y"],
            "middle": [0, 1, 2],
        })
        expected = list(grid)
        with patch.object(
            ParameterGrid,
            "__iter__",
            side_effect=AssertionError("indexed access must not scan the grid"),
        ):
            actual = [grid[index] for index in range(len(grid))]
        self.assertEqual(actual, expected)


class ParameterSamplerTests(unittest.TestCase):
    def test_sampling_is_reproducible_and_without_replacement(self):
        parameters = {"depth": [1, 2, 3], "criterion": ["gini", "entropy"]}
        first = list(ParameterSampler(parameters, 4, random_state=23))
        second = list(ParameterSampler(parameters, 4, random_state=23))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len({tuple(sorted(item.items())) for item in first}), 4)

    def test_large_grid_sampling_does_not_enumerate_cartesian_product(self):
        parameters = {
            "depth": range(100),
            "leaf": range(100),
            "feature": range(100),
        }
        with patch.object(
            ParameterGrid,
            "__iter__",
            side_effect=AssertionError("sampling must not enumerate the full grid"),
        ):
            sampler = ParameterSampler(parameters, 7, random_state=41)
            first = list(sampler)

        second = list(ParameterSampler(parameters, 7, random_state=41))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        self.assertEqual(
            len({tuple(sorted(item.items())) for item in first}),
            7,
        )

    def test_rejects_invalid_sample_counts_and_random_state(self):
        parameters = {"depth": [1, 2]}
        for count in (0, -1, True, 1.5, 3):
            with self.subTest(count=count), self.assertRaises(ValueError):
                ParameterSampler(parameters, count, random_state=1)
        with self.assertRaises(ValueError):
            ParameterSampler(parameters, 1, random_state=True)

    def test_rejects_non_sequence_parameter_values(self):
        with self.assertRaises(ValueError):
            ParameterGrid({"depth": "123"})
        with self.assertRaises(ValueError):
            ParameterGrid({"depth": (value for value in [1, 2])})


class GridSearchCVTests(unittest.TestCase):
    def test_selects_parameters_and_refits_best_estimator(self):
        from decision_tree import DecisionTreeClassifier
        from hyperparameter_search import GridSearchCV

        features = [[0], [1], [2], [3], [10], [11], [12], [13]]
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        search = GridSearchCV(
            DecisionTreeClassifier(),
            {"max_depth": [1, 2, 3]},
            cv=4,
            scoring="accuracy",
            stratified=True,
        ).fit(features, labels)
        self.assertEqual(search.best_score_, 1.0)
        self.assertEqual(search.best_params_, {"max_depth": 1})
        self.assertEqual(search.predict(features), labels)
        self.assertEqual(len(search.cv_results_["mean_test_score"]), 3)
        self.assertTrue(search.best_estimator_._fitted)

    def test_aggregates_extreme_finite_scores_without_overflow(self):
        class ExtremeScoreEstimator:
            def __init__(self, marker=0):
                self.marker = marker

            def fit(self, features, labels):
                return self

        features = [[1e308], [0.0], [-1e308], [0.0]]
        labels = [0, 0, 0, 0]
        search = GridSearchCV(
            ExtremeScoreEstimator(),
            {"marker": [0]},
            cv=2,
            scoring=lambda fitted, x, y: x[0][0],
        ).fit(features, labels)
        self.assertEqual(search.best_score_, 0.0)
        self.assertEqual(search.cv_results_["std_test_score"], [1e308])

        constant = GridSearchCV(
            ExtremeScoreEstimator(),
            {"marker": [0]},
            cv=2,
            scoring=lambda fitted, x, y: 1e308,
        ).fit(features, labels)
        self.assertEqual(constant.best_score_, 1e308)
        self.assertEqual(constant.cv_results_["std_test_score"], [0.0])

    def test_supports_pipeline_step_parameters(self):
        from decision_tree import DecisionTreeClassifier
        from hyperparameter_search import GridSearchCV
        from pipeline import Pipeline
        from preprocessing import StandardScaler

        features = [[0], [1], [2], [8], [9], [10]]
        labels = [0, 0, 0, 1, 1, 1]
        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("tree", DecisionTreeClassifier()),
        ])
        search = GridSearchCV(
            pipeline, {"tree__max_depth": [1, 2]}, cv=3,
            scoring="accuracy", stratified=True,
        ).fit(features, labels)
        self.assertEqual(search.best_params_, {"tree__max_depth": 1})
        self.assertEqual(search.predict(features), labels)

    def test_requires_fit_and_rejects_unknown_parameters(self):
        from decision_tree import DecisionTreeClassifier
        from hyperparameter_search import GridSearchCV

        search = GridSearchCV(DecisionTreeClassifier(), {"not_a_parameter": [1]}, cv=2)
        with self.assertRaises(RuntimeError):
            search.predict([[0]])
        with self.assertRaises(ValueError):
            search.fit([[0], [1], [2], [3]], [0, 0, 1, 1])



class RandomizedSearchCVTests(unittest.TestCase):
    def test_reproducibly_evaluates_distinct_candidates(self):
        from decision_tree import DecisionTreeClassifier
        from hyperparameter_search import RandomizedSearchCV

        features = [[0], [1], [2], [3], [10], [11], [12], [13]]
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        kwargs = dict(
            estimator=DecisionTreeClassifier(),
            param_distributions={"max_depth": [1, 2, 3], "criterion": ["gini", "entropy"]},
            n_iter=4,
            cv=4,
            scoring="accuracy",
            stratified=True,
            random_state=17,
        )
        first = RandomizedSearchCV(**kwargs).fit(features, labels)
        second = RandomizedSearchCV(**kwargs).fit(features, labels)
        self.assertEqual(first.sampled_params_, second.sampled_params_)
        self.assertEqual(len(first.sampled_params_), 4)
        self.assertEqual(len({tuple(sorted(item.items())) for item in first.sampled_params_}), 4)
        self.assertEqual(first.best_params_, second.best_params_)
        self.assertEqual(first.best_score_, 1.0)

    def test_rejects_invalid_iteration_count(self):
        from decision_tree import DecisionTreeClassifier
        from hyperparameter_search import RandomizedSearchCV

        for count in (0, -1, True, 1.5, 3):
            with self.subTest(count=count), self.assertRaises(ValueError):
                RandomizedSearchCV(DecisionTreeClassifier(), {"max_depth": [1, 2]},
                                   n_iter=count, cv=2)



if __name__ == "__main__":
    unittest.main()
