"""Regression hyperparameter-search integration tests."""

import unittest

from gradient_boosting_regressor import GradientBoostingRegressor
from hyperparameter_search import GridSearchCV, RandomizedSearchCV


class RegressionHyperparameterSearchTests(unittest.TestCase):
    def setUp(self):
        self.features = [[float(index)] for index in range(18)]
        self.targets = [
            0.4 * index + (3.0 if index >= 9 else 0.0)
            for index in range(18)
        ]
        self.parameters = {
            "learning_rate": [0.05, 0.1],
            "n_estimators": [5, 10],
        }

    def test_grid_search_uses_negative_regression_loss(self):
        search = GridSearchCV(
            GradientBoostingRegressor(max_depth=2, random_state=4),
            self.parameters,
            cv=3,
            scoring="neg_mean_squared_error",
        ).fit(self.features, self.targets)
        self.assertEqual(len(search.cv_results_["mean_test_score"]), 4)
        self.assertLessEqual(search.best_score_, 0.0)
        self.assertEqual(len(search.predict(self.features)), len(self.targets))
        self.assertTrue(search.best_estimator_._fitted)

    def test_randomized_search_uses_regression_scorer(self):
        search = RandomizedSearchCV(
            GradientBoostingRegressor(max_depth=2, random_state=4),
            self.parameters,
            n_iter=3,
            cv=3,
            scoring="neg_mean_absolute_error",
            random_state=11,
        ).fit(self.features, self.targets)
        self.assertEqual(len(search.sampled_params_), 3)
        self.assertLessEqual(search.best_score_, 0.0)
        self.assertEqual(len(search.predict(self.features)), len(self.targets))


if __name__ == "__main__":
    unittest.main()
