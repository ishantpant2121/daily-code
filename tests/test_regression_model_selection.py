"""Regression scorers for cross-validation helpers."""

import math
import unittest

from decision_tree_regressor import DecisionTreeRegressor
from model_selection import cross_val_score, cross_validate


class RegressionCrossValidationTests(unittest.TestCase):
    def setUp(self):
        self.features = [[float(index)] for index in range(12)]
        self.targets = [
            1.5 * index + (4.0 if index >= 6 else 0.0)
            for index in range(12)
        ]

    def test_builtin_regression_scorers_return_finite_maximization_scores(self):
        estimator = DecisionTreeRegressor(max_depth=3, random_state=2)
        for scoring in (
            "r2",
            "neg_mean_squared_error",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ):
            with self.subTest(scoring=scoring):
                scores = cross_val_score(
                    estimator, self.features, self.targets, cv=3, scoring=scoring
                )
                self.assertEqual(len(scores), 3)
                self.assertTrue(all(math.isfinite(score) for score in scores))
                if scoring.startswith("neg_"):
                    self.assertTrue(all(score <= 0.0 for score in scores))

    def test_cross_validate_supports_regression_train_scores_and_estimators(self):
        result = cross_validate(
            DecisionTreeRegressor(max_depth=3),
            self.features,
            self.targets,
            cv=3,
            scoring="neg_mean_squared_error",
            return_train_score=True,
            return_estimator=True,
        )
        self.assertEqual(len(result["test_score"]), 3)
        self.assertEqual(len(result["train_score"]), 3)
        self.assertEqual(len(result["estimator"]), 3)
        self.assertTrue(all(score <= 0.0 for score in result["test_score"]))

    def test_r2_scorer_requires_two_validation_rows(self):
        with self.assertRaises(ValueError):
            cross_val_score(
                DecisionTreeRegressor(),
                [[0.0], [1.0]],
                [0.0, 1.0],
                cv=2,
                scoring="r2",
            )

    def test_unknown_builtin_scorer_is_rejected(self):
        with self.assertRaises(ValueError):
            cross_val_score(
                DecisionTreeRegressor(),
                self.features,
                self.targets,
                cv=3,
                scoring="neg_root_mean_absolute_error",
            )


if __name__ == "__main__":
    unittest.main()
