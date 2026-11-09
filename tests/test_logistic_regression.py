"""Tests for dependency-free binary logistic regression."""

import unittest

from logistic_regression import LogisticRegression


class LogisticRegressionTests(unittest.TestCase):
    def test_learns_a_separable_binary_problem(self):
        model = LogisticRegression(learning_rate=0.1, max_iter=3000).fit(
            [[-2], [-1], [1], [2]],
            ["negative", "negative", "positive", "positive"],
        )
        self.assertEqual(
            model.predict([[-2], [-1], [1], [2]]),
            ["negative", "negative", "positive", "positive"],
        )
        self.assertEqual(model.classes_, ("negative", "positive"))
        self.assertEqual(model.score([[-2], [-1], [1], [2]], [
            "negative", "negative", "positive", "positive"
        ]), 1.0)

    def test_probabilities_are_normalized_and_decision_scores_are_ordered(self):
        model = LogisticRegression(max_iter=100).fit(
            [[0], [1], [2], [3]],
            [0, 0, 1, 1],
        )
        probabilities = model.predict_proba([[0], [2], [3]])
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-12 for row in probabilities))
        scores = model.decision_function([[0], [2], [3]])
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])

    def test_accepts_one_shot_iterables(self):
        model = LogisticRegression(max_iter=100).fit(
            ([value] for value in [-2, -1, 1, 2]),
            (label for label in ["a", "a", "b", "b"]),
        )
        self.assertEqual(model.predict([[2]]), ["b"])

    def test_l2_regularization_reduces_coefficient_magnitude(self):
        features = [[-2], [-1], [0], [1], [2]]
        labels = [0, 0, 1, 1, 1]
        plain = LogisticRegression(max_iter=1000).fit(features, labels)
        regularized = LogisticRegression(max_iter=1000, l2=2.0).fit(features, labels)
        self.assertLess(abs(regularized.coef_[0]), abs(plain.coef_[0]))

    def test_rejects_invalid_options_data_and_unfitted_use(self):
        for kwargs in (
            {"learning_rate": 0},
            {"learning_rate": float("inf")},
            {"max_iter": 0},
            {"max_iter": True},
            {"tol": -1},
            {"l2": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LogisticRegression(**kwargs)
        with self.assertRaises(RuntimeError):
            LogisticRegression().predict([[1]])
        for features, labels in (
            ([[1], [2]], [0]),
            ([[1], [2]], [1, 1]),
            ([[True], [2]], [0, 1]),
            ([[10**400], [2]], [0, 1]),
            ([[1], [2, 3]], [0, 1]),
        ):
            with self.subTest(features=features, labels=labels), self.assertRaises(ValueError):
                LogisticRegression().fit(features, labels)

    def test_rejects_feature_width_and_score_length_mismatches(self):
        model = LogisticRegression(max_iter=10).fit([[0], [1]], [0, 1])
        with self.assertRaises(ValueError):
            model.predict([[1, 2]])
        with self.assertRaises(ValueError):
            model.score([[1]], [0, 1])


if __name__ == "__main__":
    unittest.main()
