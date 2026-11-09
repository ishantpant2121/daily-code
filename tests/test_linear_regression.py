"""Tests for simple univariate ordinary least-squares regression."""

import unittest

from linear_regression import SimpleLinearRegression


class SimpleLinearRegressionTests(unittest.TestCase):
    def test_recovers_exact_line_and_predicts_unseen_values(self):
        model = SimpleLinearRegression().fit([1, 2, 3, 4], [3, 5, 7, 9])
        self.assertAlmostEqual(model.coef_, 2.0)
        self.assertAlmostEqual(model.intercept_, 1.0)
        self.assertEqual(model.predict([0, 5]), [1.0, 11.0])
        self.assertAlmostEqual(model.score([1, 2, 3, 4], [3, 5, 7, 9]), 1.0)

    def test_accepts_one_shot_iterables(self):
        model = SimpleLinearRegression().fit(
            (x for x in [1, 2, 3]), (y for y in [2, 4, 6])
        )
        self.assertEqual(model.predict([4]), [8.0])

    def test_rejects_prediction_before_fit(self):
        with self.assertRaises(RuntimeError):
            SimpleLinearRegression().predict([1])

    def test_rejects_invalid_training_data(self):
        cases = [
            lambda: SimpleLinearRegression().fit([1], [2]),
            lambda: SimpleLinearRegression().fit([1, 1], [2, 3]),
            lambda: SimpleLinearRegression().fit([1, 2], [1]),
            lambda: SimpleLinearRegression().fit([1, float("inf")], [2, 3]),
            lambda: SimpleLinearRegression().fit([True, 2], [2, 3]),
            lambda: SimpleLinearRegression().fit([10**400, 2], [2, 3]),
        ]
        for call in cases:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()

    def test_rejects_non_numeric_prediction_values(self):
        model = SimpleLinearRegression().fit([1, 2], [2, 4])
        with self.assertRaises(ValueError):
            model.predict(["three"])


if __name__ == "__main__":
    unittest.main()
