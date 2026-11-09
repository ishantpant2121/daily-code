"""Tests for dependency-free regression metrics."""

import math
import unittest

from regression_metrics import (
    mean_absolute_error,
    median_absolute_error,
    max_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


class RegressionMetricsTests(unittest.TestCase):
    def test_known_error_metrics(self):
        actual = [3, -0.5, 2, 7]
        predicted = [2.5, 0, 2, 8]
        self.assertAlmostEqual(mean_absolute_error(actual, predicted), 0.5)
        self.assertAlmostEqual(mean_squared_error(actual, predicted), 0.375)
        self.assertAlmostEqual(root_mean_squared_error(actual, predicted), math.sqrt(0.375))
        self.assertAlmostEqual(r2_score(actual, predicted), 0.9486081370449679)

    def test_median_absolute_and_maximum_error(self):
        actual = [3, -0.5, 2, 7]
        predicted = [2.5, 0, 2, 8]
        self.assertEqual(median_absolute_error(actual, predicted), 0.5)
        self.assertEqual(max_error(actual, predicted), 1.0)
        self.assertEqual(median_absolute_error([1, 2, 3], [1, 4, 3]), 0.0)

    def test_new_error_metrics_accept_generators_and_reject_overflow(self):
        self.assertEqual(
            median_absolute_error((x for x in [1, 2]), (x for x in [2, 4])),
            1.5,
        )
        with self.assertRaises(ValueError):
            max_error([1e308], [-1e308])

    def test_accepts_one_shot_iterables(self):
        self.assertEqual(
            mean_absolute_error((x for x in [1, 2]), (x for x in [2, 4])),
            1.5,
        )

    def test_r2_can_be_negative(self):
        self.assertLess(r2_score([1, 2, 3], [5, 5, 5]), 0)

    def test_constant_targets_follow_finite_policy(self):
        self.assertEqual(r2_score([4, 4], [4, 4]), 1.0)
        self.assertEqual(r2_score([4, 4], [3, 5]), 0.0)
        self.assertTrue(math.isnan(r2_score([4, 4], [4, 4], force_finite=False)))
        self.assertEqual(r2_score([4, 4], [3, 5], force_finite=False), -math.inf)

    def test_rejects_empty_mismatched_non_numeric_and_non_finite_inputs(self):
        invalid_cases = [
            ([], []),
            ([1, 2], [1]),
            ([True], [1]),
            ([1], ["1"]),
            ([float("nan")], [1]),
            ([1], [float("inf")]),
            ([10**400], [1]),
        ]
        for actual, predicted in invalid_cases:
            with self.subTest(actual=actual, predicted=predicted):
                with self.assertRaises(ValueError):
                    mean_squared_error(actual, predicted)

    def test_r2_requires_two_samples_and_boolean_policy(self):
        with self.assertRaises(ValueError):
            r2_score([1], [1])
        with self.assertRaises(ValueError):
            r2_score([1, 2], [1, 2], force_finite=1)


if __name__ == "__main__":
    unittest.main()
