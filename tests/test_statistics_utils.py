"""Tests for descriptive statistics helpers."""

import unittest

from statistics_utils import (
    covariance,
    mean,
    median,
    pearson_correlation,
    standard_deviation,
    variance,
)


class DescriptiveStatisticsTests(unittest.TestCase):
    def test_mean_and_median(self):
        self.assertEqual(mean([1, 2, 3, 4]), 2.5)
        self.assertEqual(median([4, 1, 3, 2]), 2.5)
        self.assertEqual(median([5, 1, 3]), 3.0)

    def test_sample_and_population_variance(self):
        self.assertAlmostEqual(variance([1, 2, 3]), 1.0)
        self.assertAlmostEqual(variance([1, 2, 3], sample=False), 2 / 3)
        self.assertAlmostEqual(standard_deviation([1, 2, 3]), 1.0)

    def test_covariance_and_correlation(self):
        self.assertAlmostEqual(covariance([1, 2, 3], [2, 4, 6]), 2.0)
        self.assertAlmostEqual(pearson_correlation([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(pearson_correlation([1, 2, 3], [6, 4, 2]), -1.0)

    def test_mean_and_deviation_avoid_unnecessary_large_intermediates(self):
        self.assertEqual(mean([1e308, 1e308]), 1e308)
        self.assertEqual(mean([1e308, -1e308]), 0.0)
        self.assertEqual(variance([1e308, 1e308], sample=False), 0.0)
        self.assertEqual(
            standard_deviation([1e308, -1e308], sample=False),
            1e308,
        )
        with self.assertRaisesRegex(ValueError, "variance exceeds"):
            variance([1e308, -1e308], sample=False)

    def test_covariance_and_correlation_scale_extreme_values(self):
        self.assertEqual(
            covariance([1e308, -1e308], [1e-308, -1e-308], sample=False),
            1.0,
        )
        self.assertEqual(
            pearson_correlation([1e308, -1e308], [1e308, -1e308]),
            1.0,
        )
        self.assertEqual(
            pearson_correlation([1e308, -1e308], [-1e308, 1e308]),
            -1.0,
        )

    def test_accepts_generators(self):
        self.assertEqual(mean(value for value in [2, 4, 6]), 4.0)
        self.assertEqual(covariance((x for x in [1, 2]), (x for x in [3, 5])), 1.0)

    def test_rejects_empty_invalid_and_mismatched_data(self):
        calls = [
            lambda: mean([]),
            lambda: variance([1]),
            lambda: variance([1, 2], sample=1),
            lambda: mean([1, float("nan")]),
            lambda: mean([10**400]),
            lambda: covariance([10**400, 1], [1, 2]),
            lambda: median([True]),
            lambda: covariance([1, 2], [1]),
            lambda: pearson_correlation([1, 1], [2, 3]),
            lambda: pearson_correlation([1], [2]),
        ]
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()
