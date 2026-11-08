"""Tests for linearly interpolated sample quantiles."""

import unittest
from statistics_utils import quantile


class QuantileTests(unittest.TestCase):
    def test_interpolates_known_quantiles_and_endpoints(self):
        values = [1, 2, 3, 4]
        self.assertEqual(quantile(values, 0), 1.0)
        self.assertEqual(quantile(values, 0.5), 2.5)
        self.assertEqual(quantile(values, 1), 4.0)
        self.assertEqual(quantile(values, 0.25), 1.75)

    def test_accepts_generators_and_single_values(self):
        self.assertEqual(quantile((value for value in [2, 4, 8]), 0.5), 4.0)
        self.assertEqual(quantile([7], 0.9), 7.0)

    def test_rejects_invalid_quantiles_and_empty_values(self):
        for q in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(q=q), self.assertRaises(ValueError):
                quantile([1, 2], q)
        with self.assertRaises(ValueError):
            quantile([], 0.5)


if __name__ == "__main__":
    unittest.main()
