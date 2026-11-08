"""Regression tests for finite median averaging."""

import unittest
from statistics_utils import median


class MedianOverflowTests(unittest.TestCase):
    def test_averages_large_finite_middle_values_without_overflow(self):
        self.assertEqual(median([1e308, 1e308]), 1e308)
        self.assertEqual(median([-1e308, 1e308]), 0.0)


if __name__ == "__main__":
    unittest.main()
