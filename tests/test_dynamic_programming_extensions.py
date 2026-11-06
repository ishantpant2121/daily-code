"""Tests for additional dynamic programming algorithms."""

import unittest

from dynamic_programming import knapsack_01


class Knapsack01Tests(unittest.TestCase):
    def test_solves_classic_capacity_constraint(self):
        self.assertEqual(knapsack_01([60, 100, 120], [10, 20, 30], 50), 220)

    def test_handles_zero_weight_and_zero_capacity(self):
        self.assertEqual(knapsack_01([5], [0], 0), 5)
        self.assertEqual(knapsack_01([], [], 0), 0)

    def test_rejects_invalid_dimensions_and_weights(self):
        with self.assertRaises(ValueError):
            knapsack_01([1], [], 5)
        with self.assertRaises(ValueError):
            knapsack_01([1], [-1], 5)
        with self.assertRaises(ValueError):
            knapsack_01([1], [1], -1)


if __name__ == "__main__":
    unittest.main()
