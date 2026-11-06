"""Regression tests for algorithms that accept one-shot iterables."""

import unittest

from dynamic_programming import coin_change
from greedy import activity_selection, fractional_knapsack


class GreedyIterableTests(unittest.TestCase):
    def test_activity_selection_accepts_a_generator(self):
        activities = iter([(1, 3), (2, 4), (3, 5)])
        self.assertEqual(activity_selection(activities), [(1, 3), (3, 5)])

    def test_fractional_knapsack_accepts_a_generator(self):
        items = iter([(60, 10), (100, 20), (120, 30)])
        self.assertAlmostEqual(fractional_knapsack(items, 50), 240)

    def test_invalid_generator_inputs_are_still_rejected(self):
        with self.assertRaises(ValueError):
            activity_selection(iter([(3, 1)]))
        with self.assertRaises(ValueError):
            fractional_knapsack(iter([(10, 0)]), 5)


class DynamicProgrammingIterableTests(unittest.TestCase):
    def test_coin_change_accepts_a_generator(self):
        self.assertEqual(coin_change(iter([1, 2, 5]), 11), 3)

    def test_coin_change_rejects_invalid_generator_values(self):
        with self.assertRaises(ValueError):
            coin_change(iter([1, 0, 5]), 11)


if __name__ == "__main__":
    unittest.main()
