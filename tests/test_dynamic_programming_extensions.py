"""Tests for additional dynamic programming algorithms."""

import unittest

from dynamic_programming import (
    edit_distance,
    knapsack_01,
    longest_common_subsequence_length,
)


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


class LCSLengthTests(unittest.TestCase):
    def test_computes_common_subsequence_length(self):
        self.assertEqual(longest_common_subsequence_length("abcde", "ace"), 3)
        self.assertEqual(longest_common_subsequence_length("abc", "abc"), 3)
        self.assertEqual(longest_common_subsequence_length("aaaa", "aa"), 2)

    def test_handles_empty_sequences(self):
        self.assertEqual(longest_common_subsequence_length("", "abc"), 0)
        self.assertEqual(longest_common_subsequence_length([], []), 0)


class EditDistanceTests(unittest.TestCase):
    def test_known_edit_distances(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance("intention", "execution"), 5)
        self.assertEqual(edit_distance("same", "same"), 0)

    def test_handles_empty_strings(self):
        self.assertEqual(edit_distance("", "abc"), 3)
        self.assertEqual(edit_distance("abc", ""), 3)


if __name__ == "__main__":
    unittest.main()
