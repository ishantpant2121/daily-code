"""Regression tests for array-search helpers."""

from itertools import combinations_with_replacement
import unittest

from searching import linear_search, lower_bound, rotated_search


class LinearSearchTests(unittest.TestCase):
    def test_returns_first_match_and_handles_missing_values(self):
        self.assertEqual(linear_search([4, 2, 4, 1], 4), 0)
        self.assertEqual(linear_search([4, 2, 4, 1], 9), -1)
        self.assertEqual(linear_search([], 1), -1)


class LowerBoundTests(unittest.TestCase):
    def test_finds_first_duplicate(self):
        self.assertEqual(lower_bound([1, 2, 2, 2, 5], 2), 1)

    def test_handles_values_outside_the_range(self):
        values = [2, 4, 6]
        self.assertEqual(lower_bound(values, 1), 0)
        self.assertEqual(lower_bound(values, 7), len(values))
        self.assertEqual(lower_bound([], 3), 0)


class RotatedSearchTests(unittest.TestCase):
    def test_searches_rotated_array_without_duplicates(self):
        values = [4, 5, 6, 7, 0, 1, 2]
        self.assertEqual(rotated_search(values, 0), 4)
        self.assertEqual(rotated_search(values, 6), 2)
        self.assertEqual(rotated_search(values, 3), -1)

    def test_searches_when_duplicates_make_the_sorted_half_ambiguous(self):
        cases = (
            ([1, 3, 1, 1, 1], 3),
            ([1, 1, 1, 3, 1], 3),
            ([2, 2, 2, 3, 4, 2], 3),
            ([1, 1, 1, 1], 1),
        )
        for values, target in cases:
            with self.subTest(values=values, target=target):
                index = rotated_search(values, target)
                self.assertGreaterEqual(index, 0)
                self.assertEqual(values[index], target)

    def test_handles_empty_singleton_and_missing_values(self):
        self.assertEqual(rotated_search([], 1), -1)
        self.assertEqual(rotated_search([5], 5), 0)
        self.assertEqual(rotated_search([5], 2), -1)
        self.assertEqual(rotated_search([1, 1, 1, 1], 2), -1)

    def test_all_small_sorted_multisets_and_rotations(self):
        # Exhaustively check short arrays with duplicates and every pivot.
        for length in range(7):
            for ordered in combinations_with_replacement(range(3), length):
                for pivot in range(max(1, length)):
                    values = list(ordered[pivot:]) + list(ordered[:pivot])
                    for target in range(-1, 4):
                        with self.subTest(values=values, target=target):
                            index = rotated_search(values, target)
                            if target in values:
                                self.assertGreaterEqual(index, 0)
                                self.assertEqual(values[index], target)
                            else:
                                self.assertEqual(index, -1)


if __name__ == "__main__":
    unittest.main()
