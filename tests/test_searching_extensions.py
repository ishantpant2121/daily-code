"""Tests for binary-search range queries."""

import unittest

from searching import search_range


class SearchRangeTests(unittest.TestCase):
    def test_returns_first_and_last_duplicate_positions(self):
        self.assertEqual(search_range([1, 2, 2, 2, 5], 2), (1, 3))

    def test_handles_missing_and_boundary_values(self):
        self.assertEqual(search_range([2, 4, 6], 1), (-1, -1))
        self.assertEqual(search_range([2, 4, 6], 7), (-1, -1))
        self.assertEqual(search_range([], 1), (-1, -1))
        self.assertEqual(search_range([5], 5), (0, 0))

    def test_handles_all_equal_values(self):
        self.assertEqual(search_range([3, 3, 3, 3], 3), (0, 3))


if __name__ == "__main__":
    unittest.main()
