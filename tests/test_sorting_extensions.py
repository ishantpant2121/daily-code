"""Tests for additional sorting algorithms."""

import unittest

from sorting import heap_sort


class HeapSortTests(unittest.TestCase):
    def test_sorts_mixed_values_without_mutating_input(self):
        values = [4, -2, 4, 0, 9, -8]
        original = values.copy()
        self.assertEqual(heap_sort(values), [-8, -2, 0, 4, 4, 9])
        self.assertEqual(values, original)

    def test_handles_empty_singleton_and_ordered_inputs(self):
        self.assertEqual(heap_sort([]), [])
        self.assertEqual(heap_sort([7]), [7])
        self.assertEqual(heap_sort([1, 2, 3]), [1, 2, 3])
        self.assertEqual(heap_sort([3, 2, 1]), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
