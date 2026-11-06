"""Tests for additional sorting algorithms."""

import unittest

from sorting import heap_sort, quickselect


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


class QuickselectTests(unittest.TestCase):
    def test_selects_each_order_statistic_with_duplicates(self):
        values = [7, 2, 9, 2, -1, 5]
        for index, expected in enumerate(sorted(values)):
            with self.subTest(index=index):
                self.assertEqual(quickselect(values, index), expected)

    def test_does_not_mutate_input(self):
        values = [3, 1, 2]
        quickselect(values, 1)
        self.assertEqual(values, [3, 1, 2])

    def test_rejects_out_of_range_indices(self):
        for index in (-1, 0):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    quickselect([], index)
        with self.assertRaises(ValueError):
            quickselect([1, 2], 2)


if __name__ == "__main__":
    unittest.main()
