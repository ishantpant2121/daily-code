"""Tests for additional sorting algorithms."""

import unittest

from sorting import count_inversions, heap_sort, quickselect, quick_sort


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


class InversionCountTests(unittest.TestCase):
    def test_counts_known_inversions_and_ignores_equal_values(self):
        self.assertEqual(count_inversions([2, 4, 1, 3, 5]), 3)
        self.assertEqual(count_inversions([2, 2, 1]), 2)
        self.assertEqual(count_inversions([1, 1, 1]), 0)

    def test_handles_sorted_reverse_and_empty_inputs(self):
        self.assertEqual(count_inversions([]), 0)
        self.assertEqual(count_inversions([1, 2, 3, 4]), 0)
        self.assertEqual(count_inversions([4, 3, 2, 1]), 6)

    def test_does_not_mutate_input(self):
        values = [3, 1, 2]
        self.assertEqual(count_inversions(values), 2)
        self.assertEqual(values, [3, 1, 2])


class QuickSortScalabilityTests(unittest.TestCase):
    def test_sorts_large_reverse_order_input(self):
        values = list(range(12000, 0, -1))
        self.assertEqual(quick_sort(values), list(range(1, 12001)))

    def test_handles_duplicates_and_preserves_the_input(self):
        values = [3, 1, 2, 3, -1, 2, 2]
        original = values.copy()
        self.assertEqual(quick_sort(values), [-1, 1, 2, 2, 2, 3, 3])
        self.assertEqual(values, original)


if __name__ == "__main__":
    unittest.main()
