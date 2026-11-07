"""Tests for heap-based selection and merging."""

import unittest

from heaps import kth_largest, merge_sorted_lists


class KthLargestTests(unittest.TestCase):
    def test_accepts_a_one_shot_iterable(self):
        values = (value for value in [3, 1, 5, 2, 4])
        self.assertEqual(kth_largest(values, 2), 4)

    def test_keeps_duplicates_as_separate_ordered_values(self):
        self.assertEqual(kth_largest([5, 5, 1, 5, 2], 3), 5)

    def test_rejects_invalid_k_and_short_iterables(self):
        for k in (0, -1, 1.5, True):
            with self.subTest(k=k):
                with self.assertRaises(ValueError):
                    kth_largest([1, 2], k)
        with self.assertRaises(ValueError):
            kth_largest(iter([1, 2]), 3)


class MergeSortedListsTests(unittest.TestCase):
    def test_merges_empty_and_duplicate_values(self):
        self.assertEqual(merge_sorted_lists([[1, 3], [], [1, 2, 4]]), [1, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
