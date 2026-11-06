"""Tests for greedy interval algorithms."""

import unittest

from greedy import merge_intervals


class MergeIntervalsTests(unittest.TestCase):
    def test_merges_overlapping_and_touching_intervals(self):
        intervals = [(8, 10), (1, 3), (2, 6), (6, 7), (12, 12)]
        original = intervals.copy()
        self.assertEqual(
            merge_intervals(intervals),
            [(1, 7), (8, 10), (12, 12)],
        )
        self.assertEqual(intervals, original)

    def test_handles_empty_and_non_overlapping_inputs(self):
        self.assertEqual(merge_intervals([]), [])
        self.assertEqual(merge_intervals([(1, 2), (4, 5)]), [(1, 2), (4, 5)])

    def test_rejects_malformed_or_reversed_intervals(self):
        with self.assertRaises(ValueError):
            merge_intervals([(3, 1)])
        with self.assertRaises(ValueError):
            merge_intervals([(1, 2, 3)])


if __name__ == "__main__":
    unittest.main()
