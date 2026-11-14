"""Focused tests for bounded per-column example retention."""

import unittest

from csv_inspector.inference import ColumnAccumulator


class ColumnAccumulatorExampleTests(unittest.TestCase):
    def test_examples_remain_unique_and_ordered_without_distinct_tracking(self):
        accumulator = ColumnAccumulator(
            track_distinct=False,
            max_examples=3,
        )
        for value in ("alpha", "alpha", "beta", "alpha", "gamma", "delta"):
            accumulator.add(value)

        profile = accumulator.snapshot("tag", 1)
        self.assertEqual(profile.examples, ("alpha", "beta", "gamma"))
        self.assertIsNone(profile.distinct_count)

    def test_examples_stop_growing_after_the_configured_limit(self):
        accumulator = ColumnAccumulator(max_examples=2)
        for value in ("alpha", "beta", "alpha", "gamma", "delta"):
            accumulator.add(value)

        profile = accumulator.snapshot("tag", 1)
        self.assertEqual(profile.examples, ("alpha", "beta"))
        self.assertEqual(profile.distinct_count, 4)


if __name__ == "__main__":
    unittest.main()
