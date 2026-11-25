import math
import unittest

from interval_diagnostics import interval_diagnostics


class IntervalDiagnosticsTests(unittest.TestCase):
    def test_overall_coverage_width_and_winkler_score(self):
        result = interval_diagnostics(
            [0.0, 2.0, 5.0],
            [(-1.0, 1.0), (1.0, 3.0), (0.0, 2.0)],
            alpha=0.1,
        )
        summary = result.overall
        self.assertEqual(summary.count, 3)
        self.assertEqual(summary.covered, 2)
        self.assertEqual(summary.below, 0)
        self.assertEqual(summary.above, 1)
        self.assertAlmostEqual(summary.coverage, 2 / 3)
        self.assertAlmostEqual(summary.mean_width, 2.0)
        self.assertAlmostEqual(summary.median_width, 2.0)
        self.assertAlmostEqual(summary.mean_interval_score, (2 + 2 + 62) / 3)

    def test_group_summaries_preserve_first_seen_order(self):
        result = interval_diagnostics(
            [0, 10, 1, 14],
            [(-1, 1), (9, 11), (0, 2), (12, 13)],
            groups=["A", "B", "A", "B"],
        )
        self.assertEqual(list(result.by_group), ["A", "B"])
        self.assertEqual(result.by_group["A"].coverage, 1.0)
        self.assertEqual(result.by_group["B"].coverage, 0.5)
        self.assertEqual(result.by_group["B"].above, 1)

    def test_unbounded_intervals_are_supported(self):
        result = interval_diagnostics([1.0, 2.0], [(-math.inf, math.inf), (1, 3)])
        self.assertEqual(result.overall.coverage, 1.0)
        self.assertTrue(math.isinf(result.overall.mean_width))
        self.assertTrue(math.isinf(result.overall.mean_interval_score))

    def test_empty_groups_and_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            interval_diagnostics([1, 2], [(0, 2)], groups=["A", "B"])
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            interval_diagnostics([1, 2], [(0, 2), (1, 3)], groups=["A"])

    def test_rejects_invalid_values_and_intervals(self):
        invalid = [
            (([1], [(2, 0)]), {}),
            (([1], [(0, 1, 2)]), {}),
            (([math.nan], [(0, 1)]), {}),
            (([1], [(math.nan, 2)]), {}),
            (([1], [(0, 2)]), {"alpha": 0}),
            (([1], [(0, 2)]), {"alpha": True}),
            (([1], [(0, 2)]), {"groups": [["unhashable"]]}),
        ]
        for args, kwargs in invalid:
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(ValueError):
                interval_diagnostics(*args, **kwargs)

    def test_accepts_one_shot_iterables(self):
        result = interval_diagnostics(
            iter([1, 2]), (pair for pair in [(0, 2), (1, 3)]),
            groups=iter(["x", "x"]),
        )
        self.assertEqual(result.overall.coverage, 1.0)
        self.assertEqual(result.by_group["x"].count, 2)


if __name__ == "__main__":
    unittest.main()
