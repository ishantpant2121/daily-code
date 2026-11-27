import math
import unittest

from proportion_intervals import wilson_score_interval


class WilsonScoreIntervalTests(unittest.TestCase):
    def test_balanced_sample_matches_reference_values(self):
        lower, upper = wilson_score_interval(5, 10)
        self.assertAlmostEqual(lower, 0.2365930905, places=9)
        self.assertAlmostEqual(upper, 0.7634069095, places=9)

    def test_zero_successes_has_nonzero_upper_bound(self):
        lower, upper = wilson_score_interval(0, 10)
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)
        self.assertLess(upper, 0.5)

    def test_all_successes_has_lower_bound_below_one(self):
        lower, upper = wilson_score_interval(10, 10)
        self.assertLess(lower, 1.0)
        self.assertEqual(upper, 1.0)

    def test_more_trials_narrow_interval(self):
        small = wilson_score_interval(5, 10)
        large = wilson_score_interval(50, 100)
        self.assertLess(large[1] - large[0], small[1] - small[0])

    def test_confidence_level_changes_width(self):
        narrow = wilson_score_interval(50, 100, confidence_level=0.90)
        wide = wilson_score_interval(50, 100, confidence_level=0.99)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_extreme_representable_confidence_level_stays_finite(self):
        level = math.nextafter(1.0, 0.0)
        lower, upper = wilson_score_interval(5, 10, confidence_level=level)
        self.assertTrue(math.isfinite(lower))
        self.assertTrue(math.isfinite(upper))
        self.assertTrue(0.0 <= lower <= upper <= 1.0)

    def test_bounds_are_ordered_and_inside_probability_space(self):
        for successes in range(11):
            lower, upper = wilson_score_interval(successes, 10)
            self.assertTrue(0.0 <= lower <= upper <= 1.0)

    def test_rejects_invalid_counts(self):
        invalid = [(True, 10), (1.0, 10), (-1, 10), (11, 10),
                   (0, 0), (1, -1), (1, 2.0), (1, True)]
        for successes, trials in invalid:
            with self.subTest(successes=successes, trials=trials):
                with self.assertRaises(ValueError):
                    wilson_score_interval(successes, trials)

    def test_rejects_invalid_confidence_levels(self):
        for level in [0, 1, -0.1, 1.1, math.nan, math.inf, True, "0.95"]:
            with self.subTest(level=level):
                with self.assertRaises(ValueError):
                    wilson_score_interval(1, 10, confidence_level=level)


if __name__ == "__main__":
    unittest.main()
