import math
import unittest

from split_conformal import SplitConformalRegressor, empirical_coverage, mean_interval_width


class SplitConformalValidationTests(unittest.TestCase):
    def test_rejects_invalid_alpha(self):
        for alpha in (0, 1, -0.1, 1.1, True, "0.1", math.nan, math.inf):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                SplitConformalRegressor(alpha)

    def test_rejects_empty_mismatched_and_non_finite_calibration_inputs(self):
        conformal = SplitConformalRegressor()
        cases = [([], [1]), ([1], []), ([1, 2], [1]), ([], [])]
        for targets, predictions in cases:
            with self.subTest(targets=targets, predictions=predictions), self.assertRaises(ValueError):
                conformal.fit(targets, predictions)
        for bad in (True, "1", math.nan, math.inf):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                conformal.fit([bad], [0])
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                conformal.fit([0], [bad])

    def test_rejects_non_iterable_calibration_inputs(self):
        with self.assertRaises(ValueError):
            SplitConformalRegressor().fit(None, [1])

    def test_predict_requires_fit_and_valid_finite_predictions(self):
        conformal = SplitConformalRegressor()
        with self.assertRaises(RuntimeError):
            conformal.predict([1])
        conformal.fit([1, 2, 3], [1, 2, 3])
        for predictions in ([], [True], ["2"], [math.nan], [math.inf], None):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                conformal.predict(predictions)

    def test_coverage_rejects_mismatches_and_invalid_intervals(self):
        with self.assertRaises(ValueError):
            empirical_coverage([1, 2], [(1, 1)])
        for intervals in ([], [(2, 1)], [(1,)], [(1, 2, 3)], [(math.nan, 2)], [(True, 2)], ["12"]):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                empirical_coverage([1], intervals)
        with self.assertRaises(ValueError):
            empirical_coverage([math.inf], [(0, 1)])

    def test_width_rejects_invalid_intervals_and_returns_infinity_for_unbounded(self):
        for intervals in ([], [(2, 1)], [(math.nan, 2)], [(True, 2)], [(1,)]):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                mean_interval_width(intervals)
        self.assertEqual(mean_interval_width([(-math.inf, math.inf)]), math.inf)
        self.assertEqual(mean_interval_width([(0, math.inf)]), math.inf)


if __name__ == "__main__":
    unittest.main()
