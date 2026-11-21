import math
import unittest

from split_conformal import SplitConformalRegressor, empirical_coverage, mean_interval_width


class SplitConformalRegressionTests(unittest.TestCase):
    def test_fit_predict_known_residual_quantile(self):
        conformal = SplitConformalRegressor(alpha=0.2).fit([1, 2, 3, 4, 5], [0] * 5)
        self.assertEqual(conformal.rank_, 5)
        self.assertEqual(conformal.quantile_, 5.0)
        self.assertEqual(conformal.predict([10, -2]), [(5.0, 15.0), (-7.0, 3.0)])

    def test_finite_sample_rank_and_small_sample_unbounded_interval(self):
        fitted = SplitConformalRegressor(alpha=0.1).fit([1, 2, 3, 4], [0] * 4)
        self.assertEqual(fitted.rank_, 5)
        self.assertEqual(fitted.quantile_, math.inf)
        self.assertEqual(fitted.predict([2]), [(-math.inf, math.inf)])

    def test_rank_one_quantile_with_large_alpha(self):
        fitted = SplitConformalRegressor(alpha=0.75).fit([2, 4, 8], [0, 0, 0])
        self.assertEqual(fitted.rank_, 1)
        self.assertEqual(fitted.quantile_, 2.0)

    def test_fit_accepts_generators(self):
        fitted = SplitConformalRegressor(0.5).fit(
            (x for x in [2, 4, 6]), (x for x in [1, 1, 1])
        )
        self.assertEqual(fitted.calibration_scores_, [1.0, 3.0, 5.0])
        self.assertEqual(fitted.predict((x for x in [10, 11])), [(7.0, 13.0), (8.0, 14.0)])

    def test_coverage_and_width_helpers(self):
        intervals = [(0, 1), (2, 4), (5, 7), (10, 12)]
        self.assertEqual(empirical_coverage([1, 3, 8, 11], intervals), 0.75)
        self.assertEqual(mean_interval_width(intervals), 1.75)

    def test_zero_width_intervals(self):
        intervals = [(2, 2), (3, 3)]
        self.assertEqual(mean_interval_width(intervals), 0.0)
        self.assertEqual(empirical_coverage([2, 4], intervals), 0.5)

    def test_refit_failure_clears_previous_calibration(self):
        conformal = SplitConformalRegressor(0.5).fit([1, 2, 3], [1, 2, 3])
        with self.assertRaises(ValueError):
            conformal.fit([], [])
        with self.assertRaises(RuntimeError):
            conformal.predict([1])
        self.assertFalse(hasattr(conformal, "quantile_"))


if __name__ == "__main__":
    unittest.main()
