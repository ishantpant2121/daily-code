import math
import unittest

from split_conformal import SplitConformalRegressor, mean_interval_width


class SplitConformalNumericsTests(unittest.TestCase):
    def test_extreme_finite_residuals_can_produce_conservative_infinity(self):
        fitted = SplitConformalRegressor(0.5).fit([1e308], [-1e308])
        self.assertEqual(fitted.quantile_, math.inf)
        self.assertEqual(fitted.predict([1e308]), [(-math.inf, math.inf)])

    def test_mean_width_uses_scaled_aggregation(self):
        self.assertEqual(mean_interval_width([(0, 1e300), (0, 1e300)]), 1e300)
        self.assertEqual(mean_interval_width([(0, 1e308), (0, 1e308)]), 1e308)

    def test_finite_endpoints_with_overflowing_width_report_infinity(self):
        self.assertEqual(mean_interval_width([(-1e308, 1e308)]), math.inf)

    def test_fit_uses_exact_finite_sample_correction(self):
        # n=9, alpha=0.1 -> ceil((9+1)*0.9)=9, the largest residual.
        fitted = SplitConformalRegressor(0.1).fit(range(1, 10), [0] * 9)
        self.assertEqual(fitted.rank_, 9)
        self.assertEqual(fitted.quantile_, 9.0)


if __name__ == "__main__":
    unittest.main()
