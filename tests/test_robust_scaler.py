"""Tests for robust feature scaling."""

import unittest
from preprocessing import RobustScaler


class RobustScalerTests(unittest.TestCase):
    def test_uses_median_and_interquartile_range(self):
        values = [[1, 10], [2, 10], [3, 10], [100, 10]]
        scaler = RobustScaler().fit(values)
        self.assertEqual(scaler.center_, [2.5, 10.0])
        self.assertEqual(scaler.scale_, [25.5, 1.0])
        transformed = scaler.transform(values)
        self.assertAlmostEqual(transformed[-1][0], (100 - 2.5) / 25.5)
        restored = scaler.inverse_transform(transformed)
        for actual, expected in zip(restored, values):
            for left, right in zip(actual, expected):
                self.assertAlmostEqual(left, right)

    def test_supports_custom_quantile_ranges_and_constant_columns(self):
        scaler = RobustScaler((10, 90)).fit([[5, 1], [5, 2], [5, 3]])
        self.assertEqual(scaler.center_, [5.0, 2.0])
        self.assertEqual(scaler.transform([[5, 2]]), [[0.0, 0.0]])

    def test_rejects_invalid_ranges_and_requires_fit(self):
        for value in ((0, 0), (-1, 75), (25, 101), [25, 75], (True, 75)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RobustScaler(value)
        with self.assertRaises(RuntimeError):
            RobustScaler().transform([[1]])


if __name__ == "__main__":
    unittest.main()
