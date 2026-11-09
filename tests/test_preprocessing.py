"""Tests for dependency-free feature scaling."""

import unittest

from preprocessing import MinMaxScaler, StandardScaler


class MinMaxScalerTests(unittest.TestCase):
    def test_scales_features_independently_and_inverts(self):
        values = [[1, 10], [2, 20], [3, 30]]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        transformed = scaler.fit_transform(values)
        self.assertEqual(transformed, [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        restored = scaler.inverse_transform(transformed)
        for actual, expected in zip(restored, values):
            for left, right in zip(actual, expected):
                self.assertAlmostEqual(left, right)

    def test_constant_columns_map_to_lower_bound(self):
        scaler = MinMaxScaler((2, 4)).fit([[5, 1], [5, 3]])
        self.assertEqual(scaler.transform([[5, 2]]), [[2.0, 3.0]])
        self.assertEqual(scaler.inverse_transform([[2, 3]]), [[5.0, 2.0]])

    def test_requires_fit_and_matching_feature_count(self):
        with self.assertRaises(RuntimeError):
            MinMaxScaler().transform([[1]])
        scaler = MinMaxScaler().fit([[1, 2]])
        with self.assertRaises(ValueError):
            scaler.transform([[1]])

    def test_rejects_invalid_ranges_and_matrices(self):
        for feature_range in ((1, 1), (2, 1), (0, float("inf")), [0, 1]):
            with self.subTest(feature_range=feature_range):
                with self.assertRaises(ValueError):
                    MinMaxScaler(feature_range)
        for matrix in ([], [[1], [1, 2]], [[float("nan")]], [[10**400]]):
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    MinMaxScaler().fit(matrix)


class StandardScalerTests(unittest.TestCase):
    def test_centers_and_scales_using_population_standard_deviation(self):
        values = [[1, 10], [2, 10], [3, 10]]
        scaler = StandardScaler()
        transformed = scaler.fit_transform(values)
        self.assertAlmostEqual(transformed[0][0], -1.22474487139)
        self.assertAlmostEqual(transformed[1][0], 0.0)
        self.assertAlmostEqual(transformed[2][0], 1.22474487139)
        self.assertEqual([row[1] for row in transformed], [0.0, 0.0, 0.0])
        restored = scaler.inverse_transform(transformed)
        for actual, expected in zip(restored, values):
            for left, right in zip(actual, expected):
                self.assertAlmostEqual(left, right)

    def test_rejects_transform_before_fit_and_invalid_width(self):
        with self.assertRaises(RuntimeError):
            StandardScaler().transform([[1]])
        scaler = StandardScaler().fit([[1, 2]])
        with self.assertRaises(ValueError):
            scaler.transform([[1]])


if __name__ == "__main__":
    unittest.main()
