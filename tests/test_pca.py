"""Tests for dependency-free principal component analysis."""

import unittest

from pca import PCA


class PCATests(unittest.TestCase):
    def test_finds_principal_direction_and_explained_variance(self):
        values = [[1, 2], [2, 4], [3, 6], [4, 8]]
        model = PCA(n_components=1).fit(values)
        self.assertAlmostEqual(model.explained_variance_ratio_[0], 1.0, places=10)
        component = model.components_[0]
        self.assertAlmostEqual(abs(component[0]), 1 / (5 ** 0.5), places=10)
        self.assertAlmostEqual(abs(component[1]), 2 / (5 ** 0.5), places=10)
        transformed = model.transform(values)
        self.assertEqual(len(transformed), len(values))
        self.assertTrue(all(len(row) == 1 for row in transformed))

    def test_full_components_reconstruct_centered_input(self):
        values = [[2, 0], [0, 1], [-2, 0], [0, -1]]
        model = PCA(n_components=2).fit(values)
        restored = model.inverse_transform(model.transform(values))
        for actual, expected in zip(restored, values):
            for left, right in zip(actual, expected):
                self.assertAlmostEqual(left, right, places=10)
        self.assertAlmostEqual(sum(model.explained_variance_ratio_), 1.0)

    def test_constant_data_has_zero_explained_variance(self):
        model = PCA().fit([[3, 3], [3, 3], [3, 3]])
        self.assertEqual(model.explained_variance_, [0.0, 0.0])
        self.assertEqual(model.explained_variance_ratio_, [0.0, 0.0])

    def test_requires_fit_and_valid_component_shapes(self):
        with self.assertRaises(RuntimeError):
            PCA().transform([[1, 2]])
        with self.assertRaises(ValueError):
            PCA().fit([[1, 2]])
        with self.assertRaises(ValueError):
            PCA(3).fit([[1, 2], [3, 4]])
        model = PCA(1).fit([[1, 2], [3, 4], [5, 6]])
        with self.assertRaises(ValueError):
            model.transform([[1]])
        with self.assertRaises(ValueError):
            model.inverse_transform([[1, 2]])

    def test_rejects_invalid_options_and_non_finite_data(self):
        for kwargs in (
            {"n_components": 0},
            {"n_components": True},
            {"tol": -1},
            {"tol": float("nan")},
            {"max_iter": 0},
            {"max_iter": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PCA(**kwargs)
        with self.assertRaises(ValueError):
            PCA().fit([[1, float("inf")], [2, 3]])


if __name__ == "__main__":
    unittest.main()
