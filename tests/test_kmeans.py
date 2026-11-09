"""Tests for dependency-free K-means clustering."""

import unittest

from kmeans import KMeans


class KMeansTests(unittest.TestCase):
    def test_finds_two_separated_clusters(self):
        values = [[0, 0], [0, 1], [10, 10], [10, 11]]
        model = KMeans(2, n_init=4, random_state=42).fit(values)
        centers = sorted(model.cluster_centers_, key=lambda row: row[0])
        self.assertEqual(centers, [[0.0, 0.5], [10.0, 10.5]])
        self.assertAlmostEqual(model.inertia_, 1.0)
        self.assertEqual(model.predict([[0, 0.2], [10, 10.2]]), [0, 1])

    def test_is_reproducible_and_fit_predict_returns_labels(self):
        values = [[0], [1], [9], [10]]
        first = KMeans(2, random_state=7).fit_predict(values)
        second = KMeans(2, random_state=7).fit_predict(values)
        self.assertEqual(first, second)

    def test_handles_duplicate_points_and_multiple_initializations(self):
        model = KMeans(3, n_init=2, random_state=1).fit([[2], [2], [2], [2]])
        self.assertEqual(model.inertia_, 0.0)
        self.assertEqual(model.predict([[2]]), [0])
        self.assertEqual(len(model.transform([[2]])[0]), 3)

    def test_rejects_invalid_configuration_and_data(self):
        for kwargs in (
            {"n_clusters": 0},
            {"n_clusters": True},
            {"n_init": 0},
            {"max_iter": 0},
            {"tol": -1},
            {"tol": float("nan")},
            {"random_state": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                KMeans(**kwargs)
        with self.assertRaises(RuntimeError):
            KMeans(1).predict([[1]])
        for values in (
            [],
            [[1], [1, 2]],
            [[float("inf")], [2]],
            [[True], [2]],
            [[10**400], [2]],
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                KMeans(1).fit(values)
        with self.assertRaises(ValueError):
            KMeans(3).fit([[0], [1]])

    def test_rejects_prediction_feature_width_mismatch(self):
        model = KMeans(1, random_state=1).fit([[0, 1], [1, 2]])
        with self.assertRaises(ValueError):
            model.predict([[1]])


if __name__ == "__main__":
    unittest.main()
