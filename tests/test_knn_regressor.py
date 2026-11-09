"""Tests for dependency-free k-nearest-neighbors regression."""

import unittest

from knn_regressor import KNNRegressor


class KNNRegressorTests(unittest.TestCase):
    def test_predicts_local_numeric_averages(self):
        model = KNNRegressor(2).fit([[0], [2], [4]], [0, 2, 4])
        self.assertEqual(model.predict([[1], [3]]), [1.0, 3.0])

    def test_distance_weights_and_exact_matches(self):
        model = KNNRegressor(3, weights="distance").fit(
            [[0], [0], [2]],
            [2, 4, 100],
        )
        self.assertEqual(model.predict([[0]]), [3.0])

    def test_supports_manhattan_distance_and_generators(self):
        model = KNNRegressor(1, metric="manhattan").fit(
            ([value, 0] for value in [0, 5]),
            (value for value in [10, 20]),
        )
        self.assertEqual(model.predict([[4, 0]]), [20.0])

    def test_score_uses_r_squared(self):
        model = KNNRegressor(1).fit([[0], [1], [2]], [1, 3, 5])
        self.assertEqual(model.score([[0], [1], [2]], [1, 3, 5]), 1.0)

    def test_rejects_invalid_configuration_data_and_unfitted_use(self):
        for kwargs in (
            {"n_neighbors": 0},
            {"n_neighbors": True},
            {"weights": "bad"},
            {"metric": "cosine"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                KNNRegressor(**kwargs)
        with self.assertRaises(RuntimeError):
            KNNRegressor().predict([[1]])
        for features, targets in (
            ([[0], [1]], [1]),
            ([[0], [1]], [True, 2]),
            ([[0], [1]], [10**400, 2]),
            ([[0], [1, 2]], [1, 2]),
        ):
            with self.subTest(features=features, targets=targets), self.assertRaises(ValueError):
                KNNRegressor().fit(features, targets)

    def test_rejects_prediction_width_mismatch(self):
        model = KNNRegressor(1).fit([[0, 1], [1, 2]], [1, 2])
        with self.assertRaises(ValueError):
            model.predict([[1]])


if __name__ == "__main__":
    unittest.main()
