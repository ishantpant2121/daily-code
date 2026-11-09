"""Tests for dependency-free k-nearest-neighbors classification."""

import unittest
from knn_classifier import KNNClassifier


class KNNClassifierTests(unittest.TestCase):
    def test_predicts_nearest_class_and_returns_normalized_probabilities(self):
        model = KNNClassifier(3).fit(
            [[0, 0], [0, 1], [5, 5], [6, 5]],
            ["near", "near", "far", "far"],
        )
        self.assertEqual(model.predict([[0.2, 0.2], [5.5, 5]]), ["near", "far"])
        probabilities = model.predict_proba([[0.2, 0.2], [5.5, 5]])
        self.assertEqual(model.classes_, ("near", "far"))
        for row in probabilities:
            self.assertAlmostEqual(sum(row), 1.0)

    def test_distance_weights_prioritize_exact_matches(self):
        model = KNNClassifier(3, weights="distance").fit(
            [[0], [0], [2]], ["exact", "exact", "other"]
        )
        self.assertEqual(model.predict([[0]]), ["exact"])
        self.assertEqual(model.predict_proba([[0]]), [[1.0, 0.0]])

    def test_supports_manhattan_distance_and_stable_ties(self):
        model = KNNClassifier(2, metric="manhattan").fit(
            [[0, 0], [2, 0]], ["first", "second"]
        )
        self.assertEqual(model.predict([[1, 0]]), ["first"])

    def test_rejects_invalid_configuration_data_and_unfitted_prediction(self):
        for kwargs in (
            {"n_neighbors": 0},
            {"n_neighbors": True},
            {"weights": "bad"},
            {"metric": "cosine"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                KNNClassifier(**kwargs)
        with self.assertRaises(RuntimeError):
            KNNClassifier().predict([[1]])
        with self.assertRaises(ValueError):
            KNNClassifier(3).fit([[0], [1]], ["a", "b"])
        with self.assertRaises(ValueError):
            KNNClassifier().fit([[0], [1]], ["a"])
        with self.assertRaises(ValueError):
            KNNClassifier().fit([[True]], ["a"])
        with self.assertRaises(ValueError):
            KNNClassifier().fit([[10**400]], ["a"])


if __name__ == "__main__":
    unittest.main()
