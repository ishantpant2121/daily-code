"""Tests for dependency-free Gaussian Naive Bayes classification."""

import unittest

from gaussian_nb import GaussianNB


class GaussianNBTests(unittest.TestCase):
    def test_classifies_separated_numeric_clusters(self):
        model = GaussianNB().fit(
            [[-2, 0], [-1, 0], [1, 0], [2, 0]],
            ["left", "left", "right", "right"],
        )
        self.assertEqual(model.predict([[-1.5, 0], [1.5, 0]]), ["left", "right"])
        self.assertEqual(model.classes_, ("left", "right"))

    def test_probabilities_are_finite_and_normalized(self):
        model = GaussianNB().fit(
            [[0, 1], [1, 1], [4, 1], [5, 1]],
            [0, 0, 1, 1],
        )
        probabilities = model.predict_proba([[0.5, 1], [4.5, 1]])
        for row in probabilities:
            self.assertAlmostEqual(sum(row), 1.0)
            self.assertTrue(all(0 <= value <= 1 for value in row))
        self.assertEqual(model.predict([[0.5, 1], [4.5, 1]]), [0, 1])

    def test_handles_all_constant_features_with_variance_smoothing(self):
        model = GaussianNB().fit([[2, 2], [2, 2], [2, 2], [2, 2]], ["a", "a", "b", "b"])
        probabilities = model.predict_proba([[2, 2]])
        self.assertAlmostEqual(sum(probabilities[0]), 1.0)

    def test_accepts_one_shot_iterables(self):
        model = GaussianNB().fit(
            ([value] for value in [0, 1, 4, 5]),
            (label for label in ["low", "low", "high", "high"]),
        )
        self.assertEqual(model.predict([[4.5]]), ["high"])

    def test_rejects_invalid_smoothing_data_and_unfitted_use(self):
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GaussianNB(var_smoothing=value)
        with self.assertRaises(RuntimeError):
            GaussianNB().predict([[1]])
        for features, labels in (
            ([[1], [2]], [0]),
            ([[1], [2]], [0, 1, 2]),
            ([[1], [2]], [0]),
            ([[True], [2]], [0, 1]),
            ([[10**400], [2]], [0, 1]),
            ([[1], [2, 3]], [0, 1]),
        ):
            with self.subTest(features=features, labels=labels), self.assertRaises(ValueError):
                GaussianNB().fit(features, labels)


if __name__ == "__main__":
    unittest.main()
