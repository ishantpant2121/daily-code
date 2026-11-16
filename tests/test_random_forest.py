"""Tests for bootstrap and random-feature tree ensembles."""

import unittest

from random_forest import RandomForestClassifier


class RandomForestClassifierTests(unittest.TestCase):
    def setUp(self):
        self.features = [
            [0.0, 0.0], [0.1, 0.2], [-0.1, 0.1], [0.2, -0.1],
            [8.0, 8.0], [8.1, 8.2], [7.9, 8.1], [8.2, 7.9],
        ]
        self.labels = ["low"] * 4 + ["high"] * 4

    def test_fit_predict_and_probabilities_are_reproducible(self):
        kwargs = {"n_estimators": 17, "max_depth": 3, "random_state": 19}
        first = RandomForestClassifier(**kwargs).fit(self.features, self.labels)
        second = RandomForestClassifier(**kwargs).fit(self.features, self.labels)
        self.assertEqual(first.predict(self.features), second.predict(self.features))
        self.assertEqual(first.predict_proba(self.features), second.predict_proba(self.features))
        self.assertEqual(first.classes_, ("low", "high"))
        self.assertEqual(len(first.estimators_), 17)
        self.assertGreaterEqual(first.score(self.features, self.labels), 0.75)
        for row in first.predict_proba(self.features):
            self.assertAlmostEqual(sum(row), 1.0)

    def test_probability_columns_include_classes_missing_from_bootstrap_trees(self):
        features = [[0.0], [1.0], [2.0], [3.0]]
        labels = ["a", "b", "c", "d"]
        forest = RandomForestClassifier(
            n_estimators=40,
            max_samples=2,
            max_features=1,
            random_state=7,
        ).fit(features, labels)
        probabilities = forest.predict_proba([[0.5], [2.5]])
        self.assertEqual(forest.classes_, ("a", "b", "c", "d"))
        self.assertEqual([len(row) for row in probabilities], [4, 4])
        for row in probabilities:
            self.assertAlmostEqual(sum(row), 1.0)

    def test_fractional_max_samples_and_bootstrap_free_mode(self):
        fractional = RandomForestClassifier(
            n_estimators=3, max_samples=0.5, random_state=2
        ).fit(self.features, self.labels)
        self.assertEqual(fractional.max_samples_, 4)

        full = RandomForestClassifier(
            n_estimators=3, bootstrap=False, random_state=2
        ).fit(self.features, self.labels)
        self.assertEqual(full.max_samples_, len(self.features))
        self.assertEqual(full.n_samples_fit_, len(self.features))

    def test_feature_importances_are_finite_and_normalized(self):
        features = [[0, 5], [1, 5], [2, 5], [3, 5], [8, 5], [9, 5]]
        labels = [0, 0, 0, 1, 1, 1]
        forest = RandomForestClassifier(
            n_estimators=9, max_features=None, random_state=11
        ).fit(features, labels)
        self.assertEqual(len(forest.feature_importances_), 2)
        self.assertTrue(all(value >= 0 for value in forest.feature_importances_))
        self.assertAlmostEqual(sum(forest.feature_importances_), 1.0)
        self.assertEqual(forest.feature_importances_[1], 0.0)

    def test_rejects_invalid_configuration(self):
        for kwargs in (
            {"n_estimators": 0},
            {"n_estimators": True},
            {"n_estimators": 1.5},
            {"bootstrap": 1},
            {"max_samples": 0},
            {"max_samples": 1.1},
            {"max_samples": True},
            {"max_samples": "all"},
            {"bootstrap": False, "max_samples": 0.5},
            {"random_state": True},
            {"max_features": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RandomForestClassifier(**kwargs)

    def test_rejects_invalid_data_and_prediction_before_fit(self):
        forest = RandomForestClassifier(n_estimators=3, random_state=1)
        with self.assertRaises(RuntimeError):
            forest.predict([[0]])
        with self.assertRaises(ValueError):
            forest.fit([[0], [1]], [0])
        with self.assertRaises(ValueError):
            RandomForestClassifier(n_estimators=2).fit([[0], [1]], [[0], [1]])

        forest.fit(self.features, self.labels)
        with self.assertRaises(ValueError):
            forest.predict([[1, 2, 3]])
        with self.assertRaises(ValueError):
            forest.score([[0]], [0, 1])


if __name__ == "__main__":
    unittest.main()
