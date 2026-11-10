"""Tests for hard and soft voting ensembles."""

import unittest

from decision_tree import DecisionTreeClassifier
from ensemble import VotingClassifier
from gaussian_nb import GaussianNB
from logistic_regression import LogisticRegression


class VotingClassifierTests(unittest.TestCase):
    def setUp(self):
        self.features = [[-3], [-2], [-1], [1], [2], [3]]
        self.labels = ["low", "low", "low", "high", "high", "high"]

    def test_hard_voting_combines_independent_estimator_copies(self):
        first = DecisionTreeClassifier(max_depth=2)
        second = GaussianNB()
        ensemble = VotingClassifier([
            ("tree", first),
            ("bayes", second),
        ]).fit(self.features, self.labels)
        self.assertEqual(ensemble.predict(self.features), self.labels)
        self.assertEqual(ensemble.score(self.features, self.labels), 1.0)
        self.assertFalse(first._fitted)
        self.assertFalse(second._fitted)
        self.assertIs(ensemble.named_estimators_["tree"], ensemble.estimators_[0])

    def test_soft_voting_aligns_class_probability_columns(self):
        ensemble = VotingClassifier(
            [("tree", DecisionTreeClassifier()), ("logistic", LogisticRegression(max_iter=1000))],
            voting="soft",
            weights=[2, 1],
        ).fit(self.features, self.labels)
        probabilities = ensemble.predict_proba([[-2], [2]])
        self.assertEqual(ensemble.predict([[-2], [2]]), ["low", "high"])
        self.assertEqual(len(probabilities), 2)
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-9 for row in probabilities))

    def test_zero_weight_estimator_does_not_affect_vote(self):
        ensemble = VotingClassifier(
            [("tree", DecisionTreeClassifier()), ("bayes", GaussianNB())],
            weights=[1, 0],
        ).fit(self.features, self.labels)
        self.assertEqual(ensemble.predict(self.features), self.labels)

    def test_rejects_invalid_configuration_and_unfitted_prediction(self):
        for kwargs in (
            {"voting": "unknown"},
            {"weights": [1]},
            {"weights": [0, 0]},
            {"weights": [1, float("nan")]},
            {"weights": [True, 1]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VotingClassifier(
                    [("tree", DecisionTreeClassifier()), ("bayes", GaussianNB())],
                    **kwargs,
                )
        with self.assertRaises(ValueError):
            VotingClassifier([("same", DecisionTreeClassifier()), ("same", GaussianNB())])
        model = VotingClassifier([("tree", DecisionTreeClassifier()), ("bayes", GaussianNB())])
        with self.assertRaises(RuntimeError):
            model.predict([[0]])
        with self.assertRaises(ValueError):
            VotingClassifier([("tree", DecisionTreeClassifier()), ("bayes", GaussianNB())],
                             voting="soft").fit(self.features, self.labels)


if __name__ == "__main__":
    unittest.main()
