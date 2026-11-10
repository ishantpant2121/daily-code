"""Tests for hard and soft voting ensembles."""

import unittest

from decision_tree import DecisionTreeClassifier
from ensemble import VotingClassifier
from gaussian_nb import GaussianNB
from logistic_regression import LogisticRegression


class _FixedProbabilityClassifier:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def fit(self, features, labels):
        self.classes_ = tuple(dict.fromkeys(labels))
        return self

    def predict(self, features):
        return [self.classes_[0] for _ in features]

    def predict_proba(self, features):
        return [list(self.probabilities) for _ in features]


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

    def test_rejects_weight_sum_overflow(self):
        with self.assertRaisesRegex(ValueError, "finite positive sum"):
            VotingClassifier(
                [("tree", DecisionTreeClassifier()), ("bayes", GaussianNB())],
                weights=[1e308, 1e308],
            )

    def test_rejects_probability_values_that_overflow_float_conversion(self):
        ensemble = VotingClassifier(
            [
                ("large", _FixedProbabilityClassifier([10**400, 0])),
                ("valid", _FixedProbabilityClassifier([1, 0])),
            ],
            voting="soft",
        ).fit(self.features, self.labels)
        with self.assertRaisesRegex(ValueError, "probabilities must be finite"):
            ensemble.predict_proba([[0]])

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
        class HardOnlyClassifier:
            def fit(self, features, labels):
                return self

            def predict(self, features):
                return [self.labels_[0]] * len(features)

            def __init__(self):
                self.labels_ = []

        with self.assertRaises(ValueError):
            VotingClassifier(
                [("tree", DecisionTreeClassifier()), ("hard", HardOnlyClassifier())],
                voting="soft",
            ).fit(self.features, self.labels)


if __name__ == "__main__":
    unittest.main()
