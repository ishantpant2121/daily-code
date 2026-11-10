"""Tests for cross-validation diagnostics and prediction helpers."""

import unittest

from decision_tree import DecisionTreeClassifier
from model_selection import cross_validate


class CrossValidateTests(unittest.TestCase):
    def setUp(self):
        self.features = [[0], [1], [2], [3], [10], [11], [12], [13]]
        self.labels = [0, 0, 0, 0, 1, 1, 1, 1]

    def test_returns_test_scores_and_non_negative_timings(self):
        result = cross_validate(
            DecisionTreeClassifier(max_depth=2),
            self.features,
            self.labels,
            cv=4,
            scoring="accuracy",
            stratified=True,
        )
        self.assertEqual(result["test_score"], [1.0] * 4)
        self.assertEqual(len(result["fit_time"]), 4)
        self.assertEqual(len(result["score_time"]), 4)
        self.assertTrue(all(value >= 0 for value in result["fit_time"] + result["score_time"]))

    def test_optional_training_scores_and_estimators(self):
        original = DecisionTreeClassifier(max_depth=2)
        result = cross_validate(
            original,
            self.features,
            self.labels,
            cv=4,
            scoring="accuracy",
            stratified=True,
            return_train_score=True,
            return_estimator=True,
        )
        self.assertEqual(result["train_score"], [1.0] * 4)
        self.assertEqual(len(result["estimator"]), 4)
        self.assertTrue(all(model._fitted for model in result["estimator"]))
        self.assertFalse(original._fitted)

    def test_custom_scorer_and_default_score(self):
        model = DecisionTreeClassifier(max_depth=2)
        custom = cross_validate(
            model, self.features, self.labels, cv=4, stratified=True,
            scoring=lambda fitted, x, y: fitted.score(x, y),
        )
        default = cross_validate(model, self.features, self.labels, cv=4, stratified=True)
        self.assertEqual(custom["test_score"], default["test_score"])

    def test_rejects_invalid_options_and_scorers(self):
        model = DecisionTreeClassifier()
        for kwargs in (
            {"return_train_score": 1},
            {"return_estimator": 1},
            {"cv": True},
            {"scoring": "unknown"},
            {"random_state": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cross_validate(model, self.features, self.labels, cv=4, **kwargs)
        with self.assertRaises(ValueError):
            cross_validate(model, self.features, self.labels, cv=4,
                           scoring=lambda fitted, x, y: float("nan"))


class CrossValPredictTests(unittest.TestCase):
    def test_predictions_are_restored_to_original_sample_order(self):
        from model_selection import cross_val_predict

        features = [[-3], [-2], [-1], [1], [2], [3]]
        labels = [0, 0, 0, 1, 1, 1]
        predictions = cross_val_predict(
            DecisionTreeClassifier(max_depth=2),
            features,
            labels,
            cv=3,
            stratified=True,
            shuffle=True,
            random_state=19,
        )
        self.assertEqual(predictions, labels)

    def test_supports_probability_predictions(self):
        from model_selection import cross_val_predict
        from logistic_regression import LogisticRegression

        features = [[-3], [-2], [-1], [1], [2], [3]]
        labels = [0, 0, 0, 1, 1, 1]
        probabilities = cross_val_predict(
            LogisticRegression(max_iter=1000),
            features,
            labels,
            cv=3,
            method="predict_proba",
            stratified=True,
        )
        self.assertEqual(len(probabilities), len(labels))
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-9 for row in probabilities))

    def test_rejects_invalid_method_and_mismatched_data(self):
        from model_selection import cross_val_predict

        model = DecisionTreeClassifier()
        with self.assertRaises(ValueError):
            cross_val_predict(model, [[0], [1]], [0, 1], cv=2, method="missing")
        with self.assertRaises(ValueError):
            cross_val_predict(model, [[0], [1]], [0], cv=2)
        with self.assertRaises(ValueError):
            cross_val_predict(model, [[0], [1]], [0, 1], cv=2, method="_fitted")



if __name__ == "__main__":
    unittest.main()
