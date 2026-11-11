import unittest

from model_diagnostics import bootstrap_score, learning_curve, validation_curve


class ThresholdClassifier:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, features, labels):
        return self

    def predict(self, features):
        return [1 if row[0] >= self.threshold else 0 for row in features]

    def score(self, features, labels):
        predictions = self.predict(features)
        return sum(left == right for left, right in zip(predictions, labels)) / len(labels)


class LearningCurveTests(unittest.TestCase):
    def setUp(self):
        self.features = [[0], [0.1], [0.2], [0.8], [0.9], [1.0], [0.05], [0.95]]
        self.labels = [0, 0, 0, 1, 1, 1, 0, 1]

    def test_returns_train_and_validation_scores_for_each_size(self):
        result = learning_curve(
            ThresholdClassifier(),
            self.features,
            self.labels,
            train_sizes=[0.5, 0.75, 1.0],
            cv=4,
            scoring="accuracy",
            stratified=True,
            shuffle=True,
            random_state=9,
        )
        self.assertEqual(result["train_sizes"], [3, 5, 6])
        self.assertEqual(len(result["train_scores"]), 3)
        self.assertTrue(all(len(scores) == 4 for scores in result["test_scores"]))
        self.assertTrue(all(0.0 <= score <= 1.0 for scores in result["train_scores"] for score in scores))

    def test_rejects_invalid_sizes_and_mismatched_data(self):
        for sizes in ([], [0], [1.1], [True], [0.5, 0.4], [1, 100]):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                learning_curve(
                    ThresholdClassifier(), self.features, self.labels,
                    train_sizes=sizes, cv=4,
                )
        with self.assertRaises(ValueError):
            learning_curve(ThresholdClassifier(), self.features, self.labels[:-1], cv=4)


class ValidationCurveTests(unittest.TestCase):
    def setUp(self):
        self.features = [[0], [0.1], [0.2], [0.8], [0.9], [1.0], [0.05], [0.95]]
        self.labels = [0, 0, 0, 1, 1, 1, 0, 1]

    def test_evaluates_parameter_values_with_identical_folds(self):
        model = ThresholdClassifier()
        result = validation_curve(
            model,
            self.features,
            self.labels,
            "threshold",
            [0.3, 0.5, 0.7],
            cv=4,
            scoring="accuracy",
            stratified=True,
        )
        self.assertEqual(result["param_range"], [0.3, 0.5, 0.7])
        self.assertEqual(len(result["train_scores"]), 3)
        self.assertTrue(all(len(scores) == 4 for scores in result["test_scores"]))
        self.assertEqual(model.threshold, 0.5)

    def test_supports_nested_pipeline_parameters_and_one_shot_inputs(self):
        class Wrapper:
            def __init__(self):
                self.named_steps = {"classifier": ThresholdClassifier()}
            def fit(self, features, labels):
                self.named_steps["classifier"].fit(features, labels)
                return self
            def predict(self, features):
                return self.named_steps["classifier"].predict(features)
            def score(self, features, labels):
                return self.named_steps["classifier"].score(features, labels)

        model = Wrapper()
        result = validation_curve(
            model,
            (row for row in self.features),
            (label for label in self.labels),
            "classifier__threshold",
            [0.4, 0.6],
            cv=4,
            scoring="accuracy",
            stratified=True,
        )
        self.assertEqual(result["param_range"], [0.4, 0.6])

    def test_rejects_unknown_parameter_and_empty_range(self):
        with self.assertRaises(ValueError):
            validation_curve(ThresholdClassifier(), self.features, self.labels, "unknown", [1], cv=4)
        with self.assertRaises(ValueError):
            validation_curve(ThresholdClassifier(), self.features, self.labels, "threshold", [], cv=4)


class BootstrapScoreTests(unittest.TestCase):
    def test_returns_reproducible_distribution_and_ordered_interval(self):
        features = [[0], [0.1], [0.2], [0.8], [0.9], [1.0]]
        labels = [0, 0, 0, 1, 1, 1]
        model = ThresholdClassifier().fit(features, labels)
        first = bootstrap_score(
            model, features, labels, n_bootstrap=100, random_state=12
        )
        second = bootstrap_score(
            model, features, labels, n_bootstrap=100, random_state=12
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first.scores), 100)
        self.assertEqual(first.mean_score, 1.0)
        self.assertEqual((first.lower_bound, first.upper_bound), (1.0, 1.0))

    def test_supports_custom_scorer_and_validates_options(self):
        features = [[0], [0.1], [0.8], [1.0]]
        labels = [0, 0, 1, 1]
        model = ThresholdClassifier().fit(features, labels)
        result = bootstrap_score(
            model, features, labels, n_bootstrap=5,
            confidence_level=0.8,
            scoring=lambda fitted, x, y: fitted.score(x, y),
        )
        self.assertEqual(result.confidence_level, 0.8)
        for kwargs in (
            {"n_bootstrap": 0},
            {"n_bootstrap": True},
            {"confidence_level": 1.0},
            {"confidence_level": float("nan")},
            {"random_state": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bootstrap_score(model, features, labels, **kwargs)
        with self.assertRaises(ValueError):
            bootstrap_score(model, features, labels[:-1])


if __name__ == "__main__":
    unittest.main()
