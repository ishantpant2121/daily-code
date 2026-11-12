import unittest

from feature_importance import permutation_importance


class ThresholdClassifier:
    def __init__(self):
        self.fitted = False

    def fit(self, features, labels):
        self.fitted = True
        return self

    def predict(self, features):
        if not self.fitted:
            raise RuntimeError("not fitted")
        return [1 if row[0] >= 0.5 else 0 for row in features]

    def score(self, features, labels):
        predictions = self.predict(features)
        return sum(left == right for left, right in zip(predictions, labels)) / len(labels)


class PermutationImportanceTests(unittest.TestCase):
    def setUp(self):
        self.features = [[0, 10], [0, 11], [0, 12], [1, 13], [1, 14], [1, 15]]
        self.labels = [0, 0, 0, 1, 1, 1]
        self.estimator = ThresholdClassifier().fit(self.features, self.labels)

    def test_predictive_feature_has_larger_importance_than_noise(self):
        result = permutation_importance(
            self.estimator,
            self.features,
            self.labels,
            n_repeats=20,
            random_state=7,
        )
        self.assertEqual(result.baseline_score, 1.0)
        self.assertGreater(result.importances_mean[0], result.importances_mean[1])
        self.assertEqual(len(result.importances[0]), 20)
        self.assertTrue(all(value >= 0 for value in result.importances_mean))

    def test_is_reproducible_and_does_not_mutate_inputs_or_estimator(self):
        original_features = [row[:] for row in self.features]
        original_labels = self.labels[:]
        first = permutation_importance(
            self.estimator, self.features, self.labels, n_repeats=4, random_state=3
        )
        second = permutation_importance(
            self.estimator, self.features, self.labels, n_repeats=4, random_state=3
        )
        self.assertEqual(first, second)
        self.assertEqual(self.features, original_features)
        self.assertEqual(self.labels, original_labels)
        self.assertTrue(self.estimator.fitted)

    def test_aggregates_large_finite_importances_without_overflow(self):
        calls = 0

        def scorer(model, features, labels):
            nonlocal calls
            calls += 1
            return 1e308 if calls == 1 else 0.0

        result = permutation_importance(
            object(), [[0], [1], [2]], [0, 1, 1],
            scoring=scorer, n_repeats=2, random_state=1,
        )
        self.assertEqual(result.baseline_score, 1e308)
        self.assertEqual(result.importances_mean, [1e308])
        self.assertEqual(result.importances_std, [0.0])

    def test_rejects_unrepresentable_importance_difference(self):
        scores = iter([1e308, -1e308])
        with self.assertRaisesRegex(ValueError, "importance must be a finite"):
            permutation_importance(
                object(), [[0], [1]], [0, 1],
                scoring=lambda model, features, labels: next(scores),
                n_repeats=1,
            )

    def test_supports_custom_scorer_and_rejects_bad_options(self):
        result = permutation_importance(
            self.estimator,
            self.features,
            self.labels,
            scoring=lambda model, x, y: model.score(x, y),
            n_repeats=2,
            random_state=1,
        )
        self.assertEqual(result.baseline_score, 1.0)
        for kwargs in ({"n_repeats": 0}, {"n_repeats": True}, {"random_state": True}, {"scoring": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                permutation_importance(
                    self.estimator, self.features, self.labels, **kwargs
                )
        with self.assertRaises(ValueError):
            permutation_importance(self.estimator, self.features, [0, 1])


if __name__ == "__main__":
    unittest.main()
