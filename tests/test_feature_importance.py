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


class ThresholdRegressor:
    def __init__(self):
        self.fitted = False

    def fit(self, features, labels):
        self.fitted = True
        return self

    def predict(self, features):
        if not self.fitted:
            raise RuntimeError("not fitted")
        return [float(row[0] >= 0.5) for row in features]


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
        self.assertEqual(result.n_samples_used, len(self.features))

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
        invalid_options = (
            {"n_repeats": 0},
            {"n_repeats": True},
            {"random_state": True},
            {"scoring": "unknown"},
            {"max_samples": 0},
            {"max_samples": 0.0},
            {"max_samples": True},
            {"max_samples": float("nan")},
            {"max_samples": float("inf")},
            {"max_samples": 1.1},
            {"max_samples": 7},
        )
        for kwargs in invalid_options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                permutation_importance(
                    self.estimator, self.features, self.labels, **kwargs
                )
        with self.assertRaises(ValueError):
            permutation_importance(self.estimator, self.features, [0, 1])

    def test_supports_builtin_regression_scorers(self):
        estimator = ThresholdRegressor().fit(self.features, self.labels)
        for scoring in (
            "r2",
            "neg_mean_squared_error",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ):
            with self.subTest(scoring=scoring):
                result = permutation_importance(
                    estimator,
                    self.features,
                    self.labels,
                    scoring=scoring,
                    n_repeats=12,
                    random_state=7,
                )
                self.assertGreater(result.importances_mean[0], 0.0)
                self.assertEqual(result.importances_mean[1], 0.0)

    def test_max_samples_reuses_one_subset_for_baseline_and_permutations(self):
        observed_labels = []
        observed_sizes = []

        def scorer(model, features, labels):
            observed_labels.append(tuple(labels))
            observed_sizes.append(len(features))
            return sum(row[0] == label for row, label in zip(features, labels)) / len(labels)

        result = permutation_importance(
            object(),
            self.features,
            self.labels,
            scoring=scorer,
            n_repeats=3,
            random_state=9,
            max_samples=0.5,
        )
        self.assertEqual(result.n_samples_used, 3)
        self.assertEqual(observed_sizes, [3] * 7)
        self.assertTrue(all(labels == observed_labels[0] for labels in observed_labels))

    def test_max_samples_accepts_integer_count_and_rounds_fraction_up(self):
        integer_result = permutation_importance(
            self.estimator, self.features, self.labels,
            max_samples=4, n_repeats=1, random_state=2,
        )
        fractional_result = permutation_importance(
            self.estimator, self.features, self.labels,
            max_samples=0.01, n_repeats=1, random_state=2,
        )
        self.assertEqual(integer_result.n_samples_used, 4)
        self.assertEqual(fractional_result.n_samples_used, 1)

    def test_subsampled_importance_is_reproducible(self):
        first = permutation_importance(
            self.estimator, self.features, self.labels,
            max_samples=0.5, n_repeats=5, random_state=18,
        )
        second = permutation_importance(
            self.estimator, self.features, self.labels,
            max_samples=0.5, n_repeats=5, random_state=18,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
