"""Tests for paired bootstrap and permutation model-comparison utilities."""

import unittest

from paired_model_comparison import (
    paired_bootstrap_score_difference,
    paired_permutation_test,
)


class PairedModelComparisonTests(unittest.TestCase):
    def setUp(self):
        self.y = [0, 1, 0, 1, 0, 1]
        self.predictions_a = [0, 1, 0, 1, 0, 1]
        self.predictions_b = [0, 0, 0, 1, 0, 1]

    def test_bootstrap_is_reproducible_and_paired(self):
        first = paired_bootstrap_score_difference(
            self.y, self.predictions_a, self.predictions_b,
            n_bootstrap=200, random_state=9,
        )
        second = paired_bootstrap_score_difference(
            self.y, self.predictions_a, self.predictions_b,
            n_bootstrap=200, random_state=9,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.observed_difference, 1 / 6)
        self.assertEqual(len(first.bootstrap_differences), 200)
        self.assertLessEqual(first.lower_bound, first.upper_bound)
        self.assertEqual(first.scoring, "accuracy")

    def test_identical_predictions_have_zero_difference(self):
        result = paired_bootstrap_score_difference(
            self.y, self.predictions_a, self.predictions_a,
            n_bootstrap=50, random_state=2,
        )
        self.assertEqual(result.observed_difference, 0.0)
        self.assertEqual(result.mean_difference, 0.0)
        self.assertEqual((result.lower_bound, result.upper_bound), (0.0, 0.0))

    def test_permutation_is_reproducible_and_p_value_is_bounded(self):
        first = paired_permutation_test(
            self.y, self.predictions_a, self.predictions_b,
            n_permutations=250, random_state=3,
        )
        second = paired_permutation_test(
            self.y, self.predictions_a, self.predictions_b,
            n_permutations=250, random_state=3,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.observed_difference, 1 / 6)
        self.assertEqual(len(first.permutation_differences), 250)
        self.assertGreater(first.p_value, 0.0)
        self.assertLessEqual(first.p_value, 1.0)

    def test_permutation_of_identical_predictions_has_p_value_one(self):
        result = paired_permutation_test(
            self.y, self.predictions_a, self.predictions_a,
            n_permutations=100, random_state=11,
        )
        self.assertEqual(result.observed_difference, 0.0)
        self.assertEqual(result.p_value, 1.0)

    def test_custom_scorer_accepts_one_shot_inputs(self):
        scorer = lambda actual, pred: sum(
            abs(a - p) for a, p in zip(actual, pred)
        ) / len(actual)
        result = paired_bootstrap_score_difference(
            (x for x in [1, 2, 3]),
            (x for x in [1, 2, 3]),
            (x for x in [0, 2, 4]),
            scoring=scorer, n_bootstrap=20, random_state=1,
        )
        self.assertEqual(result.scoring, "<lambda>")
        self.assertAlmostEqual(result.observed_difference, -2 / 3)

    def test_supported_builtin_scorers(self):
        for scorer in (
            "balanced_accuracy",
            "r2",
            "neg_mean_squared_error",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ):
            with self.subTest(scoring=scorer):
                result = paired_bootstrap_score_difference(
                    [1, 2, 3], [1, 2, 3], [0, 2, 4],
                    scoring=scorer, n_bootstrap=30, random_state=4,
                )
                self.assertEqual(result.scoring, scorer)


if __name__ == "__main__":
    unittest.main()
