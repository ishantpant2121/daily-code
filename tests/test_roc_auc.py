"""Tests for dependency-free ROC AUC scoring."""

import unittest

from classification_metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


class RocAucTests(unittest.TestCase):
    def test_scores_perfect_reversed_and_known_rankings(self):
        self.assertEqual(roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(roc_auc_score([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertAlmostEqual(roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)

    def test_assigns_half_credit_to_tied_positive_negative_scores(self):
        self.assertEqual(roc_auc_score([0, 1], [0.5, 0.5]), 0.5)

    def test_accepts_one_shot_iterables_and_custom_positive_labels(self):
        result = roc_auc_score(
            (label for label in ["no", "yes", "no", "yes"]),
            (score for score in [0.1, 0.9, 0.4, 0.8]),
            positive_label="yes",
        )
        self.assertEqual(result, 1.0)

    def test_rejects_invalid_lengths_scores_and_single_class_targets(self):
        cases = (
            lambda: roc_auc_score([], []),
            lambda: roc_auc_score([0, 1], [0.2]),
            lambda: roc_auc_score([1, 1], [0.2, 0.8]),
            lambda: roc_auc_score([0, 1], [float("nan"), 0.8]),
            lambda: roc_auc_score([0, 1], [0.2, float("inf")]),
            lambda: roc_auc_score([0, 1], [0.2, True]),
        )
        for call in cases:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


    def test_precision_recall_curve_groups_tied_scores(self):
        precision, recall, thresholds = precision_recall_curve(
            [0, 1, 1, 0], [0.1, 0.4, 0.4, 0.8]
        )
        self.assertEqual(thresholds, [0.1, 0.4, 0.8])
        self.assertEqual(recall, [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(precision, [0.5, 2 / 3, 0.0, 1.0])

    def test_precision_recall_curve_validates_inputs(self):
        with self.assertRaises(ValueError):
            precision_recall_curve([], [])
        with self.assertRaises(ValueError):
            precision_recall_curve([0, 1], [0.1])
        with self.assertRaises(ValueError):
            precision_recall_curve([1, 1], [0.2, 0.8])
        with self.assertRaises(ValueError):
            precision_recall_curve([0, 1], [0.2, float("nan")])

    def test_average_precision_uses_stepwise_recall_increments(self):
        self.assertAlmostEqual(
            average_precision_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]),
            5 / 6,
        )
        self.assertEqual(
            average_precision_score(["n", "y"], [0.1, 0.9], positive_label="y"),
            1.0,
        )


    def test_roc_curve_returns_monotonic_rates_and_descending_thresholds(self):
        false_positive, true_positive, thresholds = roc_curve(
            [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
        )
        self.assertEqual(thresholds, [float("inf"), 0.8, 0.4, 0.35, 0.1])
        self.assertEqual(false_positive, [0.0, 0.0, 0.5, 0.5, 1.0])
        self.assertEqual(true_positive, [0.0, 0.5, 0.5, 1.0, 1.0])

    def test_roc_curve_groups_ties_and_validates_single_class(self):
        false_positive, true_positive, thresholds = roc_curve(
            [0, 1, 0, 1], [0.5, 0.5, 0.2, 0.9]
        )
        self.assertEqual(thresholds, [float("inf"), 0.9, 0.5, 0.2])
        self.assertEqual(false_positive, [0.0, 0.0, 0.5, 1.0])
        self.assertEqual(true_positive, [0.0, 0.5, 1.0, 1.0])
        with self.assertRaises(ValueError):
            roc_curve([1, 1], [0.2, 0.8])


if __name__ == "__main__":
    unittest.main()


    def test_precision_recall_curve_groups_tied_scores(self):
        from classification_metrics import precision_recall_curve

        precision, recall, thresholds = precision_recall_curve(
            [0, 1, 1, 0], [0.1, 0.4, 0.4, 0.8]
        )
        self.assertEqual(thresholds, [0.1, 0.4, 0.8])
        self.assertEqual(recall, [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(precision, [0.5, 2 / 3, 0.0, 1.0])

    def test_precision_recall_curve_validates_inputs(self):
        from classification_metrics import precision_recall_curve

        with self.assertRaises(ValueError):
            precision_recall_curve([], [])
        with self.assertRaises(ValueError):
            precision_recall_curve([0, 1], [0.1])
        with self.assertRaises(ValueError):
            precision_recall_curve([1, 1], [0.2, 0.8])
        with self.assertRaises(ValueError):
            precision_recall_curve([0, 1], [0.2, float("nan")])
