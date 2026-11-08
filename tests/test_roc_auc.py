"""Tests for dependency-free ROC AUC scoring."""

import unittest

from classification_metrics import roc_auc_score


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


if __name__ == "__main__":
    unittest.main()
