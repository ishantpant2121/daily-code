"""Tests for dependency-free classification metrics."""

import unittest

from classification_metrics import (
    balanced_accuracy_score,
    classification_metrics,
    matthews_corrcoef,
)


class ClassificationMetricsTests(unittest.TestCase):
    def test_calculates_binary_metrics_and_confusion_counts(self):
        result = classification_metrics(
            [1, 0, 1, 1, 0],
            [1, 0, 0, 1, 1],
        )
        self.assertAlmostEqual(result["accuracy"], 0.6)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertEqual(
            (
                result["true_positive"],
                result["false_positive"],
                result["true_negative"],
                result["false_negative"],
            ),
            (2, 1, 1, 1),
        )

    def test_supports_custom_positive_labels_and_one_shot_iterables(self):
        result = classification_metrics(
            (label for label in ["yes", "no", "yes"]),
            (label for label in ["no", "no", "yes"]),
            positive_label="yes",
        )
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 2 / 3)

    def test_supports_one_vs_rest_for_a_selected_multiclass_label(self):
        result = classification_metrics(
            ["cat", "dog", "bird", "cat"],
            ["cat", "cat", "bird", "dog"],
            positive_label="cat",
        )
        self.assertEqual(
            (
                result["true_positive"],
                result["false_positive"],
                result["true_negative"],
                result["false_negative"],
            ),
            (1, 1, 1, 1),
        )
        self.assertEqual(result["accuracy"], 0.5)

    def test_zero_division_policy_is_explicit(self):
        default = classification_metrics([0, 0], [0, 0])
        self.assertEqual((default["precision"], default["recall"], default["f1"]), (0.0, 0.0, 0.0))

        configured = classification_metrics([0, 0], [0, 0], zero_division=1)
        self.assertEqual((configured["precision"], configured["recall"], configured["f1"]), (1.0, 1.0, 1.0))

    def test_balanced_accuracy_weights_each_true_class_equally(self):
        actual = [0] * 9 + [1]
        predicted = [0] * 10
        self.assertAlmostEqual(balanced_accuracy_score(actual, predicted), 0.5)
        self.assertEqual(balanced_accuracy_score([0, 1, 2], [0, 1, 2]), 1.0)
        self.assertEqual(
            balanced_accuracy_score(
                (value for value in [0, 0, 1]),
                (value for value in [0, 1, 1]),
            ),
            0.75,
        )

    def test_balanced_accuracy_rejects_invalid_lengths_and_labels(self):
        with self.assertRaises(ValueError):
            balanced_accuracy_score([], [])
        with self.assertRaises(ValueError):
            balanced_accuracy_score([1, 2], [1])
        with self.assertRaisesRegex(ValueError, "hashable"):
            balanced_accuracy_score([[1], [2]], [1, 2])

    def test_multiclass_matthews_correlation(self):
        self.assertEqual(matthews_corrcoef([0, 1, 2], [0, 1, 2]), 1.0)
        self.assertEqual(matthews_corrcoef([0, 1], [1, 0]), -1.0)
        self.assertEqual(matthews_corrcoef([1, 1, 0, 0], [1, 0, 1, 0]), 0.0)
        self.assertAlmostEqual(
            matthews_corrcoef([0, 1, 2, 2], [0, 2, 1, 2]),
            0.2,
        )
        self.assertEqual(matthews_corrcoef([1, 1], [1, 1]), 0.0)

    def test_matthews_correlation_rejects_empty_and_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            matthews_corrcoef([], [])
        with self.assertRaises(ValueError):
            matthews_corrcoef([1, 2], [1])

    def test_rejects_empty_mismatched_and_invalid_policy_inputs(self):
        with self.assertRaises(ValueError):
            classification_metrics([], [])
        with self.assertRaises(ValueError):
            classification_metrics([1, 0], [1])
        with self.assertRaises(ValueError):
            classification_metrics([1], [1], zero_division=2)


if __name__ == "__main__":
    unittest.main()
