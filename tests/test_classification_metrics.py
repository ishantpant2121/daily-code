"""Tests for dependency-free classification metrics."""

import unittest

from classification_metrics import classification_metrics


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

    def test_rejects_empty_mismatched_and_invalid_policy_inputs(self):
        with self.assertRaises(ValueError):
            classification_metrics([], [])
        with self.assertRaises(ValueError):
            classification_metrics([1, 0], [1])
        with self.assertRaises(ValueError):
            classification_metrics([1], [1], zero_division=2)


if __name__ == "__main__":
    unittest.main()
