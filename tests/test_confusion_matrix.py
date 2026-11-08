"""Tests for multiclass confusion matrices."""

import unittest

from classification_metrics import confusion_matrix


class ConfusionMatrixTests(unittest.TestCase):
    def test_infers_stable_label_order_and_counts_multiclass_pairs(self):
        result = confusion_matrix(
            ["cat", "dog", "cat", "bird"],
            ["dog", "dog", "cat", "cat"],
        )
        self.assertEqual(result, [[1, 1, 0], [0, 1, 0], [1, 0, 0]])

    def test_supports_explicit_labels_and_row_normalization(self):
        result = confusion_matrix(
            ["cat", "cat", "dog"],
            ["cat", "dog", "dog"],
            labels=["cat", "dog", "bird"],
            normalize="true",
        )
        self.assertEqual(result[0], [0.5, 0.5, 0.0])
        self.assertEqual(result[1], [0.0, 1.0, 0.0])
        self.assertEqual(result[2], [0.0, 0.0, 0.0])

    def test_supports_column_and_global_normalization(self):
        actual = ["a", "a", "b", "b"]
        predicted = ["a", "b", "a", "b"]
        self.assertEqual(
            confusion_matrix(actual, predicted, normalize="pred"),
            [[0.5, 0.5], [0.5, 0.5]],
        )
        self.assertEqual(
            confusion_matrix(actual, predicted, normalize="all"),
            [[0.25, 0.25], [0.25, 0.25]],
        )

    def test_rejects_unknown_duplicate_and_unhashable_labels(self):
        with self.assertRaises(ValueError):
            confusion_matrix(["a"], ["b"], labels=["a"])
        with self.assertRaises(ValueError):
            confusion_matrix(["a"], ["a"], labels=["a", "a"])
        with self.assertRaises(ValueError):
            confusion_matrix([[1]], [[1]])
        with self.assertRaises(ValueError):
            confusion_matrix(["a"], ["a"], normalize="bad")


if __name__ == "__main__":
    unittest.main()
