"""Tests for multiclass top-label and one-vs-rest calibration metrics."""

import unittest

from calibration_metrics import (
    classwise_calibration_curves,
    multiclass_calibration_curve,
    multiclass_expected_calibration_error,
    multiclass_maximum_calibration_error,
    multiclass_root_mean_squared_calibration_error,
)


class MulticlassCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.labels = ["cat", "dog", "bird"]
        self.y_true = ["cat", "dog", "bird"]
        self.probabilities = [
            [0.9, 0.05, 0.05],
            [0.7, 0.2, 0.1],
            [0.1, 0.2, 0.7],
        ]

    def test_top_label_curve_reports_accuracy_confidence_and_counts(self):
        result = multiclass_calibration_curve(
            self.y_true, self.probabilities, labels=self.labels, n_bins=2
        )
        self.assertEqual(result["bin_counts"], [3])
        self.assertAlmostEqual(result["prob_true"][0], 2 / 3)
        self.assertAlmostEqual(result["prob_pred"][0], 0.7666666666666667)

    def test_top_label_error_norms_match_single_bin_gap(self):
        arguments = {
            "labels": self.labels,
            "n_bins": 2,
        }
        self.assertAlmostEqual(
            multiclass_expected_calibration_error(
                self.y_true, self.probabilities, **arguments
            ),
            0.1,
        )
        self.assertAlmostEqual(
            multiclass_maximum_calibration_error(
                self.y_true, self.probabilities, **arguments
            ),
            0.1,
        )
        self.assertAlmostEqual(
            multiclass_root_mean_squared_calibration_error(
                self.y_true, self.probabilities, **arguments
            ),
            0.1,
        )

    def test_perfect_one_hot_predictions_have_zero_error(self):
        probabilities = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        self.assertEqual(
            multiclass_expected_calibration_error(
                self.y_true, probabilities, labels=self.labels
            ),
            0.0,
        )

    def test_quantile_bins_balance_top_label_counts(self):
        result = multiclass_calibration_curve(
            ["a", "b", "a", "b"],
            [[0.6, 0.4], [0.3, 0.7], [0.8, 0.2], [0.1, 0.9]],
            labels=["a", "b"],
            n_bins=2,
            strategy="quantile",
        )
        self.assertEqual(result["bin_counts"], [2, 2])

    def test_classwise_curves_follow_supplied_class_order(self):
        curves = classwise_calibration_curves(
            self.y_true, self.probabilities, labels=self.labels, n_bins=2
        )
        self.assertEqual(list(curves), self.labels)
        self.assertEqual(curves["cat"]["bin_counts"], [1, 2])
        self.assertEqual(curves["cat"]["prob_true"], [0.0, 0.5])
        self.assertEqual(sum(curves["dog"]["bin_counts"]), len(self.y_true))

    def test_accepts_one_shot_outer_and_inner_iterables(self):
        result = multiclass_calibration_curve(
            (value for value in self.y_true),
            (iter(row) for row in self.probabilities),
            labels=(value for value in self.labels),
            n_bins=3,
        )
        self.assertEqual(sum(result["bin_counts"]), 3)

    def test_rejects_invalid_shapes_labels_and_probabilities(self):
        invalid_cases = (
            (["cat"], [[0.8, 0.2]], ["cat", "cat"]),
            (["fox"], [[0.8, 0.1, 0.1]], self.labels),
            (["cat"], [[0.8, 0.2]], self.labels),
            (["cat"], [[0.8, 0.1, 0.1, 0.0]], self.labels),
            (["cat"], [[0.8, 0.3, -0.1]], self.labels),
            (["cat"], [[0.8, 0.1, float("nan")]], self.labels),
            (["cat"], [[0.8, 0.1, True]], self.labels),
            (["cat"], [[0.8, 0.1, 10**400]], self.labels),
        )
        for actual, probabilities, labels in invalid_cases:
            with self.subTest(actual=actual, probabilities=probabilities, labels=labels):
                with self.assertRaises(ValueError):
                    multiclass_calibration_curve(
                        actual, probabilities, labels=labels
                    )

    def test_rejects_invalid_bins_strategy_and_unhashable_labels(self):
        for n_bins in (0, True):
            with self.subTest(n_bins=n_bins), self.assertRaises(ValueError):
                multiclass_calibration_curve(
                    self.y_true, self.probabilities, labels=self.labels, n_bins=n_bins
                )
        with self.assertRaises(ValueError):
            multiclass_calibration_curve(
                self.y_true, self.probabilities, labels=self.labels, strategy="adaptive"
            )
        with self.assertRaises(ValueError):
            multiclass_calibration_curve(
                [["cat"], "dog"],
                [[0.8, 0.2], [0.2, 0.8]],
                labels=[["cat"], "dog"],
            )


if __name__ == "__main__":
    unittest.main()
