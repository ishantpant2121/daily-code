"""Tests for multiclass Brier and logarithmic probability losses."""

import math
import unittest

from classification_metrics import (
    multiclass_brier_score_loss,
    multiclass_log_loss,
)


class MulticlassProbabilityMetricTests(unittest.TestCase):
    def test_brier_score_sums_squared_errors_across_classes(self):
        score = multiclass_brier_score_loss(
            ["cat", "dog"],
            [[0.8, 0.2], [0.25, 0.75]],
            labels=["cat", "dog"],
        )
        self.assertAlmostEqual(score, 0.1025)

    def test_log_loss_uses_true_class_probability(self):
        score = multiclass_log_loss(
            ["cat", "dog"],
            [[0.8, 0.2], [0.25, 0.75]],
            labels=["cat", "dog"],
        )
        self.assertAlmostEqual(score, (-math.log(0.8) - math.log(0.75)) / 2)

    def test_custom_label_order_and_one_shot_inputs(self):
        score = multiclass_log_loss(
            (value for value in ["dog", "cat"]),
            (iter(row) for row in [[0.1, 0.9], [0.8, 0.2]]),
            labels=(value for value in ["cat", "dog"]),
        )
        self.assertAlmostEqual(score, (-math.log(0.9) - math.log(0.8)) / 2)

    def test_log_loss_clips_zero_probabilities_and_stays_finite(self):
        score = multiclass_log_loss(
            ["cat", "dog"],
            [[1.0, 0.0], [0.0, 1.0]],
            labels=["cat", "dog"],
        )
        self.assertTrue(math.isfinite(score))
        self.assertLess(score, 2e-15)

    def test_rejects_invalid_probability_shapes_and_labels(self):
        invalid_cases = (
            ([], [], ["cat", "dog"]),
            (["cat"], [[0.8, 0.2]], ["cat"]),
            (["cat"], [[0.8, 0.2]], ["cat", "cat"]),
            (["fox"], [[0.8, 0.2]], ["cat", "dog"]),
            (["cat"], [[0.8]], ["cat", "dog"]),
            (["cat"], [[0.8, 0.3]], ["cat", "dog"]),
            (["cat"], [[-0.1, 1.1]], ["cat", "dog"]),
            (["cat"], [[0.8, float("nan")]], ["cat", "dog"]),
            (["cat"], [[0.8, True]], ["cat", "dog"]),
            (["cat"], [[0.8, 10**400]], ["cat", "dog"]),
        )
        for actual, probabilities, labels in invalid_cases:
            with self.subTest(actual=actual, probabilities=probabilities, labels=labels):
                with self.assertRaises(ValueError):
                    multiclass_brier_score_loss(
                        actual, probabilities, labels=labels
                    )

    def test_log_loss_validates_epsilon(self):
        for eps in (0, -1, 0.5, float("nan"), float("inf"), True):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                multiclass_log_loss(
                    ["cat", "dog"],
                    [[0.8, 0.2], [0.2, 0.8]],
                    labels=["cat", "dog"],
                    eps=eps,
                )


if __name__ == "__main__":
    unittest.main()
