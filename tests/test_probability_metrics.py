"""Tests for binary probability-based classification losses."""

import unittest

from classification_metrics import brier_score_loss


class ProbabilityMetricTests(unittest.TestCase):
    def test_brier_score_matches_mean_squared_probability_error(self):
        self.assertAlmostEqual(
            brier_score_loss([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]),
            0.158125,
        )

    def test_brier_score_supports_generators_and_custom_labels(self):
        self.assertAlmostEqual(
            brier_score_loss(
                (value for value in ["no", "yes"]),
                (value for value in [0.1, 0.9]),
                positive_label="yes",
            ),
            0.01,
        )

    def test_brier_score_rejects_invalid_inputs(self):
        cases = (
            ([], []),
            ([0, 1], [0.2]),
            ([0, 1], [-0.1, 0.8]),
            ([0, 1], [0.2, 1.1]),
            ([0, 1], [0.2, float("nan")]),
            ([0, 1], [0.2, True]),
            ([0, 1], [0.2, 10**400]),
        )
        for actual, probabilities in cases:
            with self.subTest(actual=actual, probabilities=probabilities):
                with self.assertRaises(ValueError):
                    brier_score_loss(actual, probabilities)


if __name__ == "__main__":
    unittest.main()
