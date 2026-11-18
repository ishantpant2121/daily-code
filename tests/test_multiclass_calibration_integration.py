"""Integration tests for multiclass calibration with a fitted estimator."""

import unittest

from calibration_metrics import (
    classwise_calibration_curves,
    multiclass_calibration_curve,
    multiclass_expected_calibration_error,
)
from classification_metrics import multiclass_brier_score_loss, multiclass_log_loss
from gaussian_nb import GaussianNB


class MulticlassCalibrationIntegrationTests(unittest.TestCase):
    def test_gaussian_nb_probabilities_feed_multiclass_diagnostics(self):
        features = [
            [0.0, 0.0], [0.1, 0.2], [-0.1, 0.1],
            [4.0, 4.0], [4.1, 3.9], [3.9, 4.2],
            [8.0, 0.0], [8.2, 0.1], [7.9, -0.1],
        ]
        labels = ["low"] * 3 + ["middle"] * 3 + ["high"] * 3
        test_features = [[0.05, 0.05], [4.05, 4.05], [8.1, 0.05]]
        test_labels = ["low", "middle", "high"]
        model = GaussianNB().fit(features, labels)
        probabilities = model.predict_proba(test_features)

        curve = multiclass_calibration_curve(
            test_labels, probabilities, labels=model.classes_, n_bins=3
        )
        self.assertEqual(sum(curve["bin_counts"]), len(test_labels))
        self.assertGreaterEqual(
            multiclass_expected_calibration_error(
                test_labels, probabilities, labels=model.classes_, n_bins=3
            ),
            0.0,
        )
        self.assertEqual(
            list(classwise_calibration_curves(
                test_labels, probabilities, labels=model.classes_, n_bins=3
            )),
            list(model.classes_),
        )
        self.assertGreaterEqual(
            multiclass_brier_score_loss(
                test_labels, probabilities, labels=model.classes_
            ),
            0.0,
        )
        self.assertGreaterEqual(
            multiclass_log_loss(
                test_labels, probabilities, labels=model.classes_
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
