import unittest

from calibration_metrics import (
    calibration_curve,
    expected_calibration_error,
    maximum_calibration_error,
)


class CalibrationCurveTests(unittest.TestCase):
    def test_returns_non_empty_uniform_bins_and_counts(self):
        result = calibration_curve(
            [0, 1, 1, 0, 1],
            [0.05, 0.15, 0.85, 0.95, 1.0],
            n_bins=2,
        )
        self.assertEqual(result["bin_counts"], [2, 3])
        self.assertAlmostEqual(result["prob_pred"][0], 0.1)
        self.assertAlmostEqual(result["prob_true"][0], 0.5)
        self.assertAlmostEqual(result["prob_pred"][1], (0.85 + 0.95 + 1.0) / 3)
        self.assertAlmostEqual(result["prob_true"][1], 2 / 3)

    def test_empty_bins_are_omitted_and_one_shot_inputs_work(self):
        result = calibration_curve(
            (value for value in [1, 0, 1]),
            (value for value in [0.1, 0.18, 0.9]),
            n_bins=5,
        )
        self.assertEqual(result["bin_counts"], [2, 1])

    def test_ece_and_mce_use_only_observed_bins(self):
        labels = [1, 0, 1, 0]
        probabilities = [0.9, 0.1, 0.8, 0.2]
        self.assertAlmostEqual(
            expected_calibration_error(labels, probabilities, n_bins=2),
            0.15,
        )
        self.assertAlmostEqual(
            maximum_calibration_error(labels, probabilities, n_bins=2),
            0.15,
        )

    def test_perfectly_calibrated_groups_have_zero_error(self):
        labels = [0, 0, 1, 1]
        probabilities = [0.0, 0.0, 1.0, 1.0]
        self.assertEqual(expected_calibration_error(labels, probabilities, n_bins=2), 0.0)
        self.assertEqual(maximum_calibration_error(labels, probabilities, n_bins=2), 0.0)

    def test_rejects_invalid_bin_counts_lengths_and_probabilities(self):
        for kwargs in ({"n_bins": 0}, {"n_bins": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                calibration_curve([0, 1], [0.1, 0.9], **kwargs)
        for values in ([], [0.1], [0.1, 1.1], [0.1, float("nan")], [True, 0.5], [10**400, 0.5]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                calibration_curve([0, 1], values)
        with self.assertRaises(ValueError):
            calibration_curve([0, 1], [0.1])


if __name__ == "__main__":
    unittest.main()
