import unittest

from calibration_metrics import calibration_curve, expected_calibration_error, maximum_calibration_error
from data_splitting import train_test_split
from logistic_regression import LogisticRegression


class CalibrationMetricsIntegrationTests(unittest.TestCase):
    def test_reliability_metrics_accept_logistic_regression_probabilities(self):
        features = [[float(value), float(value % 3)] for value in range(24)]
        labels = [0] * 12 + [1] * 12
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.25, random_state=12
        )
        model = LogisticRegression(max_iter=2000).fit(x_train, y_train)
        positive_column = list(model.classes_).index(1)
        probabilities = [row[positive_column] for row in model.predict_proba(x_test)]
        curve = calibration_curve(y_test, probabilities, n_bins=4)
        self.assertEqual(sum(curve["bin_counts"]), len(y_test))
        self.assertGreaterEqual(expected_calibration_error(y_test, probabilities, n_bins=4), 0.0)
        self.assertLessEqual(maximum_calibration_error(y_test, probabilities, n_bins=4), 1.0)


if __name__ == "__main__":
    unittest.main()
