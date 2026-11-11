"""Measure reliability of binary probability predictions."""

from calibration_metrics import (
    calibration_curve,
    expected_calibration_error,
    maximum_calibration_error,
)
from data_splitting import train_test_split
from logistic_regression import LogisticRegression

features = [[float(value), float(value % 4)] for value in range(24)]
labels = [int(value >= 12) for value in range(24)]
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=8
)

model = LogisticRegression(max_iter=2000).fit(x_train, y_train)
positive_column = list(model.classes_).index(1)
probabilities = [row[positive_column] for row in model.predict_proba(x_test)]

print("Calibration bins:", calibration_curve(y_test, probabilities, n_bins=4))
print("Expected calibration error:", expected_calibration_error(y_test, probabilities, n_bins=4))
print("Maximum calibration error:", maximum_calibration_error(y_test, probabilities, n_bins=4))
