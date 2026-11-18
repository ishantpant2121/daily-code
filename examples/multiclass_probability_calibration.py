"""Evaluate top-label and classwise calibration for a multiclass model."""

from calibration_metrics import (
    classwise_calibration_curves,
    multiclass_calibration_curve,
    multiclass_expected_calibration_error,
    multiclass_maximum_calibration_error,
    multiclass_root_mean_squared_calibration_error,
)
from classification_metrics import multiclass_brier_score_loss, multiclass_log_loss
from gaussian_nb import GaussianNB

train_features = [
    [0.0, 0.0], [0.1, 0.2], [-0.1, 0.1],
    [4.0, 4.0], [4.1, 3.9], [3.9, 4.2],
    [8.0, 0.0], [8.2, 0.1], [7.9, -0.1],
]
train_labels = ["low"] * 3 + ["middle"] * 3 + ["high"] * 3
test_features = [[0.05, 0.05], [4.05, 4.05], [8.1, 0.05]]
test_labels = ["low", "middle", "high"]

model = GaussianNB().fit(train_features, train_labels)
probabilities = model.predict_proba(test_features)
classes = model.classes_

curve = multiclass_calibration_curve(
    test_labels, probabilities, labels=classes, n_bins=3, strategy="quantile"
)
print("Top-label bins:", curve)
print(
    "Top-label ECE:",
    multiclass_expected_calibration_error(
        test_labels, probabilities, labels=classes, n_bins=3
    ),
)
print(
    "Top-label MCE:",
    multiclass_maximum_calibration_error(
        test_labels, probabilities, labels=classes, n_bins=3
    ),
)
print(
    "Top-label RMSCE:",
    multiclass_root_mean_squared_calibration_error(
        test_labels, probabilities, labels=classes, n_bins=3
    ),
)
print(
    "Classwise curves:",
    classwise_calibration_curves(
        test_labels, probabilities, labels=classes, n_bins=3
    ),
)
print(
    "Multiclass Brier:",
    multiclass_brier_score_loss(
        test_labels, probabilities, labels=classes
    ),
)
print(
    "Multiclass log loss:",
    multiclass_log_loss(test_labels, probabilities, labels=classes),
)
