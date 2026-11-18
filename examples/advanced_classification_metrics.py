"""Demonstrate multiclass ranking and multilabel classification metrics."""

from classification_metrics import (
    classification_report,
    cohen_kappa_score,
    fbeta_score,
    hamming_loss,
    multilabel_confusion_matrix,
    multiclass_average_precision_score,
    multiclass_roc_auc_score,
    top_k_accuracy_score,
)
from gaussian_nb import GaussianNB

train_features = [
    [0.0, 0.0], [0.1, 0.2], [-0.1, 0.1],
    [4.0, 4.0], [4.1, 3.9], [3.9, 4.2],
    [8.0, 0.0], [8.2, 0.1], [7.9, -0.1],
]
train_labels = ["low"] * 3 + ["middle"] * 3 + ["high"] * 3
test_features = [
    [0.05, 0.05], [4.05, 4.05], [8.1, 0.05],
    [0.15, 0.05], [4.0, 4.1], [8.0, -0.05],
]
test_labels = ["low", "middle", "high", "low", "middle", "high"]

model = GaussianNB().fit(train_features, train_labels)
predictions = model.predict(test_features)
probabilities = model.predict_proba(test_features)

print("Classification report:", classification_report(
    test_labels, predictions, labels=model.classes_
))
print("Top-2 accuracy:", top_k_accuracy_score(
    test_labels, probabilities, labels=model.classes_, k=2
))
print("Cohen kappa:", cohen_kappa_score(
    test_labels, predictions, labels=model.classes_
))
print("Multiclass ROC AUC:", multiclass_roc_auc_score(
    test_labels, probabilities, labels=model.classes_, average="macro"
))
print("Multiclass average precision:", multiclass_average_precision_score(
    test_labels, probabilities, labels=model.classes_, average="weighted"
))
print("High-class F2:", fbeta_score(
    test_labels, predictions, positive_label="high", beta=2
))

actual_indicators = [[1, 0], [0, 1], [1, 1]]
predicted_indicators = [[1, 1], [0, 0], [1, 0]]
print("Per-label confusion:", multilabel_confusion_matrix(
    actual_indicators, predicted_indicators
))
print("Hamming loss:", hamming_loss(
    actual_indicators, predicted_indicators
))
