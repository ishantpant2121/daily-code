"""Dependency-free metrics for binary and one-vs-rest classification."""

from itertools import zip_longest
from typing import Any, Iterable


def classification_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    positive_label: Any = 1,
    zero_division: int = 0,
) -> dict[str, float | int]:
    """Calculate accuracy, precision, recall, F1, and confusion counts.

    positive_label identifies the positive class; every other label is
    treated as negative. This also supports one-vs-rest evaluation for a
    selected class in a multiclass dataset. Inputs may be one-shot iterables.

    zero_division must be 0 or 1 and is returned for precision, recall, or
    F1 whenever that metric has a zero denominator. Empty inputs and unequal
    input lengths raise ValueError.
    """
    if type(zero_division) is not int or zero_division not in (0, 1):
        raise ValueError("zero_division must be 0 or 1")

    missing = object()
    true_positive = false_positive = 0
    true_negative = false_negative = 0
    correct = total = 0

    for actual, predicted in zip_longest(y_true, y_pred, fillvalue=missing):
        if actual is missing or predicted is missing:
            raise ValueError("y_true and y_pred must have equal lengths")

        total += 1
        if actual == predicted:
            correct += 1

        actual_positive = actual == positive_label
        predicted_positive = predicted == positive_label
        if actual_positive and predicted_positive:
            true_positive += 1
        elif predicted_positive:
            false_positive += 1
        elif actual_positive:
            false_negative += 1
        else:
            true_negative += 1

    if total == 0:
        raise ValueError("y_true and y_pred must not be empty")

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative

    return {
        "accuracy": correct / total,
        "precision": (
            true_positive / precision_denominator
            if precision_denominator
            else float(zero_division)
        ),
        "recall": (
            true_positive / recall_denominator
            if recall_denominator
            else float(zero_division)
        ),
        "f1": (
            2 * true_positive / f1_denominator
            if f1_denominator
            else float(zero_division)
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }
