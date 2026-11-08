"""Dependency-free metrics for binary and one-vs-rest classification."""

from itertools import zip_longest
import math
from numbers import Real
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



def confusion_matrix(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    labels: Iterable[Any] | None = None,
    normalize: str | None = None,
) -> list[list[int | float]]:
    """Build a multiclass confusion matrix (rows=true, columns=predicted).

    If labels is omitted, class order follows first appearance across y_true
    then y_pred. normalize may be "true" (row-wise), "pred" (column-wise), or
    "all" (divide by the total); zero-support rows/columns remain zero.
    """
    if normalize not in (None, "true", "pred", "all"):
        raise ValueError("normalize must be None, 'true', 'pred', or 'all'")
    actual = list(y_true)
    predicted = list(y_pred)
    if not actual:
        raise ValueError("y_true and y_pred must not be empty")
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must have equal lengths")

    label_to_index: dict[Any, int] = {}
    if labels is None:
        ordered_labels = []
        for value in actual + predicted:
            try:
                if value not in label_to_index:
                    label_to_index[value] = len(ordered_labels)
                    ordered_labels.append(value)
            except TypeError:
                raise ValueError("class labels must be hashable") from None
    else:
        ordered_labels = list(labels)
        if not ordered_labels:
            raise ValueError("labels must not be empty")
        for value in ordered_labels:
            try:
                if value in label_to_index:
                    raise ValueError("labels must not contain duplicates")
                label_to_index[value] = len(label_to_index)
            except TypeError:
                raise ValueError("class labels must be hashable") from None

    size = len(ordered_labels)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for true_value, predicted_value in zip(actual, predicted):
        try:
            row = label_to_index[true_value]
            column = label_to_index[predicted_value]
        except (KeyError, TypeError):
            raise ValueError("observed labels must be included in labels") from None
        matrix[row][column] += 1

    if normalize == "true":
        for row in matrix:
            total = sum(row)
            if total:
                for column in range(size):
                    row[column] /= total
    elif normalize == "pred":
        totals = [sum(matrix[row][column] for row in range(size)) for column in range(size)]
        for row in range(size):
            for column in range(size):
                if totals[column]:
                    matrix[row][column] /= totals[column]
                else:
                    matrix[row][column] = 0.0
    elif normalize == "all":
        total = len(actual)
        matrix = [[value / total for value in row] for row in matrix]
    return matrix



def roc_auc_score(
    y_true: Iterable[Any],
    y_score: Iterable[Real],
    *,
    positive_label: Any = 1,
) -> float:
    """Return binary/one-vs-rest ROC AUC using average ranks for tied scores.

    Scores must be finite real numbers. Every label other than positive_label
    is treated as negative. Both positive and negative examples are required.
    Inputs may be one-shot iterables.
    """
    missing = object()
    pairs: list[tuple[float, bool]] = []
    positives = negatives = 0
    for label, score in zip_longest(y_true, y_score, fillvalue=missing):
        if label is missing or score is missing:
            raise ValueError("y_true and y_score must have equal lengths")
        if isinstance(score, bool) or not isinstance(score, Real):
            raise ValueError("scores must be real numbers")
        try:
            numeric_score = float(score)
        except (OverflowError, ValueError):
            raise ValueError("scores must be finite real numbers") from None
        if not math.isfinite(numeric_score):
            raise ValueError("scores must be finite real numbers")
        is_positive = label == positive_label
        positives += int(is_positive)
        negatives += int(not is_positive)
        pairs.append((numeric_score, is_positive))

    if not pairs:
        raise ValueError("y_true and y_score must not be empty")
    if not positives or not negatives:
        raise ValueError("ROC AUC requires both positive and negative examples")

    pairs.sort(key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            1 for _, is_positive in pairs[index:end] if is_positive
        )
        index = end

    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
