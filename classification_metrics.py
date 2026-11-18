"""Dependency-free metrics for binary and multiclass classification."""

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



def balanced_accuracy_score(y_true: Iterable[Any], y_pred: Iterable[Any]) -> float:
    """Return the unweighted mean recall across classes present in y_true.

    This multiclass-capable score gives each observed true class equal weight,
    so it is less dominated by majority classes than ordinary accuracy.
    """
    actual = list(y_true)
    predicted = list(y_pred)
    if not actual:
        raise ValueError("y_true and y_pred must not be empty")
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must have equal lengths")
    support: dict[Any, int] = {}
    correct: dict[Any, int] = {}
    try:
        for truth, estimate in zip(actual, predicted):
            support[truth] = support.get(truth, 0) + 1
            correct[truth] = correct.get(truth, 0) + int(truth == estimate)
    except TypeError:
        raise ValueError("true class labels must be hashable") from None
    recalls = [correct[label] / count for label, count in support.items()]
    return math.fsum(recalls) / len(recalls)



def matthews_corrcoef(y_true: Iterable[Any], y_pred: Iterable[Any]) -> float:
    """Return the multiclass Matthews correlation coefficient.

    The coefficient ranges from -1 (total disagreement) through 0 (no
    correlation) to 1 (perfect agreement). A zero denominator returns 0.0.
    """
    actual = list(y_true)
    predicted = list(y_pred)
    matrix = confusion_matrix(actual, predicted)
    size = len(matrix)
    total = len(actual)
    true_counts = [sum(row) for row in matrix]
    predicted_counts = [
        sum(matrix[row][column] for row in range(size))
        for column in range(size)
    ]
    correct = sum(matrix[index][index] for index in range(size))
    numerator = correct * total - sum(
        truth * estimate
        for truth, estimate in zip(true_counts, predicted_counts)
    )
    true_term = total * total - sum(value * value for value in true_counts)
    predicted_term = total * total - sum(value * value for value in predicted_counts)
    denominator = math.sqrt(true_term * predicted_term)
    return 0.0 if denominator == 0 else numerator / denominator



def precision_recall_curve(
    y_true: Iterable[Any],
    y_score: Iterable[Real],
    *,
    positive_label: Any = 1,
) -> tuple[list[float], list[float], list[float]]:
    """Return precision, recall, and ascending score thresholds for binary data.

    Equal scores are processed together so their internal order cannot affect
    the curve. Precision and recall contain one extra terminal point (1.0,
    0.0), matching the conventional threshold-curve representation.
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
        except (OverflowError, TypeError, ValueError):
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
        raise ValueError("precision-recall curve requires both classes")

    pairs.sort(key=lambda item: item[0], reverse=True)
    precision_desc: list[float] = []
    recall_desc: list[float] = []
    thresholds_desc: list[float] = []
    true_positive = false_positive = 0
    index = 0
    while index < len(pairs):
        threshold = pairs[index][0]
        end = index
        while end < len(pairs) and pairs[end][0] == threshold:
            if pairs[end][1]:
                true_positive += 1
            else:
                false_positive += 1
            end += 1
        precision_desc.append(true_positive / (true_positive + false_positive))
        recall_desc.append(true_positive / positives)
        thresholds_desc.append(threshold)
        index = end

    precision = list(reversed(precision_desc))
    recall = list(reversed(recall_desc))
    thresholds = list(reversed(thresholds_desc))
    precision.append(1.0)
    recall.append(0.0)
    return precision, recall, thresholds



def average_precision_score(
    y_true: Iterable[Any],
    y_score: Iterable[Real],
    *,
    positive_label: Any = 1,
) -> float:
    """Return non-interpolated average precision for binary score predictions.

    Average precision sums each recall increase multiplied by the precision
    at that threshold; it is not trapezoidal precision-recall curve area.
    """
    precision, recall, _ = precision_recall_curve(
        y_true, y_score, positive_label=positive_label
    )
    return math.fsum(
        (recall[index] - recall[index + 1]) * precision[index]
        for index in range(len(recall) - 1)
    )



def _binary_probability_pairs(
    y_true: Iterable[Any],
    y_probability: Iterable[Real],
    positive_label: Any,
) -> list[tuple[int, float]]:
    missing = object()
    pairs = []
    for label, probability in zip_longest(y_true, y_probability, fillvalue=missing):
        if label is missing or probability is missing:
            raise ValueError("y_true and y_probability must have equal lengths")
        if isinstance(probability, bool) or not isinstance(probability, Real):
            raise ValueError("probabilities must be real numbers")
        try:
            numeric = float(probability)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("probabilities must be finite values in [0, 1]") from None
        if not math.isfinite(numeric) or not 0 <= numeric <= 1:
            raise ValueError("probabilities must be finite values in [0, 1]")
        pairs.append((int(label == positive_label), numeric))
    if not pairs:
        raise ValueError("y_true and y_probability must not be empty")
    return pairs


def brier_score_loss(
    y_true: Iterable[Any],
    y_probability: Iterable[Real],
    *,
    positive_label: Any = 1,
) -> float:
    """Return the mean squared error of binary positive-class probabilities."""
    pairs = _binary_probability_pairs(y_true, y_probability, positive_label)
    return math.fsum((target - probability) ** 2 for target, probability in pairs) / len(pairs)



def log_loss(
    y_true: Iterable[Any],
    y_probability: Iterable[Real],
    *,
    positive_label: Any = 1,
    eps: float = 1e-15,
) -> float:
    """Return binary cross-entropy after clipping probabilities to [eps, 1-eps]."""
    if isinstance(eps, bool) or not isinstance(eps, Real):
        raise ValueError("eps must be a real number between 0 and 0.5")
    try:
        epsilon = float(eps)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("eps must be a finite number between 0 and 0.5") from None
    if not math.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("eps must be a finite number between 0 and 0.5")
    pairs = _binary_probability_pairs(y_true, y_probability, positive_label)
    losses = []
    for target, probability in pairs:
        clipped = min(1.0 - epsilon, max(epsilon, probability))
        losses.append(
            -math.log(clipped) if target else -math.log1p(-clipped)
        )
    return math.fsum(losses) / len(losses)



def roc_curve(
    y_true: Iterable[Any],
    y_score: Iterable[Real],
    *,
    positive_label: Any = 1,
) -> tuple[list[float], list[float], list[float]]:
    """Return false-positive rates, true-positive rates, and descending thresholds.

    The first point is (0, 0) at an infinite threshold. Equal scores are
    processed together so their order cannot change the curve.
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
        except (OverflowError, TypeError, ValueError):
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
        raise ValueError("ROC curve requires both classes")

    pairs.sort(key=lambda item: item[0], reverse=True)
    false_positive_rates = [0.0]
    true_positive_rates = [0.0]
    thresholds = [math.inf]
    true_positive = false_positive = 0
    index = 0
    while index < len(pairs):
        threshold = pairs[index][0]
        end = index
        while end < len(pairs) and pairs[end][0] == threshold:
            if pairs[end][1]:
                true_positive += 1
            else:
                false_positive += 1
            end += 1
        false_positive_rates.append(false_positive / negatives)
        true_positive_rates.append(true_positive / positives)
        thresholds.append(threshold)
        index = end
    return false_positive_rates, true_positive_rates, thresholds


def _multiclass_probability_inputs(
    y_true: Iterable[Any],
    y_probability: Iterable[Iterable[Real]],
    labels: Iterable[Any],
) -> tuple[list[int], list[Any], list[list[float]]]:
    """Validate class-aligned probability rows and map targets to columns."""
    try:
        actual = list(y_true)
        rows = [list(row) for row in y_probability]
        classes = list(labels)
    except TypeError:
        raise ValueError("labels and probabilities must be iterable") from None
    if len(classes) < 2:
        raise ValueError("at least two class labels are required")
    if not actual or len(actual) != len(rows):
        raise ValueError("labels and probabilities must be non-empty and have equal lengths")

    positions: dict[Any, int] = {}
    try:
        for index, label in enumerate(classes):
            if label in positions:
                raise ValueError("class labels must be unique")
            positions[label] = index
        targets = [positions[label] for label in actual]
    except TypeError:
        raise ValueError("class labels must be hashable") from None
    except KeyError:
        raise ValueError("y_true contains a label not present in labels") from None

    converted_rows = []
    for row in rows:
        if len(row) != len(classes):
            raise ValueError("each probability row must contain one value per class")
        converted = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("probabilities must be finite values in [0, 1]")
            try:
                number = float(value)
            except (OverflowError, TypeError, ValueError):
                raise ValueError("probabilities must be finite values in [0, 1]") from None
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError("probabilities must be finite values in [0, 1]")
            converted.append(number)
        if not math.isclose(math.fsum(converted), 1.0, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("each probability row must sum to one")
        converted_rows.append(converted)
    return targets, classes, converted_rows


def multiclass_brier_score_loss(
    y_true: Iterable[Any],
    y_probability: Iterable[Iterable[Real]],
    *,
    labels: Iterable[Any],
) -> float:
    """Return mean summed squared error across all class probabilities.

    The class-column order is defined by labels. For each sample, the score
    sums (p_k - 1[y=k]) squared over classes, then averages over samples.
    This multiclass definition is the unhalved sum across class columns.
    """
    targets, classes, rows = _multiclass_probability_inputs(
        y_true, y_probability, labels
    )
    losses = []
    for target, row in zip(targets, rows):
        losses.append(
            math.fsum(
                (probability - float(index == target)) ** 2
                for index, probability in enumerate(row)
            )
        )
    return math.fsum(losses) / len(rows)


def multiclass_log_loss(
    y_true: Iterable[Any],
    y_probability: Iterable[Iterable[Real]],
    *,
    labels: Iterable[Any],
    eps: Real = 1e-15,
) -> float:
    """Return multiclass negative log likelihood with clipping and renormalization.

    Probability columns follow labels. Values below eps are clipped, then each
    row is renormalized before the true-class probability is scored.
    """
    targets, classes, rows = _multiclass_probability_inputs(
        y_true, y_probability, labels
    )
    if isinstance(eps, bool) or not isinstance(eps, Real):
        raise ValueError("eps must be a finite positive number smaller than 1 / n_classes")
    try:
        epsilon = float(eps)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("eps must be a finite positive number smaller than 1 / n_classes") from None
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0 / len(classes):
        raise ValueError("eps must be a finite positive number smaller than 1 / n_classes")

    losses = []
    for target, row in zip(targets, rows):
        clipped = [max(epsilon, probability) for probability in row]
        total = math.fsum(clipped)
        normalized_true_probability = clipped[target] / total
        losses.append(-math.log(normalized_true_probability))
    return math.fsum(losses) / len(losses)


def top_k_accuracy_score(
    y_true: Iterable[Any],
    y_probability: Iterable[Iterable[Real]],
    *,
    labels: Iterable[Any],
    k: int = 2,
) -> float:
    """Return the fraction of samples whose true class is among the top k.

    Probability columns follow labels. Equal scores are resolved by the
    supplied class order, making the result deterministic.
    """
    targets, classes, rows = _multiclass_probability_inputs(
        y_true, y_probability, labels
    )
    if type(k) is not int or not 1 <= k <= len(classes):
        raise ValueError("k must be an integer between 1 and the number of classes")
    correct = 0
    for target, row in zip(targets, rows):
        ranked = sorted(range(len(classes)), key=lambda index: (-row[index], index))
        correct += int(target in ranked[:k])
    return correct / len(rows)


def fbeta_score(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    beta: Real = 1.0,
    positive_label: Any = 1,
    zero_division: int = 0,
) -> float:
    """Return the binary F-beta score with configurable precision/recall weight."""
    if isinstance(beta, bool) or not isinstance(beta, Real):
        raise ValueError("beta must be a positive finite real number")
    try:
        value = float(beta)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("beta must be a positive finite real number") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("beta must be a positive finite real number")
    if type(zero_division) is not int or zero_division not in (0, 1):
        raise ValueError("zero_division must be 0 or 1")

    metrics = classification_metrics(
        y_true, y_pred, positive_label=positive_label, zero_division=zero_division
    )
    precision = metrics["precision"]
    recall = metrics["recall"]
    beta_squared = value * value
    denominator = beta_squared * precision + recall
    if denominator == 0:
        return float(zero_division)
    return (1.0 + beta_squared) * precision * recall / denominator


def classification_report(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    labels: Iterable[Any] | None = None,
    zero_division: int = 0,
) -> dict[str, Any]:
    """Return per-class precision/recall/F1 plus macro, micro, and weighted averages.

    If labels is omitted, class order follows first appearance across y_true
    and y_pred. Explicit labels can retain absent classes in the report.
    """
    if type(zero_division) is not int or zero_division not in (0, 1):
        raise ValueError("zero_division must be 0 or 1")
    actual = list(y_true)
    predicted = list(y_pred)
    if not actual:
        raise ValueError("y_true and y_pred must not be empty")
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must have equal lengths")

    if labels is None:
        class_labels = []
        seen = set()
        try:
            for value in actual + predicted:
                if value not in seen:
                    seen.add(value)
                    class_labels.append(value)
        except TypeError:
            raise ValueError("class labels must be hashable") from None
    else:
        class_labels = list(labels)
        if not class_labels:
            raise ValueError("labels must not be empty")
        seen = set()
        try:
            for value in class_labels:
                if value in seen:
                    raise ValueError("labels must not contain duplicates")
                seen.add(value)
        except TypeError:
            raise ValueError("class labels must be hashable") from None

    matrix = confusion_matrix(actual, predicted, labels=class_labels)
    size = len(matrix)
    supports = [sum(row) for row in matrix]
    predicted_counts = [
        sum(matrix[row][column] for row in range(size))
        for column in range(size)
    ]
    per_class = {}
    precision_values = []
    recall_values = []
    f1_values = []
    for index, label in enumerate(class_labels):
        true_positive = matrix[index][index]
        false_positive = predicted_counts[index] - true_positive
        false_negative = supports[index] - true_positive
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        f1_denominator = 2 * true_positive + false_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator else float(zero_division)
        )
        recall = (
            true_positive / recall_denominator
            if recall_denominator else float(zero_division)
        )
        f1 = (
            2 * true_positive / f1_denominator
            if f1_denominator else float(zero_division)
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": supports[index],
        }
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    total = len(actual)
    correct = sum(matrix[index][index] for index in range(size))
    micro_false_positive = sum(predicted_counts) - correct
    micro_false_negative = total - correct
    micro_precision_denominator = correct + micro_false_positive
    micro_recall_denominator = correct + micro_false_negative
    micro_precision = (
        correct / micro_precision_denominator
        if micro_precision_denominator else float(zero_division)
    )
    micro_recall = (
        correct / micro_recall_denominator
        if micro_recall_denominator else float(zero_division)
    )
    micro_f1_denominator = 2 * correct + micro_false_positive + micro_false_negative
    micro_f1 = (
        2 * correct / micro_f1_denominator
        if micro_f1_denominator else float(zero_division)
    )
    weighted = lambda values: math.fsum(
        value * support for value, support in zip(values, supports)
    ) / total
    return {
        "labels": tuple(class_labels),
        "per_class": per_class,
        "accuracy": correct / total,
        "macro_avg": {
            "precision": math.fsum(precision_values) / size,
            "recall": math.fsum(recall_values) / size,
            "f1": math.fsum(f1_values) / size,
            "support": total,
        },
        "micro_avg": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "support": total,
        },
        "weighted_avg": {
            "precision": weighted(precision_values),
            "recall": weighted(recall_values),
            "f1": weighted(f1_values),
            "support": total,
        },
        "support": total,
    }


def cohen_kappa_score(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    labels: Iterable[Any] | None = None,
    weights: str | None = None,
) -> float:
    """Return Cohen's kappa, optionally with linear or quadratic disagreement weights.

    A degenerate expected-disagreement denominator raises ValueError rather
    than returning a misleading finite coefficient.
    """
    if weights not in (None, "linear", "quadratic"):
        raise ValueError("weights must be None, 'linear', or 'quadratic'")
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    size = len(matrix)
    if size < 2:
        raise ValueError("Cohen kappa requires at least two classes")
    total = sum(sum(row) for row in matrix)
    if total == 0:
        raise ValueError("y_true and y_pred must not be empty")
    row_totals = [sum(row) for row in matrix]
    column_totals = [
        sum(matrix[row][column] for row in range(size))
        for column in range(size)
    ]

    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for row in range(size):
        for column in range(size):
            distance = abs(row - column) / (size - 1)
            if weights == "linear":
                disagreement = distance
            elif weights == "quadratic":
                disagreement = distance * distance
            else:
                disagreement = float(row != column)
            observed_disagreement += disagreement * matrix[row][column]
            expected_disagreement += (
                disagreement * row_totals[row] * column_totals[column] / total
            )
    if expected_disagreement == 0:
        raise ValueError("Cohen kappa is undefined when expected disagreement is zero")
    return 1.0 - observed_disagreement / expected_disagreement


def _multilabel_indicator_rows(
    y_true: Iterable[Iterable[Any]],
    y_pred: Iterable[Iterable[Any]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Validate aligned binary indicator matrices."""
    try:
        actual = [list(row) for row in y_true]
        predicted = [list(row) for row in y_pred]
    except TypeError:
        raise ValueError("multilabel inputs must be iterable rows") from None
    if not actual or len(actual) != len(predicted):
        raise ValueError("multilabel inputs must have equal non-zero row counts")
    width = len(actual[0])
    if width == 0:
        raise ValueError("multilabel inputs must contain at least one label column")
    if any(len(row) != width for row in actual + predicted):
        raise ValueError("all multilabel rows must have equal lengths")
    converted = []
    for rows in (actual, predicted):
        normalized = []
        for row in rows:
            values = []
            for value in row:
                if value not in (0, 1, False, True):
                    raise ValueError("multilabel indicators must contain only 0, 1, or booleans")
                values.append(int(value))
            normalized.append(values)
        converted.append(normalized)
    return converted[0], converted[1]


def multilabel_confusion_matrix(
    y_true: Iterable[Iterable[Any]],
    y_pred: Iterable[Iterable[Any]],
) -> list[list[list[int]]]:
    """Return one 2x2 (TN, FP; FN, TP) confusion matrix per label column."""
    actual, predicted = _multilabel_indicator_rows(y_true, y_pred)
    width = len(actual[0])
    matrices = []
    for column in range(width):
        true_negative = false_positive = false_negative = true_positive = 0
        for actual_row, predicted_row in zip(actual, predicted):
            truth = actual_row[column]
            estimate = predicted_row[column]
            if truth and estimate:
                true_positive += 1
            elif estimate:
                false_positive += 1
            elif truth:
                false_negative += 1
            else:
                true_negative += 1
        matrices.append([
            [true_negative, false_positive],
            [false_negative, true_positive],
        ])
    return matrices


def hamming_loss(
    y_true: Iterable[Iterable[Any]],
    y_pred: Iterable[Iterable[Any]],
) -> float:
    """Return the fraction of incorrect label indicators in multilabel data."""
    actual, predicted = _multilabel_indicator_rows(y_true, y_pred)
    mismatches = sum(
        left != right
        for actual_row, predicted_row in zip(actual, predicted)
        for left, right in zip(actual_row, predicted_row)
    )
    return mismatches / (len(actual) * len(actual[0]))


def multiclass_roc_auc_score(
    y_true: Iterable[Any],
    y_probability: Iterable[Iterable[Real]],
    *,
    labels: Iterable[Any],
    average: str = "macro",
) -> float:
    """Return multiclass ROC AUC using one-vs-rest class scores.

    average may be "macro", "weighted", or "micro". Every class must have
    positive and negative examples for macro/weighted averaging.
    """
    if average not in ("macro", "weighted", "micro"):
        raise ValueError("average must be 'macro', 'weighted', or 'micro'")
    actual, classes, rows = _multiclass_probability_inputs(
        y_true, y_probability, labels
    )
    if average == "micro":
        flattened_true = [
            int(target == column)
            for target in actual
            for column in range(len(classes))
        ]
        flattened_score = [value for row in rows for value in row]
        return roc_auc_score(flattened_true, flattened_score, positive_label=1)

    scores = []
    supports = []
    for column, label in enumerate(classes):
        binary_true = [value == label for value in actual]
        binary_score = [row[column] for row in rows]
        scores.append(roc_auc_score(binary_true, binary_score, positive_label=True))
        supports.append(sum(binary_true))
    if average == "macro":
        return math.fsum(scores) / len(scores)
    return math.fsum(
        score * support for score, support in zip(scores, supports)
    ) / sum(supports)


def multiclass_average_precision_score(
    y_true: Iterable[Any],
    y_probability: Iterable[Iterable[Real]],
    *,
    labels: Iterable[Any],
    average: str = "macro",
) -> float:
    """Return multiclass average precision with macro, weighted, or micro averaging."""
    if average not in ("macro", "weighted", "micro"):
        raise ValueError("average must be 'macro', 'weighted', or 'micro'")
    actual, classes, rows = _multiclass_probability_inputs(
        y_true, y_probability, labels
    )
    if average == "micro":
        flattened_true = [
            int(target == column)
            for target in actual
            for column in range(len(classes))
        ]
        flattened_score = [value for row in rows for value in row]
        return average_precision_score(flattened_true, flattened_score, positive_label=1)

    scores = []
    supports = []
    for column, label in enumerate(classes):
        binary_true = [value == label for value in actual]
        binary_score = [row[column] for row in rows]
        scores.append(
            average_precision_score(binary_true, binary_score, positive_label=True)
        )
        supports.append(sum(binary_true))
    if average == "macro":
        return math.fsum(scores) / len(scores)
    return math.fsum(
        score * support for score, support in zip(scores, supports)
    ) / sum(supports)
