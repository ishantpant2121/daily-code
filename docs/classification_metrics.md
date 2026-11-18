# Classification metrics

The functions in `classification_metrics.py` are dependency-free evaluation
helpers. Use predictions from held-out data; scores computed on training data
usually overstate generalization.

## Top-k accuracy

`top_k_accuracy_score(y_true, y_probability, labels=..., k=2)` checks whether
the true class is among the k largest class probabilities. Probability columns
must follow the explicit labels order. Equal probabilities are resolved by
that order, so ties are deterministic.

## F-beta and classification reports

`fbeta_score` evaluates one positive class. Beta below one emphasizes
precision; beta above one emphasizes recall. `classification_report` returns
per-class precision, recall, F1 and support, together with accuracy and
macro-, micro-, and support-weighted averages. An explicit labels sequence can
retain classes that have no observations in a particular evaluation split.
`zero_division` controls values whose denominator is zero.

## Cohen kappa

`cohen_kappa_score` measures agreement beyond chance. It supports unweighted,
linear, and quadratic disagreement weights. It raises ValueError when the
expected-disagreement denominator is zero because the coefficient is
undefined for that degenerate marginal distribution.

## Multilabel indicators

`multilabel_confusion_matrix` returns one 2x2 matrix per indicator column,
with rows representing true 0/1 and columns representing predicted 0/1.
`hamming_loss` is the fraction of all label indicators that differ. Inputs
must be aligned rectangular matrices containing only 0/1 or booleans.

## Multiclass ranking metrics

`multiclass_roc_auc_score` and `multiclass_average_precision_score` support
one-vs-rest macro and support-weighted averages, plus micro averaging over the
flattened one-hot targets and probability matrix. Macro and weighted one-vs-rest
scores require each class to have both positive and negative examples in the
evaluation data. Probability rows must be finite, lie in [0, 1], sum to one,
and use the supplied labels column order.

## Example

```python
from classification_metrics import (
    classification_report,
    cohen_kappa_score,
    multiclass_average_precision_score,
    multiclass_roc_auc_score,
    top_k_accuracy_score,
)

report = classification_report(y_test, predictions, labels=model.classes_)
print(report["macro_avg"])
print(cohen_kappa_score(y_test, predictions, labels=model.classes_))
print(top_k_accuracy_score(
    y_test, probabilities, labels=model.classes_, k=2
))
print(multiclass_roc_auc_score(
    y_test, probabilities, labels=model.classes_, average="macro"
))
print(multiclass_average_precision_score(
    y_test, probabilities, labels=model.classes_, average="weighted"
))
```

The metrics summarize different properties and should not be treated as
interchangeable. For example, top-k accuracy ignores probability calibration,
while ROC AUC and average precision assess ranking. Choose the metric and
averaging scheme that matches the evaluation objective and class distribution.
