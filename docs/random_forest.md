# Random Forest Classifier

`RandomForestClassifier` is a dependency-free educational implementation of
bootstrap aggregation over CART decision trees.

## Basic use

```python
from random_forest import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=50,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
probabilities = model.predict_proba(x_test)
```

## Sampling behavior

- Each tree draws a bootstrap sample with replacement by default.
- `max_samples=None` draws as many rows as the training set contains.
- An integer `max_samples` selects that many rows; a fraction in `(0, 1]`
  selects at least one row using the floor of the fraction times the sample count.
- Set `bootstrap=False` to fit every tree on the complete training set.
- `max_features` accepts a positive integer, a fraction in `(0, 1]`,
  `"sqrt"`, `"log2"`, or `None`. The feature subset is sampled independently
  at each decision-tree node. If the subset cannot form a valid split, the tree
  checks additional features until it finds one or exhausts the columns.
- A fixed integer `random_state` makes bootstrap and feature sampling
  reproducible.

The forest averages each tree's class probabilities. A tree that did not see a
class in its bootstrap sample contributes zero probability for that class;
columns are aligned to the complete class order observed in the original
training labels.

## Attributes

- `classes_`: class labels in first-seen training order.
- `estimators_`: fitted decision trees.
- `feature_importances_`: normalized average tree impurity importances.
- `n_features_in_`: number of input columns.
- `n_samples_fit_`: original training sample count.
- `max_samples_`: rows drawn for each tree.

## Scope and limitations

Trees are fitted sequentially. Out-of-bag scoring, parallel fitting, sample
weights, class weights, and warm-start fitting are not implemented. The model
is intended for learning and small datasets, not as a replacement for an
optimized production library.

For the underlying split criterion and feature-sampling conventions, see the
[scikit-learn RandomForestClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
and [DecisionTreeClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).
