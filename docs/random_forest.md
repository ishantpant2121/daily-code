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
    oob_score=True,
)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
probabilities = model.predict_proba(x_test)
log_probabilities = model.predict_log_proba(x_test)
leaf_ids = model.apply(x_test)
tree_votes = model.predict_votes(x_test)
print("out-of-bag accuracy:", model.oob_score_)
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
- `feature_importances_std_`: per-feature population standard deviation of
  the individual trees' normalized impurity importances.
- `predict_votes(X)`: hard class-vote counts per sample, with columns in
  `classes_` order. Each row sums to the number of fitted trees.
- `n_features_in_`: number of input columns.
- `n_samples_fit_`: original training sample count.
- `max_samples_`: rows drawn for each tree.
- `estimators_samples_`: the row indices drawn for each fitted tree, including
  repeated indices when bootstrap sampling is enabled.
- `apply(X)`: sample-major terminal-node indices, with one column per tree.
- `predict_log_proba(X)`: natural-log probabilities; zero probabilities are
  represented as negative infinity.
- `oob_score_`: accuracy computed from out-of-bag predictions, available only
  when `oob_score=True`.
- `oob_decision_function_`: class-aligned mean probabilities from trees that
  did not train on each row. A row with no out-of-bag votes contains NaN values.
- `oob_counts_`: number of trees that supplied an out-of-bag prediction for
  each training row.
- `oob_coverage_`: fraction of training rows with at least one OOB vote.

## Out-of-bag evaluation

Set `oob_score=True` to evaluate each training row using only trees whose
bootstrap samples omitted that row. This gives a useful internal generalization
estimate without a separate validation split. It requires `bootstrap=True`.
The score is ordinary accuracy over rows that received at least one out-of-bag
vote. Rows with no votes are excluded from the score and represented by NaN
probabilities; if no row receives a vote, `oob_score_` is NaN. Increase
`n_estimators` when coverage is incomplete. Out-of-bag evaluation is not a
replacement for a final untouched test set.

```python
forest = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    oob_score=True,
    random_state=42,
).fit(x_train, y_train)

print(forest.oob_score_)
print(forest.oob_counts_)
```

## Scope and limitations

Trees are fitted sequentially. Parallel fitting, sample weights, class weights,
and warm-start fitting are not implemented. Out-of-bag scoring uses accuracy;
custom OOB metrics are not currently supported. The model
is intended for learning and small datasets, not as a replacement for an
optimized production library.

For the underlying split criterion and feature-sampling conventions, see the
[scikit-learn RandomForestClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
and [DecisionTreeClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).
