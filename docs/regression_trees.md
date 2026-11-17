# Regression Trees

The repository provides two dependency-free regression estimators:

- **DecisionTreeRegressor** fits a CART-style binary tree using squared-error
  reduction and predicts the mean target value in each leaf.
- **RandomForestRegressor** fits multiple regression trees on bootstrap samples
  and averages their predictions.

These implementations are intended for learning and small datasets. They do
not replace optimized scientific-computing libraries.

## Decision tree

~~~python
from decision_tree_regressor import DecisionTreeRegressor

model = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=2,
    max_features=None,
    random_state=42,
)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print("R-squared:", model.score(x_test, y_test))
print("feature importance:", model.feature_importances_)
~~~

The split criterion is squared error (variance reduction). Candidate thresholds
are placed between adjacent distinct feature values. The minimum impurity
decrease uses the weighted reduction in mean squared error over the original
training set. Tree construction is iterative and avoids recursive calls.

Supported controls:

- **max_depth**: positive integer or None.
- **min_samples_split**: minimum number of rows required to consider a split.
- **min_samples_leaf**: minimum number of training rows in either child.
- **min_impurity_decrease**: finite non-negative weighted MSE reduction.
- **max_features**: None, a positive integer, a fraction in (0, 1], sqrt, or
  log2. If the sampled columns cannot form a valid split, other columns are
  inspected until a valid split is found or all columns are exhausted.
- **random_state**: integer seed or None.

The fitted estimator exposes n_features_in_, n_samples_fit_, max_features_,
feature_importances_, node_count_, n_leaves_, and max_depth_.
The apply method returns the terminal-node index for each row.

## Random forest

~~~python
from random_forest_regressor import RandomForestRegressor

forest = RandomForestRegressor(
    n_estimators=100,
    max_features=1.0,
    bootstrap=True,
    oob_score=True,
    random_state=42,
)
forest.fit(x_train, y_train)
predictions = forest.predict(x_test)
print("test R-squared:", forest.score(x_test, y_test))
print("out-of-bag R-squared:", forest.oob_score_)
~~~

Each tree uses a bootstrap sample by default. max_samples=None draws the
training-set size with replacement; an integer selects that many rows; a
fraction in (0, 1] selects max(1, floor(fraction * n_samples)) rows.
Set bootstrap=False to train every tree on the complete training set.
max_features is sampled independently at each node, with the same accepted
forms as DecisionTreeRegressor.

When oob_score=True, only trees that omitted a row from their bootstrap
sample contribute to that row's oob_prediction_. The default OOB score is
R-squared over rows that received at least one OOB prediction. Rows without
votes contain NaN and are excluded. If fewer than two rows receive OOB
predictions, oob_score_ is NaN. OOB evaluation is a useful training-time
diagnostic, not a replacement for an untouched test set.

The fitted forest exposes estimators_, estimators_samples_,
feature_importances_, feature_importances_std_, n_features_in_,
n_samples_fit_, and max_samples_. The apply method returns a sample-major
matrix of leaf indices, with one column per tree. oob_counts_ and
oob_coverage_ describe OOB support when OOB scoring is enabled.

predict_quantiles returns requested empirical quantiles across individual
tree predictions for each row. predict_std returns the population standard
deviation across tree predictions. These diagnostics describe ensemble
disagreement; they are not calibrated predictive intervals or guaranteed
coverage bounds. They can be useful for identifying rows where fitted trees
disagree, but should not be presented as formal uncertainty estimates.

## Limitations

Both estimators accept finite real-valued features and targets. Missing-value
handling should be performed in a training-fitted pipeline, for example with
SimpleImputer. Multi-output regression, sample weights, categorical features,
parallel fitting, warm starts, pruning, and alternative split criteria are not
implemented. Forest fitting is sequential.

For established API conventions and the statistical definition of a random
forest regressor, see the
[DecisionTreeRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
and
[RandomForestRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

## Gradient boosting

~~~python
from gradient_boosting_regressor import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    subsample=0.8,
    max_depth=2,
    random_state=42,
).fit(x_train, y_train)
predictions = model.predict(x_test)
print("R-squared:", model.score(x_test, y_test))
print("feature importance:", model.feature_importances_)
~~~

The model starts with the mean training target, then fits one regression tree
per stage to the current residuals. Each tree contribution is multiplied by
learning_rate before it is added to the running prediction. The implementation
uses squared-error loss only. Targets are scaled internally to keep residual
calculation stable for large finite values.

Set subsample below 1 to fit each stage on a reproducible sample drawn without
replacement. The fitted estimator exposes estimators_, estimators_samples_,
init_prediction_, feature_importances_, n_features_in_, and n_samples_fit_.
staged_predict yields predictions after each stage, which can be used to
inspect validation performance as the ensemble grows. apply returns a
sample-major matrix with one terminal-node index per stage.

This educational implementation does not include alternative losses,
sample weights, early stopping, warm starts, parallel fitting, or missing-value
handling. Fit any imputation or scaling steps only on training data, preferably
inside a pipeline.
