"""Demonstrate fixed, temporal, and reusable validation folds."""

from advanced_splitting import PredefinedSplit, TimeSeriesSplit
from decision_tree_regressor import DecisionTreeRegressor
from model_selection import cross_val_score

features = [[float(index)] for index in range(12)]
targets = [0.5 * index + (1 if index % 3 == 0 else 0) for index in range(12)]

fixed = PredefinedSplit([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
fixed_scores = cross_val_score(
    DecisionTreeRegressor(max_depth=2),
    features,
    targets,
    cv=fixed,
    scoring="r2",
)
print("Predefined folds:", fixed_scores)

time_folds = TimeSeriesSplit(n_splits=3, test_size=2, gap=1).split(len(features))
time_scores = cross_val_score(
    DecisionTreeRegressor(max_depth=2),
    features,
    targets,
    cv=time_folds,
    scoring="r2",
)
print("Temporal folds with a one-row gap:", time_scores)
