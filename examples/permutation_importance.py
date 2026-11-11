"""Rank model features by their held-out permutation score drops."""

from data_splitting import train_test_split
from decision_tree import DecisionTreeClassifier
from feature_importance import permutation_importance

features = [[float(value), float(value % 4)] for value in range(32)]
labels = [int(value >= 16) for value in range(32)]
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=11
)
model = DecisionTreeClassifier(max_depth=2).fit(x_train, y_train)
result = permutation_importance(
    model, x_test, y_test, scoring="accuracy", n_repeats=30, random_state=11
)
for index, (mean, deviation) in enumerate(zip(result.importances_mean, result.importances_std)):
    print(f"feature {index}: score drop {mean:.3f} +/- {deviation:.3f}")
