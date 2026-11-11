"""Inspect a fitted model with curves and permutation importance."""

from decision_tree import DecisionTreeClassifier
from data_splitting import train_test_split
from feature_importance import permutation_importance
from model_diagnostics import bootstrap_score, learning_curve, validation_curve

features = [[value, value % 3] for value in range(18)]
labels = [int(value >= 9) for value in range(18)]
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=21
)

model = DecisionTreeClassifier(max_depth=2).fit(x_train, y_train)
importance = permutation_importance(
    model, x_test, y_test, scoring="accuracy", n_repeats=20, random_state=21
)
interval = bootstrap_score(
    model, x_test, y_test, scoring="accuracy", n_bootstrap=200, random_state=21
)
curve = learning_curve(
    DecisionTreeClassifier(max_depth=2),
    x_train,
    y_train,
    train_sizes=[0.5, 1.0],
    cv=3,
    scoring="accuracy",
    stratified=True,
    shuffle=True,
    random_state=21,
)
validation = validation_curve(
    DecisionTreeClassifier(),
    x_train,
    y_train,
    "max_depth",
    [1, 2, 3],
    cv=3,
    scoring="accuracy",
    stratified=True,
)

print("Permutation importances:", importance.importances_mean)
print("Bootstrap interval:", interval.lower_bound, interval.upper_bound)
print("Learning-curve test scores:", curve["test_scores"])
print("Validation-curve test scores:", validation["test_scores"])
