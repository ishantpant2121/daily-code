"""Fit a random-forest regressor and inspect held-out and OOB scores."""

from random_forest_regressor import RandomForestRegressor

features = [[float(index)] for index in range(36)]
targets = [
    0.5 * index + (4.0 if (index // 8) % 2 else 0.0)
    for index in range(36)
]

# Interleave held-out rows so the test set stays inside the training range.
x_train, y_train = features[::2], targets[::2]
x_test, y_test = features[1::2], targets[1::2]

forest = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    max_features=1.0,
    oob_score=True,
    random_state=42,
).fit(x_train, y_train)

print("predictions:", forest.predict(x_test))
print("test R-squared:", forest.score(x_test, y_test))
print("OOB R-squared:", forest.oob_score_)
print("OOB coverage:", forest.oob_coverage_)
print("feature importances:", forest.feature_importances_)
