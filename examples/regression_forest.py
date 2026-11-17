"""Fit a random-forest regressor and inspect held-out and OOB scores."""

from random_forest_regressor import RandomForestRegressor

features = [[float(index), float(index % 4)] for index in range(24)]
targets = [
    1.8 * row[0] - 0.7 * row[1] + (row[0] % 5) * 0.4
    for row in features
]

x_train, y_train = features[:18], targets[:18]
x_test, y_test = features[18:], targets[18:]

forest = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    max_features="sqrt",
    oob_score=True,
    random_state=42,
).fit(x_train, y_train)

print("predictions:", forest.predict(x_test))
print("test R-squared:", forest.score(x_test, y_test))
print("OOB R-squared:", forest.oob_score_)
print("OOB coverage:", forest.oob_coverage_)
print("feature importances:", forest.feature_importances_)
