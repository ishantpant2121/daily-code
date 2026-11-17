"""Compare a gradient-boosted tree model with a constant baseline."""

from gradient_boosting_regressor import GradientBoostingRegressor
from regression_metrics import r2_score

features = [[float(index)] for index in range(40)]
targets = [
    0.35 * index + (5.0 if index >= 20 else 0.0)
    for index in range(40)
]

# Interleave held-out rows so evaluation covers the full feature range.
x_train, y_train = features[::2], targets[::2]
x_test, y_test = features[1::2], targets[1::2]

model = GradientBoostingRegressor(
    n_estimators=80,
    learning_rate=0.08,
    subsample=0.85,
    max_depth=2,
    random_state=42,
).fit(x_train, y_train)

print("initial prediction:", model.init_prediction_)
print("test R-squared:", model.score(x_test, y_test))
print("feature importances:", model.feature_importances_)

# Staged predictions let the caller inspect generalization as trees are added.
stage_scores = [
    r2_score(y_test, predictions)
    for predictions in model.staged_predict(x_test)
]
best_stage = max(range(len(stage_scores)), key=stage_scores.__getitem__) + 1
print("best observed stage:", best_stage)
print("best observed validation R-squared:", stage_scores[best_stage - 1])
