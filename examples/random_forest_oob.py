"""Demonstrate random-forest out-of-bag evaluation."""

from random_forest import RandomForestClassifier

features = [
    [0.0, 0.1], [0.2, 0.0], [0.1, 0.3], [0.3, 0.2], [0.4, 0.1],
    [4.8, 5.0], [5.0, 4.9], [5.2, 5.1], [4.9, 5.2], [5.1, 4.8],
]
labels = ["near"] * 5 + ["far"] * 5

forest = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    oob_score=True,
    random_state=42,
).fit(features, labels)

covered = sum(count > 0 for count in forest.oob_counts_)
print(f"Out-of-bag accuracy: {forest.oob_score_:.3f}")
print(f"Rows with at least one OOB vote: {covered}/{len(features)}")
print("Per-row OOB vote counts:", forest.oob_counts_)
print("Hard class votes for first row:", forest.predict_votes(features[:1])[0])
print("Per-tree feature-importance standard deviations:", forest.feature_importances_std_)
