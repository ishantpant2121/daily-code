"""Estimate a fitted classifier's held-out score uncertainty by resampling."""

from data_splitting import train_test_split
from decision_tree import DecisionTreeClassifier
from model_diagnostics import bootstrap_score

features = [[float(value), float(value % 5)] for value in range(40)]
labels = [int(value >= 20) for value in range(40)]
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=19
)
model = DecisionTreeClassifier(max_depth=2).fit(x_train, y_train)
result = bootstrap_score(
    model, x_test, y_test, scoring="accuracy", n_bootstrap=500,
    confidence_level=0.95, random_state=19,
)
print("Mean held-out score:", result.mean_score)
print("95% percentile interval:", result.lower_bound, result.upper_bound)
print("This interval conditions on the fitted estimator; it does not refit the model.")
