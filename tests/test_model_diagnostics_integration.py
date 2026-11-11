import unittest

from decision_tree import DecisionTreeClassifier
from feature_importance import permutation_importance
from model_diagnostics import bootstrap_score, learning_curve, validation_curve
from pipeline import Pipeline
from preprocessing import StandardScaler


class ModelDiagnosticsIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.features = [[-3.0], [-2.5], [-2.0], [-1.5], [-1.0], [-0.5], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0]]
        self.labels = [0] * 6 + [1] * 6

    def test_pipeline_supports_curves_importance_and_bootstrap(self):
        estimator = Pipeline([
            ("scale", StandardScaler()),
            ("tree", DecisionTreeClassifier(max_depth=2)),
        ])
        fitted = estimator.fit(self.features, self.labels)
        importance = permutation_importance(
            fitted, self.features, self.labels, scoring="accuracy", n_repeats=5, random_state=3
        )
        self.assertEqual(len(importance.importances_mean), 1)
        learning = learning_curve(
            estimator, self.features, self.labels, train_sizes=[0.5, 1.0], cv=3,
            scoring="accuracy", stratified=True, shuffle=True, random_state=3,
        )
        self.assertEqual(learning["train_sizes"], [4, 8])
        validation = validation_curve(
            estimator, self.features, self.labels, "tree__max_depth", [1, 2],
            cv=3, scoring="accuracy", stratified=True,
        )
        self.assertEqual(validation["param_range"], [1, 2])
        interval = bootstrap_score(
            fitted, self.features, self.labels, scoring="accuracy", n_bootstrap=30, random_state=3
        )
        self.assertLessEqual(interval.lower_bound, interval.upper_bound)
        self.assertEqual(estimator.named_steps["tree"].max_depth, 2)


if __name__ == "__main__":
    unittest.main()
