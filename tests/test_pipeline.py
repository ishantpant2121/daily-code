"""Tests for leakage-safe estimator pipelines."""

import unittest

from imputation import SimpleImputer
from logistic_regression import LogisticRegression
from pipeline import Pipeline
from preprocessing import StandardScaler


class PipelineTests(unittest.TestCase):
    def test_composes_imputation_scaling_and_classification(self):
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ])
        features = [[-3.0], [-2.0], [None], [2.0], [3.0]]
        labels = ["low", "low", "low", "high", "high"]
        pipeline.fit(features, labels)
        self.assertEqual(pipeline.predict([[-2.5], [2.5]]), ["low", "high"])
        self.assertEqual(pipeline.named_steps["imputer"].statistics_, (0.0,))
        probabilities = pipeline.predict_proba([[-1.0], [1.0]])
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-12 for row in probabilities))

    def test_fit_uses_only_training_rows_for_transform_statistics(self):
        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=100)),
        ])
        pipeline.fit([[0.0], [2.0], [1000.0]], [0, 0, 1])
        self.assertAlmostEqual(pipeline.named_steps["scale"].mean_[0], 334.0)

    def test_accepts_one_shot_feature_and_label_iterables(self):
        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=100)),
        ])
        pipeline.fit(
            ([value] for value in [-2, -1, 1, 2]),
            (label for label in ["a", "a", "b", "b"]),
        )
        self.assertEqual(pipeline.predict([[2]]), ["b"])

    def test_rejects_invalid_steps_and_duplicate_names(self):
        for steps in (
            [],
            [("bad", object())],
            [("same", StandardScaler()), ("same", LogisticRegression())],
            [("bad__name", StandardScaler())],
            [("classifier", object())],
        ):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                Pipeline(steps)

    def test_requires_fit_and_checks_input_shapes(self):
        pipeline = Pipeline([("classifier", LogisticRegression(max_iter=10))])
        with self.assertRaises(RuntimeError):
            pipeline.predict([[1]])
        with self.assertRaises(ValueError):
            pipeline.fit([[1], [2]], [0])
        pipeline.fit([[0], [1]], [0, 1])
        with self.assertRaises(ValueError):
            pipeline.predict([[1, 2]])

    def test_delegates_score_and_reports_missing_optional_methods(self):
        pipeline = Pipeline([("classifier", LogisticRegression(max_iter=100))])
        with self.assertRaises(RuntimeError):
            pipeline.score([[0]], [0])
        pipeline.fit([[-1], [1]], [0, 1])
        self.assertEqual(pipeline.score([[-1], [1]], [0, 1]), 1.0)


if __name__ == "__main__":
    unittest.main()
