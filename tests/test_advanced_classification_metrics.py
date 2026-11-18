"""Regression tests for advanced classification metrics."""

import unittest

from classification_metrics import (
    classification_metrics,
    classification_report,
    cohen_kappa_score,
    fbeta_score,
    hamming_loss,
    multilabel_confusion_matrix,
    multiclass_average_precision_score,
    multiclass_roc_auc_score,
    top_k_accuracy_score,
)


class TopKAccuracyTests(unittest.TestCase):
    def test_top_k_accuracy_and_deterministic_ties(self):
        actual = ["b", "a"]
        probabilities = [[0.5, 0.5], [0.6, 0.4]]
        self.assertEqual(
            top_k_accuracy_score(actual, probabilities, labels=["a", "b"], k=1),
            0.5,
        )
        self.assertEqual(
            top_k_accuracy_score(actual, probabilities, labels=["a", "b"], k=2),
            1.0,
        )

    def test_rejects_invalid_k(self):
        for k in (0, 3, True, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                top_k_accuracy_score(
                    ["a"], [[0.7, 0.3]], labels=["a", "b"], k=k
                )


class FBetaTests(unittest.TestCase):
    def test_fbeta_matches_f1_and_weights_recall(self):
        actual = [1, 1, 1, 1, 0]
        predicted = [1, 0, 0, 0, 1]
        self.assertAlmostEqual(
            fbeta_score(actual, predicted, beta=1),
            classification_metrics(actual, predicted)["f1"],
        )
        self.assertAlmostEqual(fbeta_score(actual, predicted, beta=2), 5 / 18)
        self.assertAlmostEqual(fbeta_score(actual, predicted, beta=0.5), 5 / 24)

    def test_large_beta_is_numerically_stable(self):
        self.assertAlmostEqual(
            fbeta_score([1, 1, 1, 1], [1, 0, 0, 0], beta=1e308),
            0.25,
        )

    def test_zero_division_and_invalid_beta(self):
        self.assertEqual(fbeta_score([0, 0], [0, 0], zero_division=1), 1.0)
        for beta in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                fbeta_score([0, 1], [0, 1], beta=beta)


class ClassificationReportTests(unittest.TestCase):
    def test_per_class_and_aggregate_metrics(self):
        report = classification_report(
            ["cat", "cat", "dog", "dog"],
            ["cat", "dog", "dog", "bird"],
            labels=["cat", "dog", "bird"],
        )
        self.assertAlmostEqual(report["accuracy"], 0.5)
        self.assertAlmostEqual(report["per_class"]["cat"]["precision"], 1.0)
        self.assertAlmostEqual(report["per_class"]["cat"]["recall"], 0.5)
        self.assertEqual(report["per_class"]["bird"]["support"], 0)
        self.assertAlmostEqual(report["macro_avg"]["f1"], 7 / 18)
        self.assertAlmostEqual(report["weighted_avg"]["f1"], 7 / 12)
        self.assertAlmostEqual(report["micro_avg"]["f1"], 0.5)

    def test_infers_class_order_and_validates_inputs(self):
        report = classification_report(["b", "a"], ["a", "a"])
        self.assertEqual(report["labels"], ("b", "a"))
        with self.assertRaises(ValueError):
            classification_report(["a"], ["a"], labels=["b"])
        with self.assertRaises(ValueError):
            classification_report(["a"], ["a"], zero_division=2)


class CohenKappaTests(unittest.TestCase):
    def test_unweighted_and_weighted_kappa(self):
        actual = [0, 0, 1, 1]
        predicted = [0, 1, 1, 1]
        self.assertAlmostEqual(cohen_kappa_score(actual, predicted), 0.5)
        self.assertAlmostEqual(
            cohen_kappa_score(actual, predicted, weights="linear"), 0.5
        )
        self.assertAlmostEqual(
            cohen_kappa_score(actual, predicted, weights="quadratic"), 0.5
        )

    def test_rejects_invalid_weights_and_degenerate_cases(self):
        with self.assertRaises(ValueError):
            cohen_kappa_score([0, 1], [0, 1], weights="exponential")
        with self.assertRaises(ValueError):
            cohen_kappa_score([1, 1], [1, 1])


class MultilabelMetricTests(unittest.TestCase):
    def setUp(self):
        self.actual = [[1, 0], [0, 1], [1, 1]]
        self.predicted = [[1, 1], [0, 0], [1, 0]]

    def test_per_label_confusion_matrices(self):
        self.assertEqual(
            multilabel_confusion_matrix(self.actual, self.predicted),
            [
                [[1, 0], [0, 2]],
                [[0, 1], [2, 0]],
            ],
        )

    def test_hamming_loss_counts_mismatched_indicators(self):
        self.assertEqual(hamming_loss(self.actual, self.predicted), 0.5)

    def test_rejects_invalid_indicator_matrices(self):
        invalid = (
            ([[1, 0]], []),
            ([[1, 0]], [[1]]),
            ([[1, 0]], [[1, 2]]),
            ([[1, 0], [0]], [[1, 0], [0, 1]]),
            ([[]], [[]]),
        )
        for actual, predicted in invalid:
            with self.subTest(actual=actual, predicted=predicted):
                with self.assertRaises(ValueError):
                    hamming_loss(actual, predicted)


class MulticlassRankingMetricTests(unittest.TestCase):
    def setUp(self):
        self.labels = ["a", "b", "c"]
        self.actual = ["a", "b", "c", "a", "b", "c"]
        self.probabilities = [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]

    def test_roc_auc_supports_macro_weighted_and_micro(self):
        for average in ("macro", "weighted", "micro"):
            with self.subTest(average=average):
                self.assertAlmostEqual(
                    multiclass_roc_auc_score(
                        self.actual, self.probabilities,
                        labels=self.labels, average=average,
                    ),
                    1.0,
                )

    def test_average_precision_supports_macro_weighted_and_micro(self):
        for average in ("macro", "weighted", "micro"):
            with self.subTest(average=average):
                self.assertAlmostEqual(
                    multiclass_average_precision_score(
                        self.actual, self.probabilities,
                        labels=self.labels, average=average,
                    ),
                    1.0,
                )

    def test_rejects_unknown_average_and_missing_class_support(self):
        with self.assertRaises(ValueError):
            multiclass_roc_auc_score(
                self.actual, self.probabilities,
                labels=self.labels, average="none",
            )
        with self.assertRaises(ValueError):
            multiclass_roc_auc_score(
                ["a", "a"], [[0.9, 0.1], [0.8, 0.2]],
                labels=["a", "b"],
            )


if __name__ == "__main__":
    unittest.main()
