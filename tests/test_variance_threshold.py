"""Tests for variance-based feature selection."""

import unittest

from preprocessing import VarianceThreshold


class VarianceThresholdTests(unittest.TestCase):
    def test_removes_constant_and_low_variance_features(self):
        values = [[1, 5, 0], [2, 5, 0.1], [3, 5, 0.2]]
        selector = VarianceThreshold().fit(values)
        self.assertEqual(selector.get_support(), [True, False, True])
        self.assertEqual(selector.get_support(indices=True), [0, 2])
        self.assertEqual(selector.transform(values), [[1.0, 0.0], [2.0, 0.1], [3.0, 0.2]])

    def test_threshold_is_applied_to_population_variance(self):
        selector = VarianceThreshold(threshold=0.5).fit([[0], [1], [2]])
        self.assertEqual(selector.get_support(), [True])
        with self.assertRaises(ValueError):
            VarianceThreshold(threshold=1.0).fit([[0], [1], [2]])

    def test_requires_fit_and_matching_feature_count(self):
        selector = VarianceThreshold()
        with self.assertRaises(RuntimeError):
            selector.transform([[1]])
        selector.fit([[1, 2], [2, 2]])
        with self.assertRaises(ValueError):
            selector.transform([[1]])

    def test_rejects_invalid_thresholds_and_all_constant_features(self):
        for threshold in (-1, float("nan"), float("inf"), True, "0"):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                VarianceThreshold(threshold)
        with self.assertRaises(ValueError):
            VarianceThreshold().fit([[1, 2], [1, 2]])


if __name__ == "__main__":
    unittest.main()
