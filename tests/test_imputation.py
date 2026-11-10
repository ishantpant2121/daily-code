"""Tests for dependency-free missing-value imputation."""

import unittest

from imputation import SimpleImputer


class SimpleImputerTests(unittest.TestCase):
    def test_mean_and_median_impute_numeric_columns(self):
        values = [[1.0, 10.0], [None, 20.0], [5.0, float("nan")]]
        mean = SimpleImputer().fit(values)
        self.assertEqual(mean.transform(values), [[1.0, 10.0], [3.0, 20.0], [5.0, 15.0]])
        median = SimpleImputer(strategy="median").fit(values)
        self.assertEqual(median.transform(values), [[1.0, 10.0], [3.0, 20.0], [5.0, 15.0]])

    def test_supports_custom_missing_sentinel(self):
        imputer = SimpleImputer(strategy="mean", missing_values=-1)
        self.assertEqual(
            imputer.fit_transform([[1, -1], [3, 5]]),
            [[1, 5.0], [3, 5]],
        )

    def test_most_frequent_uses_first_value_to_break_ties(self):
        values = [["red", "small"], ["blue", "large"], [None, "small"]]
        imputer = SimpleImputer(strategy="most_frequent").fit(values)
        self.assertEqual(imputer.transform(values)[2], ["red", "small"])

    def test_constant_strategy_retains_empty_columns(self):
        imputer = SimpleImputer(strategy="constant", fill_value="unknown")
        self.assertEqual(
            imputer.fit_transform([[None, 1], [None, 2]]),
            [["unknown", 1], ["unknown", 2]],
        )

    def test_empty_statistical_columns_are_dropped_or_retained(self):
        values = [[1, None], [2, None]]
        dropped = SimpleImputer().fit(values)
        self.assertEqual(dropped.transform(values), [[1.0], [2.0]])
        self.assertEqual(dropped.get_support(indices=True), [0])
        kept = SimpleImputer(keep_empty_features=True).fit(values)
        self.assertEqual(kept.transform(values), [[1.0, 0.0], [2.0, 0.0]])

    def test_supports_one_shot_iterables_and_checks_fitted_width(self):
        imputer = SimpleImputer()
        result = imputer.fit_transform((row for row in [[1], [None], [3]]))
        self.assertEqual(result, [[1], [2.0], [3]])
        with self.assertRaises(ValueError):
            imputer.transform([[1, 2]])

    def test_rejects_invalid_strategies_and_non_numeric_statistics(self):
        with self.assertRaises(ValueError):
            SimpleImputer(strategy="unknown")
        with self.assertRaises(ValueError):
            SimpleImputer(keep_empty_features=1)
        for strategy in ("mean", "median"):
            with self.subTest(strategy=strategy), self.assertRaises(ValueError):
                SimpleImputer(strategy=strategy).fit([["a"], [None]])
        with self.assertRaises(ValueError):
            SimpleImputer(strategy="most_frequent").fit([[[1]], [[2]]])

    def test_requires_fit_and_handles_all_dropped_columns(self):
        with self.assertRaises(RuntimeError):
            SimpleImputer().transform([[1]])
        imputer = SimpleImputer().fit([[None], [float("nan")]])
        self.assertEqual(imputer.transform([[None], [2.0]]), [[], []])


if __name__ == "__main__":
    unittest.main()
