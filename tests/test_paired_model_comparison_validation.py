"""Validation tests for paired model-comparison utilities."""

import unittest

from paired_model_comparison import (
    paired_bootstrap_score_difference,
    paired_permutation_test,
)


class PairedModelComparisonValidationTests(unittest.TestCase):
    def setUp(self):
        self.actual = [0, 1, 0, 1]
        self.first = [0, 1, 0, 1]
        self.second = [1, 0, 1, 0]

    def test_rejects_empty_or_mismatched_inputs(self):
        invalid = (
            ([], [], []),
            ([1, 2], [1], [1, 2]),
            ([1, 2], [1, 2], [1]),
        )
        for actual, first, second in invalid:
            with self.subTest(actual=actual, first=first, second=second):
                with self.assertRaises(ValueError):
                    paired_bootstrap_score_difference(actual, first, second)
                with self.assertRaises(ValueError):
                    paired_permutation_test(actual, first, second)

    def test_rejects_string_inputs(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_score_difference("labels", [1, 2], [1, 2])
        with self.assertRaises(ValueError):
            paired_permutation_test([0, 1], "predictions", [0, 1])

    def test_rejects_invalid_bootstrap_parameters(self):
        invalid = (
            {"n_bootstrap": 0},
            {"n_bootstrap": -1},
            {"n_bootstrap": True},
            {"confidence_level": 0.0},
            {"confidence_level": 1.0},
            {"confidence_level": float("nan")},
            {"confidence_level": True},
            {"random_state": True},
            {"random_state": 1.5},
            {"scoring": "not-a-scorer"},
            {"scoring": None},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    paired_bootstrap_score_difference(
                        self.actual, self.first, self.second, **kwargs
                    )

    def test_rejects_invalid_permutation_parameters(self):
        invalid = (
            {"n_permutations": 0},
            {"n_permutations": -2},
            {"n_permutations": True},
            {"random_state": True},
            {"random_state": 3.5},
            {"scoring": "not-a-scorer"},
            {"scoring": None},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    paired_permutation_test(
                        self.actual, self.first, self.second, **kwargs
                    )

    def test_custom_scorer_must_return_finite_real_value(self):
        invalid_scores = (True, "0.5", None, float("nan"), float("inf"))
        for value in invalid_scores:
            with self.subTest(value=value):
                scorer = lambda y, p, result=value: result
                with self.assertRaises(ValueError):
                    paired_bootstrap_score_difference(
                        self.actual, self.first, self.second,
                        scoring=scorer, n_bootstrap=3,
                    )
                with self.assertRaises(ValueError):
                    paired_permutation_test(
                        self.actual, self.first, self.second,
                        scoring=scorer, n_permutations=3,
                    )

    def test_custom_scorer_exceptions_are_not_silently_swallowed(self):
        def broken_scorer(actual, predicted):
            raise RuntimeError("scorer failure")
        with self.assertRaisesRegex(RuntimeError, "scorer failure"):
            paired_bootstrap_score_difference(
                self.actual, self.first, self.second, scoring=broken_scorer
            )


if __name__ == "__main__":
    unittest.main()
