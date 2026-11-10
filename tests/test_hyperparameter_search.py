"""Tests for deterministic finite hyperparameter candidate utilities."""

import unittest

from hyperparameter_search import ParameterGrid, ParameterSampler


class ParameterGridTests(unittest.TestCase):
    def test_enumerates_cartesian_product_in_sorted_key_order(self):
        grid = ParameterGrid({"max_depth": [1, 2], "criterion": ["gini", "entropy"]})
        self.assertEqual(len(grid), 4)
        self.assertEqual(
            list(grid),
            [
                {"criterion": "gini", "max_depth": 1},
                {"criterion": "gini", "max_depth": 2},
                {"criterion": "entropy", "max_depth": 1},
                {"criterion": "entropy", "max_depth": 2},
            ],
        )

    def test_empty_mapping_and_empty_values(self):
        self.assertEqual(list(ParameterGrid({})), [{}])
        self.assertEqual(len(ParameterGrid({"depth": []})), 0)
        self.assertEqual(list(ParameterGrid({"depth": []})), [])

    def test_indexing_supports_negative_indices(self):
        grid = ParameterGrid({"depth": [1, 2, 3]})
        self.assertEqual(grid[0], {"depth": 1})
        self.assertEqual(grid[-1], {"depth": 3})
        with self.assertRaises(IndexError):
            _ = grid[3]
        with self.assertRaises(TypeError):
            _ = grid[1.5]


class ParameterSamplerTests(unittest.TestCase):
    def test_sampling_is_reproducible_and_without_replacement(self):
        parameters = {"depth": [1, 2, 3], "criterion": ["gini", "entropy"]}
        first = list(ParameterSampler(parameters, 4, random_state=23))
        second = list(ParameterSampler(parameters, 4, random_state=23))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len({tuple(sorted(item.items())) for item in first}), 4)

    def test_rejects_invalid_sample_counts_and_random_state(self):
        parameters = {"depth": [1, 2]}
        for count in (0, -1, True, 1.5, 3):
            with self.subTest(count=count), self.assertRaises(ValueError):
                ParameterSampler(parameters, count, random_state=1)
        with self.assertRaises(ValueError):
            ParameterSampler(parameters, 1, random_state=True)

    def test_rejects_non_sequence_parameter_values(self):
        with self.assertRaises(ValueError):
            ParameterGrid({"depth": "123"})
        with self.assertRaises(ValueError):
            ParameterGrid({"depth": (value for value in [1, 2])})


if __name__ == "__main__":
    unittest.main()
