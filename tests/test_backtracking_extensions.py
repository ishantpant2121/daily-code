"""Tests for additional backtracking algorithms."""

import unittest

from backtracking import combination_sum, solve_n_queens


class NQueensTests(unittest.TestCase):
    def test_known_solution_counts(self):
        self.assertEqual(len(solve_n_queens(4)), 2)
        self.assertEqual(len(solve_n_queens(5)), 10)

    def test_each_solution_has_unique_columns_and_diagonals(self):
        for solution in solve_n_queens(4):
            self.assertEqual(len(set(solution)), 4)
            self.assertEqual(
                len({row - column for row, column in enumerate(solution)}), 4
            )
            self.assertEqual(
                len({row + column for row, column in enumerate(solution)}), 4
            )

    def test_handles_small_and_empty_boards(self):
        self.assertEqual(solve_n_queens(0), [()])
        self.assertEqual(solve_n_queens(1), [(0,)])
        self.assertEqual(solve_n_queens(2), [])
        self.assertEqual(solve_n_queens(3), [])

    def test_rejects_negative_or_non_integer_sizes(self):
        for size in (-1, 2.5):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    solve_n_queens(size)


class CombinationSumTests(unittest.TestCase):
    def test_returns_unique_combinations_with_reuse(self):
        result = combination_sum([2, 3, 6, 7], 7)
        self.assertEqual(result, [[2, 2, 3], [7]])

    def test_deduplicates_candidates_and_handles_zero_target(self):
        self.assertEqual(combination_sum([2, 3, 2], 7), [[2, 2, 3]])
        self.assertEqual(combination_sum([1, 2], 0), [[]])

    def test_returns_empty_when_no_combination_exists(self):
        self.assertEqual(combination_sum([4, 6], 5), [])

    def test_rejects_invalid_candidates_and_targets(self):
        with self.assertRaises(ValueError):
            combination_sum([0, 1], 3)
        with self.assertRaises(ValueError):
            combination_sum([1, -2], 3)
        with self.assertRaises(ValueError):
            combination_sum([1, 2], -1)


if __name__ == "__main__":
    unittest.main()
