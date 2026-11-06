"""Tests for additional backtracking algorithms."""

import unittest

from backtracking import solve_n_queens


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


if __name__ == "__main__":
    unittest.main()
