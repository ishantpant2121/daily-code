"""Tests for matrix transformations and multiplication."""

import unittest

from matrix_algorithms import matrix_multiply, rotate_clockwise, spiral_order, transpose


class MatrixAlgorithmsTests(unittest.TestCase):
    def test_transposes_rectangular_and_empty_matrices(self):
        self.assertEqual(transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]])
        self.assertEqual(transpose([]), [])
        self.assertEqual(transpose([[], []]), [])

    def test_multiplies_compatible_matrices(self):
        self.assertEqual(
            matrix_multiply([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]),
            [[58, 64], [139, 154]],
        )
        self.assertEqual(matrix_multiply([[2]], [[3]]), [[6]])

    def test_returns_clockwise_spiral_order(self):
        self.assertEqual(
            spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            [1, 2, 3, 6, 9, 8, 7, 4, 5],
        )
        self.assertEqual(spiral_order([[1, 2, 3, 4]]), [1, 2, 3, 4])
        self.assertEqual(spiral_order([[1], [2], [3]]), [1, 2, 3])
        self.assertEqual(spiral_order([]), [])

    def test_rotates_rectangular_matrices(self):
        self.assertEqual(
            rotate_clockwise([[1, 2, 3], [4, 5, 6]]),
            [[4, 1], [5, 2], [6, 3]],
        )
        self.assertEqual(rotate_clockwise([]), [])

    def test_rejects_ragged_and_incompatible_matrices(self):
        with self.assertRaises(ValueError):
            transpose([[1, 2], [3]])
        with self.assertRaises(ValueError):
            matrix_multiply([[1, 2]], [[1, 2]])
        with self.assertRaises(ValueError):
            matrix_multiply([], [[1]])
        with self.assertRaises(ValueError):
            rotate_clockwise([[1, 2], [3]])


if __name__ == "__main__":
    unittest.main()
