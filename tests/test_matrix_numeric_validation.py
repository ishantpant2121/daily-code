"""Tests for matrix multiplication numeric validation."""

import unittest
from matrix_algorithms import matrix_multiply


class MatrixMultiplicationValidationTests(unittest.TestCase):
    def test_returns_float_products_for_integer_inputs(self):
        self.assertEqual(matrix_multiply([[1, 2]], [[3], [4]]), [[11.0]])

    def test_rejects_boolean_non_numeric_and_non_finite_values(self):
        invalid = (True, "2", float("nan"), float("inf"), 10 ** 10000)
        for index, value in enumerate(invalid):
            with self.subTest(index=index, value_type=type(value).__name__):
                with self.assertRaises(ValueError):
                    matrix_multiply([[value]], [[1]])


if __name__ == "__main__":
    unittest.main()
