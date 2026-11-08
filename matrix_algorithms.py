"""Basic matrix transformations and multiplication using Python sequences."""

import math
from numbers import Real
from typing import Sequence, TypeVar

T = TypeVar("T")


def _rectangular(matrix: Sequence[Sequence[T]], *, allow_empty: bool = True) -> tuple[int, int]:
    if not matrix:
        if allow_empty:
            return 0, 0
        raise ValueError("matrix must not be empty")
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError("matrix rows must have equal lengths")
    return len(matrix), columns


def _numeric_matrix(matrix: Sequence[Sequence[Real]], name: str) -> list[list[float]]:
    _rectangular(matrix, allow_empty=False)
    result = []
    for row in matrix:
        converted = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must contain real numbers")
            try:
                number = float(value)
            except (OverflowError, ValueError):
                raise ValueError(f"{name} must contain finite real numbers") from None
            if not math.isfinite(number):
                raise ValueError(f"{name} must contain finite real numbers")
            converted.append(number)
        result.append(converted)
    return result


def transpose(matrix: Sequence[Sequence[T]]) -> list[list[T]]:
    """Return a transposed copy of a rectangular matrix."""
    rows, columns = _rectangular(matrix)
    return [[matrix[row][column] for row in range(rows)] for column in range(columns)]


def matrix_multiply(
    first: Sequence[Sequence[Real]],
    second: Sequence[Sequence[Real]],
) -> list[list[float]]:
    """Multiply two compatible rectangular matrices of finite real numbers."""
    left = _numeric_matrix(first, "first matrix")
    right = _numeric_matrix(second, "second matrix")
    first_rows, first_columns = _rectangular(left, allow_empty=False)
    second_rows, second_columns = _rectangular(right, allow_empty=False)
    if first_columns != second_rows:
        raise ValueError("matrix dimensions are not compatible for multiplication")
    result = []
    for row in range(first_rows):
        output_row = []
        for column in range(second_columns):
            try:
                value = math.fsum(
                    left[row][index] * right[index][column]
                    for index in range(first_columns)
                )
            except OverflowError:
                raise ValueError("matrix product exceeds finite float range") from None
            if not math.isfinite(value):
                raise ValueError("matrix product exceeds finite float range")
            output_row.append(value)
        result.append(output_row)
    return result


def spiral_order(matrix: Sequence[Sequence[T]]) -> list[T]:
    """Return matrix elements in clockwise spiral order."""
    rows, columns = _rectangular(matrix)
    if rows == 0 or columns == 0:
        return []
    top, bottom, left, right = 0, rows - 1, 0, columns - 1
    result: list[T] = []
    while top <= bottom and left <= right:
        result.extend(matrix[top][column] for column in range(left, right + 1))
        top += 1
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        if top <= bottom:
            result.extend(matrix[bottom][column] for column in range(right, left - 1, -1))
            bottom -= 1
        if left <= right:
            result.extend(matrix[row][left] for row in range(bottom, top - 1, -1))
            left += 1
    return result


def rotate_clockwise(matrix: Sequence[Sequence[T]]) -> list[list[T]]:
    """Return a 90-degree clockwise rotation without modifying the input."""
    rows, columns = _rectangular(matrix)
    return [
        [matrix[row][column] for row in range(rows - 1, -1, -1)]
        for column in range(columns)
    ]
