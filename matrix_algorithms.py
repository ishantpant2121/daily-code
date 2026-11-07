"""Basic matrix transformations and multiplication using Python sequences."""

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


def transpose(matrix: Sequence[Sequence[T]]) -> list[list[T]]:
    """Return a transposed copy of a rectangular matrix."""
    rows, columns = _rectangular(matrix)
    return [[matrix[row][column] for row in range(rows)] for column in range(columns)]


def matrix_multiply(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Multiply two compatible rectangular matrices."""
    first_rows, first_columns = _rectangular(first, allow_empty=False)
    second_rows, second_columns = _rectangular(second, allow_empty=False)
    if first_columns != second_rows:
        raise ValueError("matrix dimensions are not compatible for multiplication")
    return [
        [sum(first[row][k] * second[k][column] for k in range(first_columns)) for column in range(second_columns)]
        for row in range(first_rows)
    ]


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
