"""Deterministic parameter-grid utilities for small educational searches."""

from collections.abc import Mapping, Sequence
from itertools import product
from math import prod
from typing import Any, Iterator


class ParameterGrid:
    """Enumerate a finite Cartesian product of estimator parameters.

    Keys are sorted to make iteration deterministic regardless of mapping
    insertion order. An empty mapping represents one candidate with no
    parameters; an empty value sequence represents an empty grid.
    """

    def __init__(self, param_grid: Mapping[str, Sequence[Any]]) -> None:
        if not isinstance(param_grid, Mapping):
            raise ValueError("param_grid must be a mapping of parameter names to sequences")
        normalized: dict[str, tuple[Any, ...]] = {}
        for name, values in param_grid.items():
            if not isinstance(name, str) or not name:
                raise ValueError("parameter names must be non-empty strings")
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise ValueError(f"values for parameter {name!r} must be a non-string sequence")
            normalized[name] = tuple(values)
        self._grid = normalized
        self._keys = tuple(sorted(normalized))

    def __len__(self) -> int:
        if not self._keys:
            return 1
        return prod(len(self._grid[key]) for key in self._keys)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self._keys:
            yield {}
            return
        value_sequences = [self._grid[key] for key in self._keys]
        for combination in product(*value_sequences):
            yield dict(zip(self._keys, combination))

    def __getitem__(self, index: int) -> dict[str, Any]:
        if type(index) is not int:
            raise TypeError("grid indices must be integers")
        size = len(self)
        if index < 0:
            index += size
        if not 0 <= index < size:
            raise IndexError("parameter grid index out of range")
        for position, candidate in enumerate(self):
            if position == index:
                return candidate
        raise IndexError("parameter grid index out of range")
