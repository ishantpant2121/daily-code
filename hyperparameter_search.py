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


class ParameterSampler:
    """Sample distinct candidates from a finite parameter grid reproducibly.

    This deliberately supports finite sequences only. Sampling is without
    replacement, so n_iter cannot exceed the number of grid combinations.
    """

    def __init__(
        self,
        param_distributions: Mapping[str, Sequence[Any]],
        n_iter: int,
        *,
        random_state: int | None = None,
    ) -> None:
        import random

        if type(n_iter) is not int or n_iter < 1:
            raise ValueError("n_iter must be a positive integer")
        if random_state is not None and type(random_state) is not int:
            raise ValueError("random_state must be an integer or None")
        candidates = list(ParameterGrid(param_distributions))
        if n_iter > len(candidates):
            raise ValueError("n_iter cannot exceed the number of finite parameter combinations")
        self._sampled = random.Random(random_state).sample(candidates, n_iter)

    def __len__(self) -> int:
        return len(self._sampled)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter([candidate.copy() for candidate in self._sampled])
