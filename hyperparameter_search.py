"""Deterministic parameter-grid utilities for small educational searches."""

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
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
        if not self._keys:
            return {}

        # itertools.product varies the rightmost key fastest. Mixed-radix
        # decomposition preserves that order without scanning earlier items.
        selected = {}
        for key in reversed(self._keys):
            values = self._grid[key]
            index, offset = divmod(index, len(values))
            selected[key] = values[offset]
        return {key: selected[key] for key in self._keys}


class ParameterSampler:
    """Sample distinct candidates from a finite parameter grid reproducibly.

    This deliberately supports finite sequences only. Sampling is without
    replacement, so n_iter cannot exceed the number of grid combinations.
    Only sampled integer positions are retained; the Cartesian product is
    not materialized.
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
        grid = ParameterGrid(param_distributions)
        grid_size = len(grid)
        if n_iter > grid_size:
            raise ValueError("n_iter cannot exceed the number of finite parameter combinations")
        self._grid = grid
        self._sampled_indices = tuple(
            random.Random(random_state).sample(range(grid_size), n_iter)
        )

    def __len__(self) -> int:
        return len(self._sampled_indices)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in self._sampled_indices:
            yield self._grid[index]


def _set_estimator_params(estimator: Any, parameters: Mapping[str, Any]) -> None:
    """Set direct or pipeline-step parameters on an estimator copy."""
    for name, value in parameters.items():
        if "__" in name:
            step_name, attribute = name.split("__", 1)
            named_steps = getattr(estimator, "named_steps", None)
            if not isinstance(named_steps, Mapping) or step_name not in named_steps:
                raise ValueError(f"unknown pipeline step in parameter {name!r}")
            target = named_steps[step_name]
        else:
            attribute = name
            target = estimator
        if not attribute or not hasattr(target, attribute):
            raise ValueError(f"unknown estimator parameter {name!r}")
        setattr(target, attribute, value)


class GridSearchCV:
    """Exhaustively select estimator parameters using cross-validation.

    Each candidate and each fold receives a fresh deep copy. The winning
    estimator is then fitted once on the complete input dataset. Parameter
    names may address direct estimator attributes or Pipeline steps using
    the step__attribute naming convention.
    """

    def __init__(
        self,
        estimator: Any,
        param_grid: Mapping[str, Sequence[Any]],
        cv: int = 5,
        *,
        scoring: Any = None,
        stratified: bool = False,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        if not callable(getattr(estimator, "fit", None)):
            raise ValueError("estimator must implement fit")
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

    def _fit_candidates(
        self,
        features: list[list[Any]],
        labels: list[Any],
        candidates: list[dict[str, Any]],
    ) -> "GridSearchCV":
        import math
        from model_selection import cross_val_score

        if not candidates:
            raise ValueError("parameter search requires at least one candidate")
        rows = []
        best_index = 0
        best_score = -math.inf
        for index, parameters in enumerate(candidates):
            candidate = deepcopy(self.estimator)
            _set_estimator_params(candidate, parameters)
            scores = cross_val_score(
                candidate,
                features,
                labels,
                self.cv,
                scoring=self.scoring,
                stratified=self.stratified,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
            mean_score, std_score = _mean_and_std(scores)
            row = {"params": parameters.copy(), "mean_test_score": mean_score,
                   "std_test_score": std_score}
            row.update({f"split{fold}_test_score": score for fold, score in enumerate(scores)})
            rows.append(row)
            if mean_score > best_score:
                best_index = index
                best_score = mean_score

        self.cv_results_ = {
            key: [row[key] for row in rows]
            for key in rows[0]
        }
        self.best_index_ = best_index
        self.best_params_ = candidates[best_index].copy()
        self.best_score_ = best_score
        self.best_estimator_ = deepcopy(self.estimator)
        _set_estimator_params(self.best_estimator_, self.best_params_)
        self.best_estimator_.fit(features, labels)
        self.n_splits_ = self.cv
        self._fitted = True
        return self

    def fit(self, features: Iterable[Iterable[Any]], labels: Iterable[Any]) -> "GridSearchCV":
        from model_selection import _feature_rows

        matrix = _feature_rows(features)
        target = list(labels)
        if len(matrix) != len(target):
            raise ValueError("features and labels must have equal sample counts")
        candidates = list(ParameterGrid(self.param_grid))
        return self._fit_candidates(matrix, target, candidates)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before using the search result")

    def predict(self, features: Iterable[Iterable[Any]]) -> list[Any]:
        self._check_fitted()
        return self.best_estimator_.predict(features)

    def predict_proba(self, features: Iterable[Iterable[Any]]) -> list[list[float]]:
        self._check_fitted()
        method = getattr(self.best_estimator_, "predict_proba", None)
        if not callable(method):
            raise AttributeError("best estimator does not implement predict_proba")
        return method(features)

    def score(self, features: Iterable[Iterable[Any]], labels: Iterable[Any]) -> float:
        self._check_fitted()
        method = getattr(self.best_estimator_, "score", None)
        if not callable(method):
            raise AttributeError("best estimator does not implement score")
        return method(features, labels)


class RandomizedSearchCV(GridSearchCV):
    """Evaluate a reproducible random subset of finite parameter combinations."""

    def __init__(
        self,
        estimator: Any,
        param_distributions: Mapping[str, Sequence[Any]],
        n_iter: int = 10,
        cv: int = 5,
        *,
        scoring: Any = None,
        stratified: bool = False,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            estimator,
            param_distributions,
            cv,
            scoring=scoring,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
        )
        if type(n_iter) is not int or n_iter < 1:
            raise ValueError("n_iter must be a positive integer")
        if n_iter > len(ParameterGrid(param_distributions)):
            raise ValueError("n_iter cannot exceed the number of finite parameter combinations")
        self.n_iter = n_iter

    def fit(self, features: Iterable[Iterable[Any]], labels: Iterable[Any]) -> "RandomizedSearchCV":
        from model_selection import _feature_rows

        matrix = _feature_rows(features)
        target = list(labels)
        if len(matrix) != len(target):
            raise ValueError("features and labels must have equal sample counts")
        candidates = list(
            ParameterSampler(self.param_grid, self.n_iter, random_state=self.random_state)
        )
        self.sampled_params_ = [candidate.copy() for candidate in candidates]
        self._fit_candidates(matrix, target, candidates)
        return self
