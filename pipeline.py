"""Composable, dependency-free preprocessing and estimator pipelines."""

from typing import Any, Iterable


def _matrix(values: Iterable[Iterable[Any]]) -> list[list[Any]]:
    try:
        rows = [list(row) for row in values]
    except TypeError:
        raise ValueError("features must be an iterable of rows") from None
    if not rows or not rows[0]:
        raise ValueError("features must contain at least one row and one column")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all feature rows must have equal lengths")
    return rows


class Pipeline:
    """Apply fitted transformers in order, then delegate to a final estimator.

    Each intermediate step must implement fit(X) and transform(X), preserving
    the number of samples. The final step must implement fit(X, y) or fit(X),
    plus predict(X). A pipeline fit learns every transform only from the data
    passed to fit, so held-out rows do not contribute to learned statistics.
    """

    def __init__(self, steps: Iterable[tuple[str, Any]]) -> None:
        materialized = list(steps)
        if not materialized:
            raise ValueError("steps must contain at least one transformer or estimator")
        names = []
        normalized = []
        for item in materialized:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("each step must be a (name, object) pair")
            name, step = item
            if not isinstance(name, str) or not name or "__" in name:
                raise ValueError("step names must be non-empty strings without double underscores")
            if name in names:
                raise ValueError("step names must be unique")
            if step is None:
                raise ValueError("pipeline steps cannot be None")
            names.append(name)
            normalized.append((name, step))

        for name, step in normalized[:-1]:
            if not callable(getattr(step, "fit", None)) or not callable(
                getattr(step, "transform", None)
            ):
                raise ValueError(f"intermediate step {name!r} must implement fit and transform")
        final_name, final_step = normalized[-1]
        if not callable(getattr(final_step, "fit", None)) or not callable(
            getattr(final_step, "predict", None)
        ):
            raise ValueError(f"final step {final_name!r} must implement fit and predict")

        self.steps = tuple(normalized)
        self._fitted = False

    @property
    def named_steps(self) -> dict[str, Any]:
        return dict(self.steps)

    def fit(
        self,
        features: Iterable[Iterable[Any]],
        labels: Iterable[Any] | None = None,
    ) -> "Pipeline":
        rows = _matrix(features)
        target = None if labels is None else list(labels)
        if target is not None and len(target) != len(rows):
            raise ValueError("features and labels must have equal sample counts")

        self._fitted = False
        transformed = rows
        for _, transformer in self.steps[:-1]:
            transformer.fit(transformed)
            next_transformed = _matrix(transformer.transform(transformed))
            if len(next_transformed) != len(transformed):
                raise ValueError("pipeline transformers must preserve sample count")
            transformed = next_transformed

        estimator = self.steps[-1][1]
        if target is None:
            estimator.fit(transformed)
        else:
            estimator.fit(transformed, target)
        self.n_features_in_ = len(rows[0])
        self._fitted = True
        return self

    def predict(self, features: Iterable[Iterable[Any]]) -> list[Any]:
        self._check_fitted()
        transformed = self._transform_features(features)
        return self.steps[-1][1].predict(transformed)

    def predict_proba(self, features: Iterable[Iterable[Any]]) -> list[list[float]]:
        self._check_fitted()
        estimator = self.steps[-1][1]
        method = getattr(estimator, "predict_proba", None)
        if not callable(method):
            raise AttributeError("final estimator does not implement predict_proba")
        return method(self._transform_features(features))

    def score(
        self,
        features: Iterable[Iterable[Any]],
        labels: Iterable[Any],
    ) -> float:
        self._check_fitted()
        estimator = self.steps[-1][1]
        method = getattr(estimator, "score", None)
        if not callable(method):
            raise AttributeError("final estimator does not implement score")
        return method(self._transform_features(features), labels)

    def _transform_features(
        self,
        features: Iterable[Iterable[Any]],
    ) -> list[list[Any]]:
        transformed = _matrix(features)
        if len(transformed[0]) != self.n_features_in_:
            raise ValueError("input feature count does not match fitted data")
        for _, transformer in self.steps[:-1]:
            next_transformed = _matrix(transformer.transform(transformed))
            if len(next_transformed) != len(transformed):
                raise ValueError("pipeline transformers must preserve sample count")
            transformed = next_transformed
        return transformed

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before prediction")
