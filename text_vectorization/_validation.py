"""Shared input validation for dependency-free text vectorizers."""

import math
import re
from numbers import Real
from typing import Iterable

DEFAULT_TOKEN_PATTERN = r"(?u)\b\w\w+\b"


def validate_documents(documents: Iterable[str]) -> list[str]:
    """Materialize a document iterable and reject ambiguous or invalid inputs."""
    if isinstance(documents, str):
        raise ValueError("documents must be an iterable of strings, not one string")
    try:
        result = list(documents)
    except TypeError:
        raise ValueError("documents must be an iterable of strings") from None
    if any(not isinstance(document, str) for document in result):
        raise ValueError("every document must be a string")
    return result


def validate_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def validate_ngram_range(value: object) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise ValueError("ngram_range must be a pair of integers")
    minimum, maximum = value
    if minimum < 1 or maximum < minimum:
        raise ValueError("ngram_range must satisfy 1 <= min_n <= max_n")
    return minimum, maximum


def validate_df_threshold(value: object, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a non-negative integer or a fraction in [0, 1]")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"fractional {name} must be in [0, 1]")
    return number


def validate_max_features(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError("max_features must be a positive integer or None")
    return value


def compile_token_pattern(value: object) -> re.Pattern[str]:
    if not isinstance(value, str) or not value:
        raise ValueError("token_pattern must be a non-empty regular-expression string")
    try:
        pattern = re.compile(value)
    except re.error as error:
        raise ValueError(f"invalid token_pattern: {error}") from None
    if pattern.groups > 1:
        raise ValueError("token_pattern may contain at most one capturing group")
    if pattern.search("") is not None:
        raise ValueError("token_pattern must not match an empty string")
    return pattern


def validate_stop_words(value: object, *, lowercase: bool) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        raise ValueError("stop_words must be an iterable of words, not a string")
    try:
        words = list(value)
    except TypeError:
        raise ValueError("stop_words must be an iterable of strings or None") from None
    if any(not isinstance(word, str) or not word for word in words):
        raise ValueError("stop_words must contain non-empty strings")
    return frozenset(word.lower() if lowercase else word for word in words)


def validate_feature_matrix(matrix: Iterable[Iterable[Real]], width: int) -> list[list[float]]:
    """Validate a dense count or TF-IDF matrix before inverse transformation."""
    if isinstance(matrix, (str, bytes)):
        raise ValueError("matrix must be an iterable of numeric rows")
    try:
        rows = list(matrix)
    except TypeError:
        raise ValueError("matrix must be an iterable of numeric rows") from None

    validated: list[list[float]] = []
    for row in rows:
        if isinstance(row, (str, bytes)):
            raise ValueError("matrix rows must be numeric iterables")
        try:
            values = list(row)
        except TypeError:
            raise ValueError("matrix rows must be numeric iterables") from None
        if len(values) != width:
            raise ValueError(f"each matrix row must contain exactly {width} values")
        converted: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("matrix values must be finite non-negative real numbers")
            try:
                number = float(value)
            except (OverflowError, TypeError, ValueError):
                raise ValueError("matrix values must be finite non-negative real numbers") from None
            if not math.isfinite(number) or number < 0.0:
                raise ValueError("matrix values must be finite non-negative real numbers")
            converted.append(number)
        validated.append(converted)
    return validated
