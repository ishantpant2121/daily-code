"""Scalar type inference and per-column aggregation."""

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Iterable

from .models import ColumnProfile

DEFAULT_MISSING_VALUES = frozenset({"", "na", "n/a", "null", "none", "nan"})


def classify_value(value: str) -> str:
    """Infer a scalar kind without changing the value stored in the source."""
    candidate = value.strip()
    lowered = candidate.casefold()
    if lowered in {"true", "false"}:
        return "boolean"
    try:
        date.fromisoformat(candidate)
        return "date"
    except ValueError:
        # Python 3.11 added compact YYYYMMDD support to fromisoformat().
        # Parse this basic ISO date explicitly so Python 3.10 behaves the same.
        if len(candidate) == 8 and candidate.isascii() and candidate.isdigit():
            try:
                date(
                    int(candidate[:4]),
                    int(candidate[4:6]),
                    int(candidate[6:]),
                )
            except ValueError:
                pass
            else:
                return "date"
    try:
        datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        if "T" in candidate or " " in candidate:
            return "datetime"
    except ValueError:
        pass
    try:
        int(candidate)
        return "integer"
    except ValueError:
        pass
    try:
        number = Decimal(candidate)
        if number.is_finite():
            return "number"
    except (InvalidOperation, ValueError):
        pass
    return "string"


def resolve_type(kinds: Iterable[str]) -> str:
    observed = set(kinds)
    if not observed:
        return "empty"
    if observed <= {"integer", "number"}:
        return "integer" if observed == {"integer"} else "number"
    return next(iter(observed)) if len(observed) == 1 else "mixed"


class ColumnAccumulator:
    """Accumulate exact counts, numeric bounds, and bounded examples."""

    def __init__(
        self,
        track_distinct: bool = True,
        max_examples: int = 5,
        missing_values: Iterable[str] = DEFAULT_MISSING_VALUES,
    ) -> None:
        if type(max_examples) is not int or max_examples < 0:
            raise ValueError("max_examples must be a non-negative integer")
        self.track_distinct = track_distinct
        self.null_count = 0
        self.kinds: set[str] = set()
        self.distinct: set[str] | None = set() if track_distinct else None
        self.examples: list[str] = []
        self.numeric_min: Decimal | None = None
        self.numeric_max: Decimal | None = None
        self.all_numeric = True
        self.seen_non_null = False
        self.max_examples = max_examples
        self.missing_values = frozenset(
            value.strip().casefold() for value in missing_values
        )

    def add(self, value: str) -> None:
        normalized = value.strip()
        if normalized.casefold() in self.missing_values:
            self.null_count += 1
            return
        self.seen_non_null = True
        kind = classify_value(normalized)
        self.kinds.add(kind)
        if self.distinct is not None:
            self.distinct.add(normalized)
        if len(self.examples) < self.max_examples and normalized not in self.examples:
            self.examples.append(normalized)
        if kind in {"integer", "number"}:
            number = Decimal(normalized)
            self.numeric_min = (
                number if self.numeric_min is None else min(self.numeric_min, number)
            )
            self.numeric_max = (
                number if self.numeric_max is None else max(self.numeric_max, number)
            )
        else:
            self.all_numeric = False

    def snapshot(self, name: str, position: int) -> ColumnProfile:
        numeric = self.seen_non_null and self.all_numeric
        return ColumnProfile(
            name=name,
            position=position,
            inferred_type=resolve_type(self.kinds),
            null_count=self.null_count,
            distinct_count=len(self.distinct) if self.distinct is not None else None,
            min_value=(
                str(self.numeric_min)
                if numeric and self.numeric_min is not None
                else None
            ),
            max_value=(
                str(self.numeric_max)
                if numeric and self.numeric_max is not None
                else None
            ),
            examples=tuple(self.examples),
        )
