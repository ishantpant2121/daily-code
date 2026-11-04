"""Immutable result models for CSV profiling."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RowIssue:
    code: str
    row_number: int | None
    message: str
    expected_fields: int | None = None
    actual_fields: int | None = None


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    name: str
    position: int
    inferred_type: str
    null_count: int
    distinct_count: int | None
    min_value: str | None
    max_value: str | None
    examples: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CSVProfile:
    path: str
    delimiter: str
    encoding: str
    row_count: int
    column_count: int
    columns: tuple[ColumnProfile, ...]
    issues: tuple[RowIssue, ...]
    issue_count: int
    issues_truncated: int

    @property
    def has_issues(self) -> bool:
        return self.issue_count > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
