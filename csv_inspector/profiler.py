"""CSV file reading and structural diagnostics."""

import csv
from collections.abc import Iterable
from pathlib import Path

from .inference import DEFAULT_MISSING_VALUES, ColumnAccumulator
from .models import CSVProfile, RowIssue


class CSVInspectionError(ValueError):
    """Raised when a file cannot be interpreted as a header-based CSV."""


def _choose_delimiter(sample: str, requested: str | None) -> str:
    if requested is not None:
        if not isinstance(requested, str) or len(requested) != 1:
            raise CSVInspectionError("delimiter must be exactly one character")
        if requested in {"\r", "\n", '"'}:
            raise CSVInspectionError(
                "delimiter cannot be a newline or the CSV quote character"
            )
        return requested
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _normalize_missing_values(values: Iterable[str]) -> frozenset[str]:
    if isinstance(values, str):
        raise ValueError("missing_values must be an iterable of strings, not a string")
    try:
        iterator = iter(values)
    except TypeError:
        raise ValueError("missing_values must be an iterable of strings") from None
    normalized = set()
    for value in iterator:
        if not isinstance(value, str):
            raise ValueError("missing_values must contain only strings")
        normalized.add(value.strip().casefold())
    return frozenset(normalized)


def profile_csv(
    path: str | Path,
    *,
    delimiter: str | None = None,
    encoding: str = "utf-8-sig",
    track_distinct: bool = True,
    max_issues: int = 100,
    max_examples: int = 5,
    track_duplicates: bool = False,
    missing_values: Iterable[str] = (),
) -> CSVProfile:
    """Profile a CSV one record at a time and retain bounded diagnostics.

    Exact distinct counts use memory proportional to unique non-empty values.
    Set track_distinct=False to avoid that allocation. At most max_examples
    distinct non-missing examples are retained per column; use zero to disable
    examples. Duplicate detection is opt-in and retains unique row tuples in
    memory. Additional missing markers are case-insensitive and are combined
    with the built-in defaults. Only the first max_issues diagnostics are
    retained, while the total is preserved.
    """
    if type(max_issues) is not int or max_issues < 1:
        raise ValueError("max_issues must be a positive integer")
    if type(max_examples) is not int or max_examples < 0:
        raise ValueError("max_examples must be a non-negative integer")
    if type(track_distinct) is not bool:
        raise ValueError("track_distinct must be a boolean")
    if type(track_duplicates) is not bool:
        raise ValueError("track_duplicates must be a boolean")
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError("encoding must be a non-empty string")
    missing_markers = DEFAULT_MISSING_VALUES | _normalize_missing_values(missing_values)

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {source}")
    with source.open("r", encoding=encoding, newline="") as handle:
        sample = handle.read(65536)
    if not sample:
        raise CSVInspectionError("CSV file is empty")
    selected_delimiter = _choose_delimiter(sample, delimiter)
    issues: list[RowIssue] = []
    issue_count = 0

    seen_rows: set[tuple[str, ...]] | None = set() if track_duplicates else None
    duplicate_row_count = 0

    def add_issue(issue: RowIssue) -> None:
        nonlocal issue_count
        issue_count += 1
        if len(issues) < max_issues:
            issues.append(issue)

    with source.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter=selected_delimiter, strict=True)
        try:
            headers = next(reader)
        except StopIteration as exc:
            raise CSVInspectionError("CSV file has no header row") from exc
        except csv.Error as exc:
            raise CSVInspectionError(
                f"malformed CSV near physical line {reader.line_num}: {exc}"
            ) from exc
        if not headers:
            raise CSVInspectionError("CSV header row is blank")
        for index, header in enumerate(headers, start=1):
            if not header.strip():
                add_issue(RowIssue("blank_header", 1, f"Header at column {index} is blank"))
        seen_headers: set[str] = set()
        for index, header in enumerate(headers, start=1):
            if header in seen_headers:
                add_issue(RowIssue("duplicate_header", 1, f"Header {header!r} is duplicated at column {index}"))
            seen_headers.add(header)
        accumulators = [
            ColumnAccumulator(
                track_distinct, max_examples=max_examples, missing_values=missing_markers
            )
            for _ in headers
        ]
        row_count = 0
        try:
            for row in reader:
                row_count += 1
                record_number = row_count + 1
                if seen_rows is not None:
                    row_key = tuple(row)
                    if row_key in seen_rows:
                        duplicate_row_count += 1
                    else:
                        seen_rows.add(row_key)
                if len(row) != len(headers):
                    add_issue(RowIssue(
                        "row_width_mismatch", record_number,
                        f"Record {record_number} has {len(row)} fields; expected {len(headers)}",
                        len(headers), len(row),
                    ))
                for index, accumulator in enumerate(accumulators):
                    accumulator.add(row[index] if index < len(row) else "")
        except csv.Error as exc:
            raise CSVInspectionError(
                f"malformed CSV near physical line {reader.line_num}: {exc}"
            ) from exc

    columns = tuple(
        accumulator.snapshot(name, index)
        for index, (name, accumulator) in enumerate(zip(headers, accumulators), start=1)
    )
    return CSVProfile(
        path=str(source), delimiter=selected_delimiter, encoding=encoding,
        row_count=row_count, column_count=len(headers), columns=columns,
        issues=tuple(issues), issue_count=issue_count,
        issues_truncated=max(0, issue_count - len(issues)),
        duplicate_row_count=duplicate_row_count if seen_rows is not None else None,
    )
