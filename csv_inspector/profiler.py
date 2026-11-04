"""CSV file reading and structural diagnostics."""

import csv
from pathlib import Path

from .inference import ColumnAccumulator
from .models import CSVProfile, RowIssue


class CSVInspectionError(ValueError):
    """Raised when a file cannot be interpreted as a header-based CSV."""


def _choose_delimiter(sample: str, requested: str | None) -> str:
    if requested is not None:
        if len(requested) != 1:
            raise CSVInspectionError("delimiter must be exactly one character")
        return requested
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def profile_csv(
    path: str | Path,
    *,
    delimiter: str | None = None,
    encoding: str = "utf-8-sig",
    track_distinct: bool = True,
    max_issues: int = 100,
) -> CSVProfile:
    """Profile a CSV one record at a time and retain bounded diagnostics.

    Exact distinct counts use memory proportional to unique non-empty values.
    Set track_distinct=False to avoid that allocation. Only the first
    max_issues diagnostics are retained, while the total count is preserved.
    """
    if max_issues < 1:
        raise ValueError("max_issues must be at least 1")
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
        accumulators = [ColumnAccumulator(track_distinct) for _ in headers]
        row_count = 0
        try:
            for row in reader:
                row_count += 1
                record_number = row_count + 1
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
    )
