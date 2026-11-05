"""Command-line interface and report formatting for CSV Inspector."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

from .models import CSVProfile
from .profiler import CSVInspectionError, profile_csv


def render_text(profile: CSVProfile) -> str:
    headings = ("COLUMN", "TYPE", "NULLS", "DISTINCT", "MIN", "MAX", "EXAMPLES")
    rows = [
        (column.name, column.inferred_type, str(column.null_count),
         "disabled" if column.distinct_count is None else str(column.distinct_count),
         column.min_value or "—", column.max_value or "—",
         ", ".join(json.dumps(value, ensure_ascii=False) for value in column.examples) or "—")
        for column in profile.columns
    ]
    widths = [max(len(headings[i]), *(len(row[i]) for row in rows)) for i in range(len(headings))]

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(widths[i]) for i, value in enumerate(row))

    lines = [
        f"CSV profile: {profile.path}",
        f"Rows: {profile.row_count} | Columns: {profile.column_count} | Delimiter: {profile.delimiter!r}",
        f"Duplicate rows: {profile.duplicate_row_count if profile.duplicate_row_count is not None else 'not tracked'}",
        "", format_row(headings), format_row(tuple("-" * width for width in widths)),
    ]
    lines.extend(format_row(row) for row in rows)
    lines.extend(("", f"Issues: {profile.issue_count}"))
    if profile.issues:
        lines.extend(f"- [{issue.code}] {issue.message}" for issue in profile.issues)
    elif not profile.has_issues:
        lines.append("none")
    if profile.issues_truncated:
        lines.append(f"- ... {profile.issues_truncated} additional issue(s) omitted")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="csv-inspector",
        description="Profile a CSV file using only the Python standard library.",
    )
    parser.add_argument("path", type=Path, help="CSV file to inspect")
    parser.add_argument("--delimiter", help="one-character separator (auto-detected by default)")
    parser.add_argument("--encoding", default="utf-8-sig", help="input encoding (default: utf-8-sig)")
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    parser.add_argument("--no-distinct", action="store_true", help="skip exact distinct counts to reduce memory")
    parser.add_argument("--duplicates", action="store_true", help="count exact duplicate data rows (uses memory proportional to unique rows)")
    parser.add_argument("--missing-value", action="append", default=[], metavar="VALUE", help="additional value treated as missing; may be repeated")
    parser.add_argument("--max-issues", type=int, default=100, help="maximum diagnostics retained (default: 100)")
    parser.add_argument("--strict", action="store_true", help="return status 1 when structural issues are found")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = profile_csv(
            args.path, delimiter=args.delimiter, encoding=args.encoding,
            track_distinct=not args.no_distinct, max_issues=args.max_issues,
            track_duplicates=args.duplicates, missing_values=tuple(args.missing_value),
        )
    except (CSVInspectionError, OSError, UnicodeError, csv.Error, ValueError) as exc:
        print(f"csv-inspector: error: {exc}", file=sys.stderr)
        return 2
    output = json.dumps(profile.to_dict(), indent=2, ensure_ascii=False) if args.json else render_text(profile)
    print(output)
    return 1 if args.strict and profile.has_issues else 0
