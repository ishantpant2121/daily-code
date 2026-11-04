# Daily Code Lab

A growing collection of practical Python exercises and small developer/data tools.
The repository contains foundational DSA examples and **CSV Inspector**, a
CSV profiler that uses only the Python standard library.

## CSV Inspector

CSV Inspector reads a CSV file one record at a time and reports row/column counts,
delimiter, inferred scalar types, missing-value counts, exact distinct counts,
numeric ranges, and structural problems such as duplicate headers or inconsistent
row widths. It does not upload files or make network requests.

### Run from the repository

~~~bash
python -m csv_inspector examples/sales.csv
python -m csv_inspector examples/sales.csv --json
python -m csv_inspector examples/sales.csv --strict
python -m csv_inspector examples/malformed.csv --strict  # exits 1 when issues are found
~~~

### Install as a command

~~~bash
python -m pip install .
csv-inspector examples/sales.csv
~~~

### Options

- **--delimiter CHAR**: override automatic delimiter detection.
- **--encoding NAME**: choose input encoding (default **utf-8-sig**, including BOM support).
- **--json**: emit a machine-readable JSON report.
- **--no-distinct**: avoid retaining unique values in memory.
- **--max-issues N**: cap stored diagnostics (default 100); the total issue count remains available.
- **--strict**: exit with status 1 when structural issues are found.

Exact distinct counts use memory proportional to the number of unique non-empty
values. Use **--no-distinct** for high-cardinality or very large files. Rows are
streamed and issue storage is capped. The first record is treated as the header.
Malformed rows are reported rather than silently repaired. Missing markers are
empty strings, **NA**, **N/A**, **null**, **none**, and **nan** (case-insensitive after trim).

## DSA examples

- **day1.py**: hash map, two pointers, and sliding window.
- **day2.py**: prefix sums, binary search, and stack.
- **day3_linked_lists.py**: typed linked-list reversal, Floyd cycle detection, and cycle-safe conversion.

## Tests

No third-party test runner is needed:

~~~bash
python -m unittest discover -s tests -v
~~~

Python 3.10 or newer is required.
