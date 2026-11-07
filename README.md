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
python -m csv_inspector examples/duplicates.csv --duplicates --missing-value unknown
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
- **--duplicates**: count exact repeated data rows (disabled by default; retains unique row tuples in memory).
- **--missing-value VALUE**: add a case-insensitive missing-value marker; repeat the option for multiple markers.
- **--max-issues N**: cap stored diagnostics (default 100); the total issue count remains available.
- **--strict**: exit with status 1 when structural issues are found.

Exact distinct counts use memory proportional to the number of unique non-empty
values. Use **--no-distinct** for high-cardinality or very large files. Exact
duplicate-row detection is opt-in because it retains unique row tuples in memory;
use it when the file size and row cardinality are appropriate. Additional missing
markers are combined with the built-in markers and compared case-insensitively.
Rows are streamed and issue storage is capped. The first record is treated as the header.
Malformed rows are reported rather than silently repaired. Missing markers are
empty strings, **NA**, **N/A**, **null**, **none**, and **nan** (case-insensitive after trim).

## DSA examples

- **day1.py**: hash map, two pointers, and sliding window.
- **day2.py**: prefix sums, binary search, and stack.
- **day3_linked_lists.py**: typed linked-list reversal, Floyd cycle detection, and cycle-safe conversion.

## Algorithm library

The repository also includes focused, dependency-free implementations:

- **Sorting** — merge sort, three-way quick sort, counting sort, heap sort,
  quickselect, and merge-based inversion counting.
- **Searching** — linear search, lower bound, duplicate-aware rotated-array
  search, and first/last position queries for repeated values.
- **Graphs** — BFS, DFS, Dijkstra distances and path reconstruction,
  topological ordering, connected components, and minimum spanning trees
  with Kruskal's and Prim's algorithms.
- **Dynamic programming** — climbing stairs, coin change, longest increasing
  subsequence, 0/1 knapsack, longest common subsequence length, and edit distance.
- **Backtracking** — subsets, permutations, N-Queens, and combination sum.
- **Greedy algorithms** — activity selection, fractional knapsack, and interval
  merging.
- **Trees, heaps, and union-find** — binary-tree traversals, heap-based
  selection and merging, and disjoint-set operations.

Each algorithm is implemented in its own focused module and covered by
standard-library unittest regression tests. Graph algorithms that operate on
undirected graphs expect symmetric adjacency lists; the MST helpers document
their respective edge-list and adjacency-list input formats.

## Data science utility

- **classification_metrics.py** — dependency-free accuracy, precision, recall,
  F1, and confusion counts for binary classification or one-vs-rest evaluation.
  Inputs may be iterables; the zero-division policy is explicit.

## Tests

No third-party test runner is needed:

~~~bash
python -m unittest discover -s tests -v
~~~

Python 3.10 or newer is required.
