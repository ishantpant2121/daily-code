# Daily Code Lab

A growing collection of practical Python exercises, data-science utilities, and
small developer tools. The repository contains foundational DSA examples and
**CSV Inspector**, a CSV profiler that uses only the Python standard library.

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
python -m csv_inspector examples/malformed.csv --strict
python -m csv_inspector examples/duplicates.csv --duplicates --missing-value unknown
~~~

### Install as a command

~~~bash
python -m pip install .
csv-inspector examples/sales.csv
~~~

The installation also exposes the top-level learning modules for imports from
outside the repository.

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
Rows are streamed and issue storage is capped. The first record is treated as the
header. Malformed rows are reported rather than silently repaired. Missing markers
are empty strings, **NA**, **N/A**, **null**, **none**, and **nan** (case-insensitive
after trim).

## DSA examples

- **day1.py**: hash map, two pointers, and sliding window.
- **day2.py**: prefix sums, binary search, and stack.
- **day3_linked_lists.py**: typed linked-list reversal, Floyd cycle detection, and cycle-safe conversion.

## Algorithm library

The repository includes focused, dependency-free implementations:

- **Sorting** — merge sort, three-way quick sort, counting sort (including
  one-shot iterables), heap sort, quickselect, and merge-based inversion counting.
- **Searching** — linear search, lower bound, duplicate-aware rotated-array
  search, and first/last position queries for repeated values.
- **Graphs** — BFS, DFS, Dijkstra distances and path reconstruction, Bellman-Ford,
  Floyd-Warshall, topological ordering, connected components, iterative strongly
  connected components, and minimum spanning trees with Kruskal and Prim.
- **Pathfinding** — A* shortest-path search with optional heuristic and
  non-negative weighted edges.
- **Dynamic programming** — climbing stairs, coin change, longest increasing
  subsequence, 0/1 knapsack, LCS length and sequence reconstruction, and edit distance.
- **Backtracking** — subsets, permutations, N-Queens, and combination sum.
- **Greedy algorithms** — activity selection, fractional knapsack, and interval merging.
- **Trees, heaps, and union-find** — binary-tree traversals, heap-based
  selection and merging, and disjoint-set operations.
- **Strings and matrices** — KMP and Rabin-Karp matching, prefix and Z functions,
  finite-real matrix multiplication, transpose, clockwise rotation, and spiral traversal.

Graph algorithms that operate on undirected graphs expect symmetric adjacency
lists. Dijkstra requires finite non-negative real weights; Bellman-Ford and
Floyd-Warshall support finite negative edges and detect reachable or global
negative cycles respectively. MST helpers support finite negative weights.

## Data science and machine learning utilities

- **classification_metrics.py** — accuracy, precision, recall, F1, balanced
  accuracy, Matthews correlation, confusion matrices, ROC/precision-recall
  curves, average precision, Brier score, and binary log loss.
- **regression_metrics.py** — absolute, squared, logarithmic, median, and maximum
  errors, explained variance, and R-squared with an explicit constant-target policy.
- **data_splitting.py** — reproducible train/test splits, stratified holdout
  splits, balanced K-fold indices, and stratified K-fold indices.
- **preprocessing.py** — MinMaxScaler, StandardScaler, RobustScaler, polynomial
  expansion, variance-based feature selection, one-hot encoding, and label encoding.
- **knn_classifier.py** and **knn_regressor.py** — dependency-free nearest-neighbor
  classification and regression with Euclidean/Manhattan distance and optional
  inverse-distance weighting.
- **logistic_regression.py** — binary logistic regression with batch gradient
  descent, stable sigmoid probabilities, and optional L2 regularization.
- **gaussian_nb.py** — Gaussian Naive Bayes with variance smoothing and normalized
  log-probability calculations.
- **kmeans.py** — k-means clustering with k-means++ initialization, multiple
  starts, deterministic seeds, and empty-cluster handling.
- **pca.py** — principal component analysis using a symmetric Jacobi eigensolver.
- **linear_regression.py** — simple univariate ordinary least-squares regression.
- **statistics_utils.py** — mean, median, interpolated quantiles, sample/population
  variance, standard deviation, covariance, and Pearson correlation.

Example classification workflow:

~~~python
from data_splitting import stratified_train_test_split
from knn_classifier import KNNClassifier
from preprocessing import StandardScaler

features = [[0.0, 1.0], [0.2, 0.8], [4.8, 5.0], [5.2, 4.9]]
labels = ["near", "near", "far", "far"]
x_train, x_test, y_train, y_test = stratified_train_test_split(
    features, labels, labels=labels, test_size=0.5, random_state=42
)

scaler = StandardScaler().fit(x_train)
model = KNNClassifier(n_neighbors=1).fit(scaler.transform(x_train), y_train)
predictions = model.predict(scaler.transform(x_test))
print(predictions)
~~~

Additional unsupervised and dimensionality-reduction example:

~~~python
from kmeans import KMeans
from pca import PCA
from preprocessing import StandardScaler

features = [[1.0, 2.0], [1.2, 1.8], [8.0, 9.0], [8.2, 8.8]]
scaled = StandardScaler().fit_transform(features)
projection = PCA(n_components=1).fit_transform(scaled)
clusters = KMeans(n_clusters=2, random_state=42).fit_predict(projection)
print(clusters)
~~~

Fit preprocessing steps on training data only, then apply the fitted transform
to validation or test data. This avoids leaking information from held-out samples
into the training process. These utilities are educational, small-data tools and
do not replace the broader functionality or numerical performance of dedicated
scientific-computing libraries.

## Tests

No third-party test runner is needed:

~~~bash
python -m unittest discover -s tests -v
~~~

Python 3.10 or newer is required.
