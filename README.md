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

- **decision_tree.py** — deterministic CART classification with Gini or entropy splits, leaf probabilities, depth and sample constraints, and feature importances.
- **imputation.py** — SimpleImputer with mean, median, most-frequent, and constant strategies, configurable missing sentinels, and empty-column handling.
- **pipeline.py** — ordered transformer and estimator composition so preprocessing statistics are learned from training data only.
- **model_selection.py** — reproducible cross-validation scoring with optional stratification and fresh estimator copies per fold.

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

Additional model-selection and pipeline example:

~~~python
from decision_tree import DecisionTreeClassifier
from imputation import SimpleImputer
from model_selection import cross_val_score
from pipeline import Pipeline
from preprocessing import StandardScaler

features = [[0.0], [1.0], [None], [8.0], [9.0], [10.0]]
labels = ["near", "near", "near", "far", "far", "far"]
model = Pipeline([
    ("impute", SimpleImputer(strategy="mean")),
    ("scale", StandardScaler()),
    ("tree", DecisionTreeClassifier(max_depth=2)),
])
print(cross_val_score(
    model, features, labels, cv=3, scoring="accuracy", stratified=True
))
~~~

Cross-validation clones the supplied estimator for each fold. When a pipeline is
used, imputation and scaling are fitted independently on each training fold,
then applied to that fold's validation rows. Built-in scorers are accuracy and
balanced accuracy; a custom scorer receives the fitted estimator, validation
features, and validation labels. These small-data helpers are educational and
are not intended to replace optimized model-selection libraries.

## Tests

No third-party test runner is needed:

~~~bash
python -m unittest discover -s tests -v
~~~

Python 3.10 or newer is required.


## Model selection and ensembles

The model-selection helpers also include timed fold diagnostics and out-of-fold
predictions:

~~~python
from decision_tree import DecisionTreeClassifier
from model_selection import cross_validate, cross_val_predict

features = [[0], [1], [2], [8], [9], [10]]
labels = ["low", "low", "low", "high", "high", "high"]
tree = DecisionTreeClassifier(max_depth=2)

diagnostics = cross_validate(
    tree, features, labels, cv=3, stratified=True,
    scoring="accuracy", return_train_score=True,
)
out_of_fold = cross_val_predict(
    tree, features, labels, cv=3, stratified=True,
)
print(diagnostics["test_score"])
print(out_of_fold)
~~~

cross_validate reports fit time, score time, and test scores for every fold;
training scores and fitted fold estimators are optional. cross_val_predict
returns one held-out prediction per input row in original row order. Use these
predictions for diagnostics or model blending, not as a replacement for the
mean of fold-level scores.

Finite hyperparameter grids can be enumerated, sampled, or searched:

~~~python
from decision_tree import DecisionTreeClassifier
from hyperparameter_search import GridSearchCV, RandomizedSearchCV

grid = GridSearchCV(
    DecisionTreeClassifier(),
    {"max_depth": [1, 2, 3], "criterion": ["gini", "entropy"]},
    cv=3,
    scoring="accuracy",
    stratified=True,
).fit(features, labels)
print(grid.best_params_, grid.best_score_)

sampled = RandomizedSearchCV(
    DecisionTreeClassifier(),
    {"max_depth": [1, 2, 3, 4], "criterion": ["gini", "entropy"]},
    n_iter=4,
    cv=3,
    scoring="accuracy",
    stratified=True,
    random_state=42,
).fit(features, labels)
print(sampled.best_params_)
~~~

The search helpers support finite parameter sequences, deterministic sampling,
and direct estimator attributes or pipeline-step parameters such as
tree__max_depth. The best candidate is refitted on the full training dataset.
Randomized sampling selects integer positions from the finite Cartesian product
without constructing every candidate. See the
[hyperparameter search guide](docs/hyperparameter_search.md) for complexity
details and a large-grid example.

A voting ensemble can combine independent classifiers:

~~~python
from decision_tree import DecisionTreeClassifier
from ensemble import VotingClassifier
from gaussian_nb import GaussianNB

voter = VotingClassifier(
    [("tree", DecisionTreeClassifier()), ("bayes", GaussianNB())],
    voting="soft",
    weights=[2, 1],
).fit(features, labels)
print(voter.predict(features))
~~~

Hard voting combines predicted labels. Soft voting averages class-aligned
probabilities and therefore requires every component classifier to implement
predict_proba and expose classes_. Component estimators are copied before
fitting, so the original objects remain unchanged.


## Advanced validation and model evaluation

- **advanced_splitting.py** — chronological `TimeSeriesSplit`, group-disjoint `GroupKFold` and `GroupShuffleSplit`, repeated K-fold and stratified K-fold, stratified randomized holdouts, and leave-one-out indices. Use chronological or group-aware splits when random row-level splitting would leak information.
- **clustering_metrics.py** — per-sample and mean silhouette coefficients, adjusted Rand agreement, and Davies-Bouldin cluster separation.
- **feature_importance.py** — model-agnostic permutation importance for an already-fitted estimator, with repeat-level score drops, means, and standard deviations.
- **model_diagnostics.py** — learning curves, validation curves (including `step__parameter` pipeline parameters), and paired bootstrap confidence intervals for a fitted model's evaluation score.
- **calibration_metrics.py** — binary reliability bins, expected calibration error, and maximum calibration error for probability predictions.

Examples:

~~~bash
python examples/time_series_validation.py
python examples/group_validation.py
python examples/clustering_evaluation.py
python examples/model_inspection.py
python examples/probability_calibration.py
~~~

Focused guides: [advanced validation](docs/advanced_validation.md), [clustering metrics](docs/clustering_metrics.md), [model inspection](docs/model_inspection.md), and [probability calibration](docs/probability_calibration.md).

These are educational, dependency-free implementations. Silhouette scoring is quadratic in sample count; permutation importance performs one model scoring pass per feature and repeat; bootstrap intervals condition on the already-fitted model. Use held-out data for model inspection, and choose time- or group-aware validation whenever observations are not independent.
