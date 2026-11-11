# Clustering evaluation metrics

`clustering_metrics.py` provides small, dependency-free metrics for checking
cluster assignments.

## Internal metrics

`silhouette_samples(X, labels)` returns one Euclidean silhouette coefficient
per row. `silhouette_score(X, labels)` returns their mean. Values near 1
indicate that samples are closer to their own cluster than to neighboring
clusters; values near 0 indicate overlap; negative values can indicate
questionable assignments. A singleton cluster receives a sample score of zero.

`davies_bouldin_score(X, labels)` measures the average worst-case ratio between
within-cluster spread and centroid separation. Lower values are preferable.
Coincident centroids with non-zero spread yield positive infinity because the
separation is undefined.

Both internal metrics require at least two clusters and fewer clusters than
samples. They use Euclidean distance. Silhouette computation is quadratic in
the number of samples and intentionally avoids allocating a full pairwise
distance matrix.

## External metric

`adjusted_rand_score(reference_labels, predicted_labels)` compares two
partitions without requiring their cluster IDs to match. It is 1 for identical
partitions (even when labels are renamed), approximately 0 for chance-level
agreement, and can be negative for less-than-chance agreement. It requires
hashable labels and at least two observations.

## Example

```python
from clustering_metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from kmeans import KMeans

X = [[0.0], [0.2], [0.1], [8.0], [8.2], [8.1]]
reference = ["near"] * 3 + ["far"] * 3
predicted = KMeans(2, random_state=7).fit_predict(X)

print(silhouette_score(X, predicted))
print(davies_bouldin_score(X, predicted))
print(adjusted_rand_score(reference, predicted))
```

Internal scores assess geometry in the supplied feature space; external scores
require reference labels. Scale features appropriately before distance-based
clustering, and do not compare internal scores across datasets with different
distance geometry without care.
