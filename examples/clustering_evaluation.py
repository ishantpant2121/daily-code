"""Evaluate K-means assignments with internal and external metrics."""

from clustering_metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from kmeans import KMeans

features = [[0.0, 0.0], [0.2, 0.1], [0.1, 0.3], [8.0, 8.0], [8.2, 8.1], [8.1, 8.3]]
reference_labels = ["near"] * 3 + ["far"] * 3
predicted = KMeans(n_clusters=2, random_state=12, n_init=5).fit_predict(features)

print("Silhouette:", silhouette_score(features, predicted))
print("Davies-Bouldin:", davies_bouldin_score(features, predicted))
print("Adjusted Rand:", adjusted_rand_score(reference_labels, predicted))
