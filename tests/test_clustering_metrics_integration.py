import unittest

from clustering_metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from kmeans import KMeans


class ClusteringMetricsIntegrationTests(unittest.TestCase):
    def test_metrics_evaluate_kmeans_assignments_against_geometry_and_reference(self):
        features = [[0.0], [0.1], [0.2], [0.3], [8.0], [8.1], [8.2], [8.3]]
        reference = ["left"] * 4 + ["right"] * 4
        predicted = KMeans(2, n_init=4, random_state=17).fit_predict(features)
        self.assertGreater(silhouette_score(features, predicted), 0.95)
        self.assertLess(davies_bouldin_score(features, predicted), 0.05)
        self.assertEqual(adjusted_rand_score(reference, predicted), 1.0)


if __name__ == "__main__":
    unittest.main()
