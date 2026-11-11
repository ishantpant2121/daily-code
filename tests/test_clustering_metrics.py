import math
import unittest

from clustering_metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)


class SilhouetteTests(unittest.TestCase):
    def test_separated_clusters_have_high_silhouette(self):
        features = [[0.0], [0.2], [0.4], [10.0], [10.2], [10.4]]
        labels = ["left"] * 3 + ["right"] * 3
        scores = silhouette_samples(features, labels)
        self.assertEqual(len(scores), len(features))
        self.assertTrue(all(score > 0.95 for score in scores))
        self.assertAlmostEqual(silhouette_score(features, labels), sum(scores) / len(scores))

    def test_singleton_cluster_receives_zero(self):
        scores = silhouette_samples([[0], [1], [10], [11]], [0, 0, 1, 2])
        self.assertEqual(scores[-1], 0.0)

    def test_rejects_invalid_cluster_counts_and_data(self):
        for labels in ([0, 0, 0], [0, 1, 2, 3]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                silhouette_score([[0], [1], [2], [3]], labels)
        with self.assertRaises(ValueError):
            silhouette_score([[0], [1]], [0])
        with self.assertRaises(ValueError):
            silhouette_score([[0], [1, 2]], [0, 1])
        with self.assertRaises(ValueError):
            silhouette_score([[float("inf")], [1]], [0, 1])


class AdjustedRandTests(unittest.TestCase):
    def test_is_one_for_identical_partitions_and_label_renaming(self):
        self.assertEqual(adjusted_rand_score([0, 0, 1, 1], [8, 8, 3, 3]), 1.0)

    def test_is_symmetric_and_detects_disagreement(self):
        actual = [0, 0, 1, 1]
        predicted = [0, 1, 0, 1]
        self.assertAlmostEqual(
            adjusted_rand_score(actual, predicted),
            adjusted_rand_score(predicted, actual),
        )
        self.assertLess(adjusted_rand_score(actual, predicted), 0.0)

    def test_degenerate_identical_partitions_and_validation(self):
        self.assertEqual(adjusted_rand_score([1, 1, 1], [2, 2, 2]), 1.0)
        for left, right in (([], []), ([0, 1], [0]), ([[0], [1]], [0, 1])):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                adjusted_rand_score(left, right)


class DaviesBouldinTests(unittest.TestCase):
    def test_separated_compact_clusters_have_low_score(self):
        features = [[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]]
        score = davies_bouldin_score(features, [0, 0, 0, 1, 1, 1])
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 0.02)

    def test_coincident_centroids_with_spread_are_infinite(self):
        score = davies_bouldin_score(
            [[-1.0], [1.0], [-1.0], [1.0]],
            ["a", "a", "b", "b"],
        )
        self.assertTrue(math.isinf(score))

    def test_rejects_single_cluster_and_unhashable_labels(self):
        with self.assertRaises(ValueError):
            davies_bouldin_score([[0], [1]], [0, 0])
        with self.assertRaises(ValueError):
            davies_bouldin_score([[0], [1], [2]], [[0], [1], [1]])


if __name__ == "__main__":
    unittest.main()
