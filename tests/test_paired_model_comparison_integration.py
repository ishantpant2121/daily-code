"""Integration tests comparing predictions from existing estimators."""

import unittest

from decision_tree import DecisionTreeClassifier
from knn_classifier import KNNClassifier
from paired_model_comparison import (
    paired_bootstrap_score_difference,
    paired_permutation_test,
)


class PairedModelComparisonIntegrationTests(unittest.TestCase):
    def test_compares_fitted_tree_and_knn_on_shared_holdout(self):
        train_x = [[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]]
        train_y = [0, 0, 0, 1, 1, 1]
        test_x = [[0.05], [0.15], [0.85], [0.95]]
        test_y = [0, 0, 1, 1]
        tree = DecisionTreeClassifier(max_depth=2, random_state=4).fit(train_x, train_y)
        knn = KNNClassifier(n_neighbors=1).fit(train_x, train_y)
        tree_predictions = tree.predict(test_x)
        knn_predictions = knn.predict(test_x)

        interval = paired_bootstrap_score_difference(
            test_y, tree_predictions, knn_predictions,
            scoring="accuracy", n_bootstrap=40, random_state=5,
        )
        test = paired_permutation_test(
            test_y, tree_predictions, knn_predictions,
            scoring="accuracy", n_permutations=40, random_state=5,
        )
        self.assertEqual(interval.observed_difference, 0.0)
        self.assertEqual((interval.lower_bound, interval.upper_bound), (0.0, 0.0))
        self.assertEqual(test.p_value, 1.0)


if __name__ == "__main__":
    unittest.main()
