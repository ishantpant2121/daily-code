"""Tests for the dependency-free CART decision-tree classifier."""

import unittest

from decision_tree import DecisionTreeClassifier


class DecisionTreeClassifierTests(unittest.TestCase):
    def test_learns_a_separable_binary_problem_and_probabilities(self):
        features = [[0], [1], [2], [3]]
        labels = ["low", "low", "high", "high"]
        model = DecisionTreeClassifier().fit(features, labels)
        self.assertEqual(model.predict(features), labels)
        self.assertEqual(model.classes_, ("low", "high"))
        self.assertEqual(model.predict_proba([[0], [3]]), [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(model.score(features, labels), 1.0)

    def test_supports_entropy_and_multiclass_labels(self):
        features = [[0], [1], [2], [3], [4], [5]]
        labels = ["a", "a", "b", "b", "c", "c"]
        model = DecisionTreeClassifier(criterion="entropy").fit(features, labels)
        self.assertEqual(model.predict(features), labels)
        self.assertEqual(model.classes_, ("a", "b", "c"))

    def test_depth_and_leaf_constraints_limit_splitting(self):
        features = [[0], [1], [2], [3]]
        labels = [0, 0, 1, 1]
        shallow = DecisionTreeClassifier(max_depth=1).fit(features, labels)
        self.assertEqual(shallow.max_depth_, 1)
        self.assertEqual(shallow.node_count_, 3)
        constrained = DecisionTreeClassifier(min_samples_leaf=2).fit(features, labels)
        self.assertEqual(constrained.n_leaves_, 2)
        self.assertEqual(constrained.predict(features), labels)

    def test_min_samples_split_prevents_small_nodes_from_splitting(self):
        model = DecisionTreeClassifier(min_samples_split=5).fit(
            [[0], [1], [2], [3]], [0, 0, 1, 1]
        )
        self.assertEqual(model.node_count_, 1)
        self.assertEqual(model.n_leaves_, 1)

    def test_feature_importances_are_normalized(self):
        features = [[0, 5], [1, 5], [2, 5], [3, 5]]
        labels = [0, 0, 1, 1]
        model = DecisionTreeClassifier().fit(features, labels)
        self.assertAlmostEqual(sum(model.feature_importances_), 1.0)
        self.assertEqual(model.feature_importances_, [1.0, 0.0])

    def test_constant_features_and_single_class_data_make_leaf_models(self):
        model = DecisionTreeClassifier().fit([[2], [2], [2]], ["same", "same", "same"])
        self.assertEqual(model.predict([[2]]), ["same"])
        self.assertEqual(model.feature_importances_, [0.0])
        self.assertEqual(model.n_leaves_, 1)
        self.assertEqual(model.max_depth_, 0)

    def test_accepts_one_shot_iterables_and_apply_returns_leaf_ids(self):
        model = DecisionTreeClassifier().fit(
            ([value] for value in [0, 1, 2, 3]),
            (label for label in ["a", "a", "b", "b"]),
        )
        leaves = model.apply([[0], [3]])
        self.assertEqual(len(set(leaves)), 2)

    def test_rejects_invalid_configuration_and_data(self):
        for kwargs in (
            {"criterion": "bad"},
            {"max_depth": 0},
            {"max_depth": True},
            {"min_samples_split": 1},
            {"min_samples_leaf": 0},
            {"min_impurity_decrease": -1},
            {"min_impurity_decrease": float("nan")},
            {"min_impurity_decrease": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DecisionTreeClassifier(**kwargs)

        for features, labels in (
            ([], []),
            ([[1], [2]], [0]),
            ([[True], [2]], [0, 1]),
            ([[float("inf")], [2]], [0, 1]),
            ([[10**400], [2]], [0, 1]),
            ([[1], [2, 3]], [0, 1]),
            ([[1], [2]], [[0], [1]]),
        ):
            with self.subTest(features=features, labels=labels), self.assertRaises(ValueError):
                DecisionTreeClassifier().fit(features, labels)

    def test_rejects_unfitted_prediction_and_feature_width_mismatch(self):
        model = DecisionTreeClassifier()
        with self.assertRaises(RuntimeError):
            model.predict([[1]])
        model.fit([[0, 1], [1, 2]], [0, 1])
        with self.assertRaises(ValueError):
            model.predict([[1]])


if __name__ == "__main__":
    unittest.main()
