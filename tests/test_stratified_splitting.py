"""Tests for stratified cross-validation splits."""

import unittest
from collections import Counter

from data_splitting import stratified_k_fold_indices


class StratifiedKFoldTests(unittest.TestCase):
    def test_preserves_class_balance_and_uses_each_sample_once(self):
        labels = ["a"] * 7 + ["b"] * 5 + ["c"] * 4
        folds = stratified_k_fold_indices(labels, n_splits=4)
        validation_indices = [index for _, validation in folds for index in validation]
        self.assertEqual(sorted(validation_indices), list(range(len(labels))))
        sizes = [len(validation) for _, validation in folds]
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        per_class = [Counter(labels[index] for index in validation) for _, validation in folds]
        for label in set(labels):
            counts = [fold.get(label, 0) for fold in per_class]
            self.assertLessEqual(max(counts) - min(counts), 1)
        for training, validation in folds:
            self.assertFalse(set(training) & set(validation))
            self.assertEqual(len(training) + len(validation), len(labels))

    def test_shuffled_folds_are_reproducible(self):
        labels = ["positive"] * 8 + ["negative"] * 8
        self.assertEqual(
            stratified_k_fold_indices(labels, 4, shuffle=True, random_state=12),
            stratified_k_fold_indices(labels, 4, shuffle=True, random_state=12),
        )

    def test_rejects_small_classes_unhashable_labels_and_invalid_options(self):
        with self.assertRaisesRegex(ValueError, "each class"):
            stratified_k_fold_indices(["a", "a", "a", "b"], 3)
        with self.assertRaisesRegex(ValueError, "hashable"):
            stratified_k_fold_indices([[1], [1], [2], [2]], 2)
        for kwargs in (
            {"n_splits": 1},
            {"n_splits": 5},
            {"shuffle": 1},
            {"random_state": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                stratified_k_fold_indices(["a", "a", "b", "b"], **kwargs)


if __name__ == "__main__":
    unittest.main()
