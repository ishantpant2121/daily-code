"""Tests for deterministic data splitting utilities."""

import unittest

from data_splitting import k_fold_indices, train_test_split


class TrainTestSplitTests(unittest.TestCase):
    def test_splits_multiple_arrays_with_aligned_indices(self):
        features = list(range(10))
        labels = [value * 10 for value in features]
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=7
        )
        self.assertEqual(len(x_train), 8)
        self.assertEqual(len(x_test), 2)
        self.assertEqual(y_train, [value * 10 for value in x_train])
        self.assertEqual(y_test, [value * 10 for value in x_test])

    def test_is_reproducible_and_does_not_mutate_inputs(self):
        values = list(range(12))
        original = values.copy()
        first = train_test_split(values, random_state=42)
        second = train_test_split(values, random_state=42)
        self.assertEqual(first, second)
        self.assertEqual(values, original)

    def test_supports_ordered_holdout_without_shuffling(self):
        self.assertEqual(
            train_test_split([1, 2, 3, 4], test_size=1, shuffle=False),
            ([1, 2, 3], [4]),
        )

    def test_rejects_invalid_inputs(self):
        cases = [
            lambda: train_test_split(),
            lambda: train_test_split([1]),
            lambda: train_test_split([1, 2], [1]),
            lambda: train_test_split([1, 2], test_size=0),
            lambda: train_test_split([1, 2], test_size=2),
            lambda: train_test_split([1, 2], test_size=True),
            lambda: train_test_split([1, 2], random_state=True),
        ]
        for call in cases:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


class KFoldIndicesTests(unittest.TestCase):
    def test_each_sample_is_validated_once_and_fold_sizes_are_balanced(self):
        folds = k_fold_indices(11, n_splits=4)
        validation = [index for _, test in folds for index in test]
        self.assertEqual(sorted(validation), list(range(11)))
        self.assertEqual(sorted(len(test) for _, test in folds), [2, 3, 3, 3])
        for train, test in folds:
            self.assertFalse(set(train) & set(test))
            self.assertEqual(len(train) + len(test), 11)

    def test_shuffled_folds_are_reproducible(self):
        self.assertEqual(
            k_fold_indices(10, 3, shuffle=True, random_state=5),
            k_fold_indices(10, 3, shuffle=True, random_state=5),
        )

    def test_rejects_invalid_fold_counts(self):
        for args in ((1, 2), (5, 1), (5, 6), (True, 2)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    k_fold_indices(*args)


if __name__ == "__main__":
    unittest.main()
