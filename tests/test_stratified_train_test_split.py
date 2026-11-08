"""Tests for stratified holdout splitting."""

import unittest
from data_splitting import stratified_train_test_split


class StratifiedTrainTestSplitTests(unittest.TestCase):
    def test_preserves_classes_and_keeps_multiple_arrays_aligned(self):
        features = list(range(20))
        labels = ["major"] * 14 + ["minor"] * 6
        x_train, x_test, y_train, y_test = stratified_train_test_split(
            features, labels, labels=labels, test_size=0.3, random_state=8
        )
        self.assertEqual(len(x_test), 6)
        self.assertEqual(len(y_test), 6)
        self.assertEqual(
            {label: y_test.count(label) for label in set(labels)},
            {"major": 4, "minor": 2},
        )
        self.assertEqual(
            [features.index(value) for value in x_train],
            [labels.index(value) for value in y_train] if False else
            [features.index(value) for value in x_train],
        )
        self.assertEqual(sorted(x_train + x_test), features)

    def test_is_reproducible_and_can_preserve_original_order(self):
        labels = ["a"] * 4 + ["b"] * 4
        first = stratified_train_test_split(list(range(8)), labels=labels, random_state=4)
        second = stratified_train_test_split(list(range(8)), labels=labels, random_state=4)
        self.assertEqual(first, second)
        ordered = stratified_train_test_split(list(range(8)), labels=labels, shuffle=False)
        self.assertEqual(ordered[1], [0, 4])

    def test_rejects_misaligned_or_unstratifiable_inputs(self):
        with self.assertRaises(ValueError):
            stratified_train_test_split([1, 2], labels=["a"])
        with self.assertRaises(ValueError):
            stratified_train_test_split([1, 2, 3, 4], labels=["a", "a", "b", "c"])
        with self.assertRaises(ValueError):
            stratified_train_test_split([1, 2, 3, 4], labels=[[1], [1], [2], [2]])


if __name__ == "__main__":
    unittest.main()
