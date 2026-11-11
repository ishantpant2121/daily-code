import unittest

from advanced_splitting import (GroupKFold, GroupShuffleSplit, LeaveOneOut, RepeatedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit, TimeSeriesSplit)


class TimeSeriesSplitTests(unittest.TestCase):
    def test_expanding_windows_keep_future_out_of_training(self):
        folds = TimeSeriesSplit(n_splits=3, test_size=2).split(10)
        self.assertEqual(
            folds,
            [
                ([0, 1, 2, 3], [4, 5]),
                ([0, 1, 2, 3, 4, 5], [6, 7]),
                ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9]),
            ],
        )
        for train, test in folds:
            self.assertLess(max(train), min(test))

    def test_gap_and_bounded_rolling_window(self):
        folds = TimeSeriesSplit(
            n_splits=3, test_size=2, gap=1, max_train_size=3
        ).split(12)
        self.assertEqual(folds[0], ([2, 3, 4], [6, 7]))
        self.assertEqual(folds[-1], ([6, 7, 8], [10, 11]))
        for train, test in folds:
            self.assertEqual(len(train), 3)
            self.assertEqual(min(test) - max(train), 2)

    def test_rejects_impossible_or_invalid_splits(self):
        for kwargs in (
            {"n_splits": 1},
            {"n_splits": True},
            {"max_train_size": 0},
            {"test_size": 0},
            {"gap": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TimeSeriesSplit(**kwargs)
        with self.assertRaises(ValueError):
            TimeSeriesSplit(n_splits=4, test_size=3).split(10)


class GroupKFoldTests(unittest.TestCase):
    def test_groups_never_cross_fold_and_every_sample_is_tested_once(self):
        groups = ["a", "a", "a", "b", "b", "c", "d", "d", "d", "d"]
        folds = GroupKFold(3).split(len(groups), groups)
        validation = [index for _, test in folds for index in test]
        self.assertEqual(sorted(validation), list(range(len(groups))))
        for train, test in folds:
            self.assertFalse(set(train) & set(test))
            self.assertFalse({groups[index] for index in train} & {groups[index] for index in test})

    def test_greedy_assignment_balances_unequal_group_sizes(self):
        groups = ["large"] * 6 + ["medium"] * 3 + ["small-a"] * 2 + ["small-b"] * 2
        folds = GroupKFold(3).split(len(groups), groups)
        sizes = sorted(len(test) for _, test in folds)
        self.assertLessEqual(sizes[-1] - sizes[0], 3)

    def test_rejects_unhashable_or_insufficient_groups(self):
        with self.assertRaises(ValueError):
            GroupKFold(2).split(3, ["only", "only", "only"])
        with self.assertRaisesRegex(ValueError, "hashable"):
            GroupKFold(2).split(4, [["a"], ["a"], ["b"], ["b"]])
        with self.assertRaises(ValueError):
            GroupKFold(2).split(3, ["a", "b"])


class RepeatedKFoldTests(unittest.TestCase):
    def test_repeats_complete_folds_reproducibly(self):
        first = RepeatedKFold(4, 3, random_state=42).split(13)
        second = RepeatedKFold(4, 3, random_state=42).split(13)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        for start in range(0, len(first), 4):
            validation = [index for _, test in first[start:start + 4] for index in test]
            self.assertEqual(sorted(validation), list(range(13)))

    def test_repeats_use_different_partitions_and_validate_options(self):
        folds = RepeatedKFold(3, 3, random_state=5).split(12)
        self.assertNotEqual(folds[:3], folds[3:6])
        for kwargs in (
            {"n_splits": 1},
            {"n_repeats": 0},
            {"random_state": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RepeatedKFold(**kwargs)
        with self.assertRaises(ValueError):
            RepeatedKFold(5, 1).split(4)


class RepeatedStratifiedKFoldTests(unittest.TestCase):
    def test_preserves_class_coverage_in_each_reproducible_repeat(self):
        labels = [0] * 6 + [1] * 6
        first = RepeatedStratifiedKFold(3, 2, random_state=14).split(labels)
        second = RepeatedStratifiedKFold(3, 2, random_state=14).split(labels)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        for train, test in first:
            self.assertEqual({labels[index] for index in test}, {0, 1})
            self.assertFalse(set(train) & set(test))

    def test_rejects_classes_too_small_for_requested_folds(self):
        with self.assertRaises(ValueError):
            RepeatedStratifiedKFold(3, 2, random_state=1).split([0, 0, 1, 1, 1])


class GroupShuffleSplitTests(unittest.TestCase):
    def test_never_splits_groups_and_is_reproducible(self):
        groups = ["a", "a", "b", "b", "c", "d", "d", "e"]
        splitter = GroupShuffleSplit(4, test_size=0.4, random_state=22)
        first = splitter.split(len(groups), groups)
        second = splitter.split(len(groups), groups)
        self.assertEqual(first, second)
        for train, test in first:
            self.assertFalse({groups[index] for index in train} & {groups[index] for index in test})
            self.assertTrue(train)
            self.assertTrue(test)

    def test_rejects_invalid_group_test_sizes(self):
        for size in (0, 1.0, True, float("nan"), 5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                GroupShuffleSplit(test_size=size).split(4, ["a", "b", "c", "d"])


class LeaveOneOutTests(unittest.TestCase):
    def test_each_sample_is_held_out_once(self):
        folds = LeaveOneOut().split(4)
        self.assertEqual(folds, [
            ([1, 2, 3], [0]),
            ([0, 2, 3], [1]),
            ([0, 1, 3], [2]),
            ([0, 1, 2], [3]),
        ])

    def test_rejects_too_few_samples(self):
        with self.assertRaises(ValueError):
            LeaveOneOut().split(1)


class StratifiedShuffleSplitTests(unittest.TestCase):
    def test_preserves_each_class_in_train_and_test(self):
        labels = [0] * 8 + [1] * 4 + [2] * 4
        folds = StratifiedShuffleSplit(3, test_size=0.25, random_state=18).split(labels)
        self.assertEqual(len(folds), 3)
        for train, test in folds:
            self.assertEqual({labels[index] for index in train}, {0, 1, 2})
            self.assertEqual({labels[index] for index in test}, {0, 1, 2})
            self.assertFalse(set(train) & set(test))
            self.assertEqual(len(test), 4)

    def test_is_reproducible_and_supports_integer_test_size(self):
        labels = ["a"] * 5 + ["b"] * 5
        splitter = StratifiedShuffleSplit(2, test_size=4, random_state=2)
        self.assertEqual(splitter.split(labels), splitter.split(labels))
        self.assertTrue(all(len(test) == 4 for _, test in splitter.split(labels)))

    def test_rejects_unhashable_classes_and_impossible_sizes(self):
        with self.assertRaises(ValueError):
            StratifiedShuffleSplit().split([[0], [0], [1], [1]])
        for size in (0, 1.0, True, 10):
            with self.subTest(size=size), self.assertRaises(ValueError):
                StratifiedShuffleSplit(test_size=size).split([0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
