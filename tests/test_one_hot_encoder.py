"""Tests for dependency-free categorical preprocessing."""

import unittest

from preprocessing import OneHotEncoder


class OneHotEncoderTests(unittest.TestCase):
    def test_encodes_multiple_columns_in_first_seen_category_order(self):
        values = [["red", "small"], ["blue", "large"], ["red", "large"]]
        encoder = OneHotEncoder().fit(values)
        self.assertEqual(encoder.categories_, (("red", "blue"), ("small", "large")))
        self.assertEqual(
            encoder.transform(values),
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]],
        )

    def test_unknown_categories_can_be_ignored(self):
        encoder = OneHotEncoder(handle_unknown="ignore").fit([["a"], ["b"]])
        self.assertEqual(encoder.transform([["c"], ["a"]]), [[0, 0], [1, 0]])

    def test_unknown_categories_raise_by_default(self):
        encoder = OneHotEncoder().fit([["a"], ["b"]])
        with self.assertRaises(ValueError):
            encoder.transform([["c"]])

    def test_rejects_invalid_inputs_and_unfitted_transform(self):
        with self.assertRaises(RuntimeError):
            OneHotEncoder().transform([["a"]])
        for values in ([], [[]], [["a"], ["a", "b"]], [[[1]], [[2]]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                OneHotEncoder().fit(values)
        with self.assertRaises(ValueError):
            OneHotEncoder(handle_unknown="other")
        with self.assertRaises(ValueError):
            OneHotEncoder().fit([["a"], [["unhashable"]]])


if __name__ == "__main__":
    unittest.main()
