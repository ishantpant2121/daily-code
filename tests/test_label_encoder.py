"""Tests for stable categorical label encoding."""

import unittest

from preprocessing import LabelEncoder


class LabelEncoderTests(unittest.TestCase):
    def test_encodes_and_restores_labels_in_first_seen_order(self):
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(["dog", "cat", "dog", "bird"])
        self.assertEqual(encoded, [0, 1, 0, 2])
        self.assertEqual(encoder.classes_, ("dog", "cat", "bird"))
        self.assertEqual(encoder.inverse_transform(encoded), ["dog", "cat", "dog", "bird"])

    def test_supports_one_shot_iterables(self):
        encoder = LabelEncoder().fit(value for value in ["yes", "no", "yes"])
        self.assertEqual(encoder.transform(["no", "yes"]), [1, 0])

    def test_rejects_unknown_labels_and_invalid_indices(self):
        encoder = LabelEncoder().fit(["a", "b"])
        with self.assertRaises(ValueError):
            encoder.transform(["c"])
        for values in ([True], [-1], [2], [1.0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                encoder.inverse_transform(values)

    def test_rejects_unfitted_empty_and_unhashable_inputs(self):
        with self.assertRaises(RuntimeError):
            LabelEncoder().transform(["a"])
        with self.assertRaises(ValueError):
            LabelEncoder().fit([])
        with self.assertRaises(ValueError):
            LabelEncoder().fit([[1], [2]])


if __name__ == "__main__":
    unittest.main()
