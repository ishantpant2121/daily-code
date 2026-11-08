"""Tests for iterable-aware sorting and selection inputs."""

import unittest
from sorting import counting_sort, quickselect


class CountingSortIterableTests(unittest.TestCase):
    def test_accepts_one_shot_generators(self):
        self.assertEqual(counting_sort(value for value in [3, -1, 2, 3, 0]), [-1, 0, 2, 3, 3])

    def test_handles_empty_generators_and_rejects_non_integers(self):
        self.assertEqual(counting_sort(iter(())), [])
        with self.assertRaises(TypeError):
            counting_sort(value for value in [1, True, 2])


class QuickselectIndexValidationTests(unittest.TestCase):
    def test_rejects_boolean_and_fractional_indices(self):
        for index in (True, 1.0, 1.5, "1"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                quickselect([1, 2, 3], index)


if __name__ == "__main__":
    unittest.main()
