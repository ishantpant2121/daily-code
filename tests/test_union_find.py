"""Regression tests for disjoint-set operations and input validation."""

import unittest

from union_find import DisjointSet


class DisjointSetTests(unittest.TestCase):
    def test_union_find_and_component_count(self):
        sets = DisjointSet(5)
        self.assertTrue(sets.union(0, 1))
        self.assertTrue(sets.union(2, 3))
        self.assertTrue(sets.union(1, 2))
        self.assertFalse(sets.union(0, 3))
        self.assertTrue(sets.connected(0, 3))
        self.assertFalse(sets.connected(0, 4))
        self.assertEqual(sets.components, 2)

    def test_supports_an_empty_set(self):
        sets = DisjointSet(0)
        self.assertEqual(sets.components, 0)
        with self.assertRaises(IndexError):
            sets.find(0)

    def test_rejects_invalid_sizes(self):
        for size in (-1, 1.5, True):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    DisjointSet(size)

    def test_rejects_invalid_element_indices(self):
        sets = DisjointSet(3)
        for element in (-1, 3, 1.5, True):
            with self.subTest(element=element):
                with self.assertRaises(IndexError):
                    sets.find(element)


if __name__ == "__main__":
    unittest.main()
