"""Tests for KMP and Z-function string algorithms."""

import unittest

from string_algorithms import kmp_search, prefix_function, z_function


class PrefixFunctionTests(unittest.TestCase):
    def test_computes_known_prefix_values(self):
        self.assertEqual(prefix_function("ababcabab"), [0, 0, 1, 2, 0, 1, 2, 3, 4])
        self.assertEqual(prefix_function("aaaa"), [0, 1, 2, 3])

    def test_handles_empty_and_single_character_strings(self):
        self.assertEqual(prefix_function(""), [])
        self.assertEqual(prefix_function("x"), [0])


class KmpSearchTests(unittest.TestCase):
    def test_finds_overlapping_matches(self):
        self.assertEqual(kmp_search("aaaaa", "aaa"), [0, 1, 2])
        self.assertEqual(kmp_search("ababcabcabababd", "ababd"), [10])

    def test_empty_pattern_matches_every_boundary(self):
        self.assertEqual(kmp_search("abc", ""), [0, 1, 2, 3])

    def test_returns_empty_when_no_match_exists(self):
        self.assertEqual(kmp_search("abcdef", "xyz"), [])

    def test_rejects_non_string_inputs(self):
        with self.assertRaises(TypeError):
            kmp_search("abc", 1)


class ZFunctionTests(unittest.TestCase):
    def test_computes_known_z_values(self):
        self.assertEqual(z_function("aaaaa"), [5, 4, 3, 2, 1])
        self.assertEqual(
            z_function("aabcaabxaaaz"),
            [12, 1, 0, 0, 3, 1, 0, 0, 2, 2, 1, 0],
        )

    def test_handles_empty_string(self):
        self.assertEqual(z_function(""), [])


if __name__ == "__main__":
    unittest.main()
