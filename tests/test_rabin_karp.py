"""Tests for rolling-hash Rabin-Karp matching."""

import unittest
from string_algorithms import kmp_search, rabin_karp_search


class RabinKarpSearchTests(unittest.TestCase):
    def test_finds_overlapping_and_multiple_matches(self):
        self.assertEqual(rabin_karp_search("aaaaa", "aa"), [0, 1, 2, 3])
        self.assertEqual(rabin_karp_search("abracadabra", "abra"), [0, 7])

    def test_matches_kmp_for_unicode_and_empty_patterns(self):
        text = "🙂a🙂a🙂"
        pattern = "🙂a"
        self.assertEqual(rabin_karp_search(text, pattern), [0, 2])
        self.assertEqual(
            rabin_karp_search("abc", ""),
            kmp_search("abc", ""),
        )

    def test_handles_longer_missing_and_non_string_inputs(self):
        self.assertEqual(rabin_karp_search("short", "longer"), [])
        self.assertEqual(rabin_karp_search("abcdef", "xyz"), [])
        with self.assertRaises(TypeError):
            rabin_karp_search(123, "2")
        with self.assertRaises(TypeError):
            rabin_karp_search("123", None)


if __name__ == "__main__":
    unittest.main()
