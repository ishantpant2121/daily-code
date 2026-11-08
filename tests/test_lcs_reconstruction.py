"""Tests for LCS sequence reconstruction."""

import unittest
from dynamic_programming import longest_common_subsequence, longest_common_subsequence_length


def is_subsequence(candidate, source):
    iterator = iter(source)
    return all(any(value == item for item in iterator) for value in candidate)


class LongestCommonSubsequenceReconstructionTests(unittest.TestCase):
    def test_reconstructs_a_valid_maximum_length_subsequence(self):
        first = "ABCBDAB"
        second = "BDCABA"
        result = longest_common_subsequence(first, second)
        self.assertEqual(len(result), 4)
        self.assertTrue(is_subsequence(result, first))
        self.assertTrue(is_subsequence(result, second))
        self.assertEqual(len(result), longest_common_subsequence_length(first, second))

    def test_supports_generic_sequences_and_empty_inputs(self):
        self.assertEqual(longest_common_subsequence([1, 2, 3], [2, 3, 4]), [2, 3])
        self.assertEqual(longest_common_subsequence([], [1, 2]), [])
        self.assertEqual(longest_common_subsequence("same", "same"), list("same"))

    def test_does_not_modify_inputs_and_is_deterministic(self):
        first = [1, 2, 1, 2]
        second = [2, 1, 2, 1]
        original = (first.copy(), second.copy())
        result = longest_common_subsequence(first, second)
        self.assertEqual(result, longest_common_subsequence(first, second))
        self.assertEqual((first, second), original)


if __name__ == "__main__":
    unittest.main()
