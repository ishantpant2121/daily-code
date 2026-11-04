"""Regression tests for the repository's original DSA examples."""

import unittest

from day1 import is_palindrome, max_sum_subarray, two_sum
from day2 import binary_search, is_balanced, range_sum
from day3_linked_lists import ListNode, has_cycle, reverse_list, to_list


class ExistingDSATests(unittest.TestCase):
    def test_day_one_patterns(self):
        self.assertEqual(two_sum([2, 7, 11, 15], 9), [0, 1])
        self.assertEqual(two_sum([1, 2], 9), [])
        self.assertTrue(is_palindrome("A man, a plan, a canal: Panama"))
        self.assertFalse(is_palindrome("daily code"))
        self.assertEqual(max_sum_subarray([-5, -2, -8], 1), -2)
        with self.assertRaises(ValueError):
            max_sum_subarray([1, 2], 0)

    def test_day_two_patterns(self):
        self.assertEqual(range_sum([2, 4, 1, 7, 3], 1, 3), 12)
        self.assertEqual(binary_search([1, 3, 5, 7, 9], 7), 3)
        self.assertEqual(binary_search([], 7), -1)
        self.assertTrue(is_balanced("{[()]}"))
        self.assertFalse(is_balanced("([)]"))
        with self.assertRaises(IndexError):
            range_sum([1, 2], 0, 2)

    def test_day_three_linked_lists(self):
        head = ListNode(1, ListNode(2, ListNode(3)))
        reversed_head = reverse_list(head)
        self.assertEqual(to_list(reversed_head), [3, 2, 1])
        self.assertFalse(has_cycle(reversed_head))
        first, second = ListNode(4), ListNode(5)
        first.next = second
        second.next = first
        self.assertTrue(has_cycle(first))
        self.assertFalse(has_cycle(None))
        with self.assertRaises(ValueError):
            to_list(first)


if __name__ == "__main__":
    unittest.main()
