"""
Day 1 — DSA Pattern Practice
============================

A small, runnable collection of three foundational patterns:
1. Hash map / frequency counting
2. Two pointers
3. Sliding window

Run:
    python day1.py
"""


def two_sum(nums: list[int], target: int) -> list[int]:
    """Return indices of two values adding to target, or [] if none exist.

    Pattern: Hash map
    Time: O(n) average | Space: O(n)
    """
    seen: dict[int, int] = {}

    for index, value in enumerate(nums):
        complement = target - value
        if complement in seen:
            return [seen[complement], index]
        seen[value] = index

    return []


def is_palindrome(text: str) -> bool:
    """Check a string using two pointers, ignoring case and non-alphanumerics.

    Pattern: Two pointers
    Time: O(n) | Space: O(1) auxiliary (excluding the normalized string)
    """
    left, right = 0, len(text) - 1

    while left < right:
        while left < right and not text[left].isalnum():
            left += 1
        while left < right and not text[right].isalnum():
            right -= 1

        if text[left].lower() != text[right].lower():
            return False

        left += 1
        right -= 1

    return True


def max_sum_subarray(nums: list[int], window_size: int) -> int:
    """Return the largest sum among all fixed-size contiguous windows.

    Pattern: Fixed-size sliding window
    Time: O(n) | Space: O(1)
    Raises ValueError when the requested window is invalid.
    """
    if window_size <= 0 or window_size > len(nums):
        raise ValueError("window_size must be between 1 and len(nums)")

    current_sum = sum(nums[:window_size])
    best_sum = current_sum

    for right in range(window_size, len(nums)):
        current_sum += nums[right] - nums[right - window_size]
        best_sum = max(best_sum, current_sum)

    return best_sum


def main() -> None:
    print("Day 1 — DSA Pattern Practice")
    print("-" * 32)

    print("Hash map | Two Sum:", two_sum([2, 7, 11, 15], 9))
    print("Two pointers | Palindrome:", is_palindrome("A man, a plan, a canal: Panama"))
    print("Sliding window | Maximum sum:", max_sum_subarray([2, 1, 5, 1, 3, 2], 3))


if __name__ == "__main__":
    main()
