"""
Day 2 — DSA Pattern Practice
=============================

Patterns covered:
1. Prefix sums
2. Binary search
3. Stack

Run:
    python day2.py
"""


def range_sum(nums: list[int], left: int, right: int) -> int:
    """Return the inclusive sum nums[left:right + 1].

    Prefix sums: build once in O(n), answer each query in O(1).
    Space: O(n). Raises IndexError for invalid bounds.
    """
    if left < 0 or right >= len(nums) or left > right:
        raise IndexError("invalid inclusive range")

    prefix = [0]
    for value in nums:
        prefix.append(prefix[-1] + value)

    return prefix[right + 1] - prefix[left]


def binary_search(nums: list[int], target: int) -> int:
    """Return the index of target in sorted nums, or -1.

    Time: O(log n) | Auxiliary space: O(1).
    """
    low, high = 0, len(nums) - 1

    while low <= high:
        mid = low + (high - low) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1


def is_balanced(expression: str) -> bool:
    """Check whether (), [] and {} are correctly matched.

    Stack pattern: each closing bracket must match the latest opening one.
    Time: O(n) | Space: O(n).
    """
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []

    for char in expression:
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False

    return not stack


def main() -> None:
    print("Day 2 — DSA Pattern Practice")
    print("-" * 32)
    print("Prefix sum | range [1, 3]:", range_sum([2, 4, 1, 7, 3], 1, 3))
    print("Binary search | index of 7:", binary_search([1, 3, 5, 7, 9], 7))
    print("Stack | balanced brackets:", is_balanced("{[()]}"))


if __name__ == "__main__":
    main()
