"""Search helpers for unsorted and sorted sequences."""

from collections.abc import Sequence


def linear_search(values: Sequence[int], target: int) -> int:
    """Return the first matching index, or -1 when target is absent.

    Time: O(n) | Auxiliary space: O(1).
    """
    for index, value in enumerate(values):
        if value == target:
            return index
    return -1


def lower_bound(values: Sequence[int], target: int) -> int:
    """Return the first index whose value is at least target.

    The input must be sorted in ascending order. Returns len(values) when
    every value is smaller than target.

    Time: O(log n) | Auxiliary space: O(1).
    """
    low, high = 0, len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle
    return low


def rotated_search(values: Sequence[int], target: int) -> int:
    """Search an ascending array rotated at an unknown pivot.

    Duplicate values are supported. In ambiguous cases where both ends and
    the midpoint are equal, the search narrows both bounds by one. This keeps
    the usual O(log n) behavior when the sorted half is identifiable, with
    O(n) worst-case time for arrays containing many duplicates.

    Return any matching index, or -1 when target is absent.
    """
    low, high = 0, len(values) - 1

    while low <= high:
        middle = (low + high) // 2
        if values[middle] == target:
            return middle

        if values[low] == values[middle] == values[high]:
            low += 1
            high -= 1
            continue

        if values[low] <= values[middle]:
            if values[low] <= target < values[middle]:
                high = middle - 1
            else:
                low = middle + 1
        elif values[middle] < target <= values[high]:
            low = middle + 1
        else:
            high = middle - 1

    return -1



def search_range(values: Sequence[int], target: int) -> tuple[int, int]:
    """Return the first and last positions of target in a sorted sequence.

    Returns (-1, -1) when target is absent. Both bounds are found with binary
    search, so the running time is O(log n).
    """
    first = lower_bound(values, target)
    if first == len(values) or values[first] != target:
        return -1, -1

    low, high = first, len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] <= target:
            low = middle + 1
        else:
            high = middle
    return first, low - 1
