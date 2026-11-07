"""Heap-based selection and k-way merging."""

from heapq import heappop, heappush


def kth_largest(values, k):
    """Return the k-th largest value from an iterable using O(k) space.

    The input is consumed once and is not sorted or materialized. The running
    time is O(n log k), making the function suitable for one-shot streams.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")

    smallest_of_largest = []
    for value in values:
        heappush(smallest_of_largest, value)
        if len(smallest_of_largest) > k:
            heappop(smallest_of_largest)

    if len(smallest_of_largest) < k:
        raise ValueError("k exceeds the number of values")
    return smallest_of_largest[0]


def merge_sorted_lists(lists):
    """Merge sorted input sequences into one sorted list.

    Uses a min heap with one active entry per input sequence. Inputs must
    already be sorted in ascending order.
    """
    heap = []
    for list_index, values in enumerate(lists):
        if values:
            heappush(heap, (values[0], list_index, 0))

    merged = []
    while heap:
        value, list_index, value_index = heappop(heap)
        merged.append(value)
        next_index = value_index + 1
        if next_index < len(lists[list_index]):
            heappush(
                heap,
                (lists[list_index][next_index], list_index, next_index),
            )
    return merged
