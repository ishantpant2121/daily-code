"""Sorting algorithms for practice."""


def merge_sort(values):
    """Return a stable sorted copy using divide and conquer."""
    if len(values) < 2:
        return values.copy()

    middle = len(values) // 2
    left = merge_sort(values[:middle])
    right = merge_sort(values[middle:])
    output = []
    left_index = right_index = 0

    while left_index < len(left) and right_index < len(right):
        if left[left_index] <= right[right_index]:
            output.append(left[left_index])
            left_index += 1
        else:
            output.append(right[right_index])
            right_index += 1
    return output + left[left_index:] + right[right_index:]


def quick_sort(values):
    """Return a sorted copy using an in-place three-way partition."""
    result = values.copy()

    def partition(low, high):
        if low >= high:
            return
        pivot = result[(low + high) // 2]
        smaller = current = low
        larger = high
        while current <= larger:
            if result[current] < pivot:
                result[smaller], result[current] = result[current], result[smaller]
                smaller += 1
                current += 1
            elif result[current] > pivot:
                result[current], result[larger] = result[larger], result[current]
                larger -= 1
            else:
                current += 1
        partition(low, smaller - 1)
        partition(larger + 1, high)

    partition(0, len(result) - 1)
    return result


def counting_sort(values):
    """Sort signed integers in O(n + range) time."""
    if any(type(value) is not int for value in values):
        raise TypeError("integers only")
    if not values:
        return []
    low, high = min(values), max(values)
    counts = [0] * (high - low + 1)
    for value in values:
        counts[value - low] += 1
    return [
        value + low
        for value, count in enumerate(counts)
        for _ in range(count)
    ]


def heap_sort(values):
    """Return a sorted copy using a max heap.

    Time: O(n log n) | Auxiliary space: O(1) beyond the returned copy.
    """
    result = values.copy()
    size = len(result)

    def sift_down(root, end):
        while 2 * root + 1 < end:
            child = 2 * root + 1
            if child + 1 < end and result[child] < result[child + 1]:
                child += 1
            if result[root] >= result[child]:
                return
            result[root], result[child] = result[child], result[root]
            root = child

    for root in range(size // 2 - 1, -1, -1):
        sift_down(root, size)

    for end in range(size - 1, 0, -1):
        result[0], result[end] = result[end], result[0]
        sift_down(0, end)

    return result
