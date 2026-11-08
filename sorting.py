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
    """Return a sorted copy using iterative three-way partitioning."""
    result = values.copy()
    pending = [(0, len(result) - 1)]
    while pending:
        low, high = pending.pop()
        while low < high:
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
            left = (low, smaller - 1)
            right = (larger + 1, high)
            left_size = left[1] - left[0] + 1
            right_size = right[1] - right[0] + 1
            if left_size <= right_size:
                if right_size > 1:
                    pending.append(right)
                low, high = left
            else:
                if left_size > 1:
                    pending.append(left)
                low, high = right
    return result


def counting_sort(values):
    """Sort signed integers in O(n + range) time; accepts any iterable."""
    materialized = list(values)
    if any(type(value) is not int for value in materialized):
        raise TypeError("integers only")
    if not materialized:
        return []
    low, high = min(materialized), max(materialized)
    counts = [0] * (high - low + 1)
    for value in materialized:
        counts[value - low] += 1
    return [
        value + low
        for value, count in enumerate(counts)
        for _ in range(count)
    ]


def heap_sort(values):
    """Return a sorted copy using a max heap."""
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


def quickselect(values, k):
    """Return the zero-based k-th smallest value without sorting all values."""
    if type(k) is not int or not 0 <= k < len(values):
        raise ValueError("k must be an integer index in range")
    result = values.copy()
    low, high = 0, len(result) - 1
    while low <= high:
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
        if k < smaller:
            high = smaller - 1
        elif k > larger:
            low = larger + 1
        else:
            return result[k]
    raise RuntimeError("selection failed")


def count_inversions(values):
    """Count pairs (i, j) where i < j and values[i] > values[j] in O(n log n)."""
    result = list(values)
    scratch = result.copy()
    def count_range(start, end):
        if end - start < 2:
            return 0
        middle = (start + end) // 2
        inversions = count_range(start, middle) + count_range(middle, end)
        left, right, output = start, middle, start
        while left < middle and right < end:
            if result[left] <= result[right]:
                scratch[output] = result[left]
                left += 1
            else:
                scratch[output] = result[right]
                right += 1
                inversions += middle - left
            output += 1
        while left < middle:
            scratch[output] = result[left]
            left += 1
            output += 1
        while right < end:
            scratch[output] = result[right]
            right += 1
            output += 1
        result[start:end] = scratch[start:end]
        return inversions
    return count_range(0, len(result))
