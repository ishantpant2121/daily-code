"""Greedy scheduling and fractional knapsack."""


def activity_selection(activities):
    """Select the maximum number of non-overlapping intervals.

    Accepts any iterable of (start, finish) pairs, including one-shot
    generators. Intervals are treated as half-open at their finish time.
    """
    activities = list(activities)
    if any(start >= finish for start, finish in activities):
        raise ValueError("invalid interval")

    selected = []
    for activity in sorted(activities, key=lambda item: (item[1], item[0])):
        if not selected or activity[0] >= selected[-1][1]:
            selected.append(activity)
    return selected


def fractional_knapsack(items, capacity):
    """Return the maximum value obtainable with fractional item selection.

    Accepts any iterable of (value, weight) pairs, including one-shot
    generators. Items must have non-negative values and positive weights.
    """
    items = list(items)
    if capacity < 0 or any(value < 0 or weight <= 0 for value, weight in items):
        raise ValueError("invalid item or capacity")

    total = 0.0
    for value, weight in sorted(
        items, key=lambda item: item[0] / item[1], reverse=True
    ):
        taken = min(weight, capacity)
        total += value * taken / weight
        capacity -= taken
        if capacity <= 0:
            break
    return total
