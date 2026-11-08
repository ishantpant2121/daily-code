"""Space-efficient dynamic programming exercises."""


def climbing_stairs(n):
    if type(n) is not int or n < 0:
        raise ValueError("n must be a non-negative integer")
    previous, current = 1, 0
    for _ in range(n):
        previous, current = previous + current, previous
    return previous


def coin_change(coins, amount):
    """Return the minimum number of coins needed to make amount.

    Coins may be supplied as any iterable, including a one-shot generator.
    Returns -1 when the amount cannot be formed.
    """
    if type(amount) is not int or amount < 0:
        raise ValueError("amount must be a non-negative integer")
    coins = list(coins)
    if any(type(coin) is not int or coin <= 0 for coin in coins):
        raise ValueError("coins must be positive integers")
    coins = sorted(set(coins))
    dp = [0] + [amount + 1] * amount
    for total in range(1, amount + 1):
        for coin in coins:
            if coin <= total:
                dp[total] = min(dp[total], dp[total - coin] + 1)
    return dp[amount] if dp[amount] <= amount else -1


def longest_increasing_subsequence(values):
    from bisect import bisect_left
    tails = []
    for value in values:
        index = bisect_left(tails, value)
        if index == len(tails):
            tails.append(value)
        else:
            tails[index] = value
    return len(tails)


def knapsack_01(values, weights, capacity):
    """Return the maximum value for 0/1 knapsack using O(capacity) space."""
    if len(values) != len(weights):
        raise ValueError("values and weights must have equal lengths")
    if type(capacity) is not int or capacity < 0:
        raise ValueError("capacity must be a non-negative integer")
    if any(type(weight) is not int or weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative integers")
    best = [0] * (capacity + 1)
    for value, weight in zip(values, weights):
        for remaining in range(capacity, weight - 1, -1):
            best[remaining] = max(
                best[remaining],
                best[remaining - weight] + value,
            )
    return best[capacity]


def longest_common_subsequence_length(first, second):
    """Return the LCS length in O(m*n) time and O(min(m, n)) space."""
    if len(first) < len(second):
        first, second = second, first
    previous = [0] * (len(second) + 1)
    for left in first:
        current = [0]
        for index, right in enumerate(second, start=1):
            if left == right:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def longest_common_subsequence(first, second):
    """Return one longest common subsequence as a list of elements.

    Uses a full O(m*n) dynamic-programming table to reconstruct an actual
    subsequence. On equal-length alternatives, backtracking moves upward,
    making the result deterministic for a given pair of inputs.
    """
    left_values = list(first)
    right_values = list(second)
    rows = len(left_values)
    columns = len(right_values)
    table = [[0] * (columns + 1) for _ in range(rows + 1)]

    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            if left_values[row - 1] == right_values[column - 1]:
                table[row][column] = table[row - 1][column - 1] + 1
            else:
                table[row][column] = max(
                    table[row - 1][column],
                    table[row][column - 1],
                )

    result = []
    row, column = rows, columns
    while row and column:
        if left_values[row - 1] == right_values[column - 1]:
            result.append(left_values[row - 1])
            row -= 1
            column -= 1
        elif table[row - 1][column] >= table[row][column - 1]:
            row -= 1
        else:
            column -= 1
    result.reverse()
    return result


def edit_distance(source, target):
    """Return Levenshtein distance using O(m*n) time and O(min(m, n)) space."""
    if len(source) < len(target):
        source, target = target, source
    previous = list(range(len(target) + 1))
    for row, left in enumerate(source, start=1):
        current = [row]
        for column, right in enumerate(target, start=1):
            substitution = 0 if left == right else 1
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + substitution,
            ))
        previous = current
    return previous[-1]
