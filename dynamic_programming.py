"""Space-efficient dynamic programming exercises."""


def climbing_stairs(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    previous, current = 1, 0
    for _ in range(n):
        previous, current = previous + current, previous
    return previous


def coin_change(coins, amount):
    """Return the minimum number of coins needed to make amount.

    Coins may be supplied as any iterable, including a one-shot generator.
    Returns -1 when the amount cannot be formed.
    """
    coins = list(coins)
    if amount < 0 or any(type(coin) is not int or coin <= 0 for coin in coins):
        raise ValueError("invalid coin input")

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
