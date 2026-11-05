"""Space-efficient dynamic programming exercises."""
def climbing_stairs(n):
    if n<0:raise ValueError("n must be non-negative")
    a,b=1,0
    for _ in range(n):a,b=a+b,a
    return a
def coin_change(coins,amount):
    if amount<0 or any(type(c)is not int or c<=0 for c in coins):raise ValueError("invalid coin input")
    dp=[0]+[amount+1]*amount
    for total in range(1,amount+1):
        for coin in coins:
            if coin<=total:dp[total]=min(dp[total],dp[total-coin]+1)
    return dp[amount] if dp[amount]<=amount else -1
def longest_increasing_subsequence(a):
    from bisect import bisect_left
    tails=[]
    for x in a:
        i=bisect_left(tails,x)
        if i==len(tails):tails.append(x)
        else:tails[i]=x
    return len(tails)
