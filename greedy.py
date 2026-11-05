"""Greedy scheduling and fractional knapsack."""
def activity_selection(activities):
    if any(s>=f for s,f in activities):raise ValueError("invalid interval")
    out=[]
    for item in sorted(activities,key=lambda x:(x[1],x[0])):
        if not out or item[0]>=out[-1][1]:out.append(item)
    return out
def fractional_knapsack(items,capacity):
    if capacity<0 or any(v<0 or w<=0 for v,w in items):raise ValueError("invalid item or capacity")
    total=0.0
    for v,w in sorted(items,key=lambda x:x[0]/x[1],reverse=True):
        take=min(w,capacity); total+=v*take/w; capacity-=take
        if capacity<=0:break
    return total
