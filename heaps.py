"""Heap-based selection and k-way merging."""
from heapq import heappop,heappush
def kth_largest(values,k):
    if not 1<=k<=len(values):raise ValueError("k out of range")
    h=[]
    for x in values:
        heappush(h,x)
        if len(h)>k:heappop(h)
    return h[0]
def merge_sorted_lists(lists):
    h=[]
    for i,row in enumerate(lists):
        if row:heappush(h,(row[0],i,0))
    out=[]
    while h:
        x,i,j=heappop(h); out.append(x); j+=1
        if j<len(lists[i]):heappush(h,(lists[i][j],i,j))
    return out
