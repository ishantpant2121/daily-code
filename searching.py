"""Array search patterns."""
def linear_search(a,target):
    for i,x in enumerate(a):
        if x==target:return i
    return -1
def lower_bound(a,target):
    lo,hi=0,len(a)
    while lo<hi:
        mid=(lo+hi)//2
        if a[mid]<target:lo=mid+1
        else:hi=mid
    return lo
def rotated_search(a,target):
    lo,hi=0,len(a)-1
    while lo<=hi:
        mid=(lo+hi)//2
        if a[mid]==target:return mid
        if a[lo]<=a[mid]:
            if a[lo]<=target<a[mid]:hi=mid-1
            else:lo=mid+1
        elif a[mid]<target<=a[hi]:lo=mid+1
        else:hi=mid-1
    return -1
