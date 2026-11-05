"""Sorting algorithms for practice."""
def merge_sort(a):
    if len(a)<2:return a.copy()
    m=len(a)//2; x=merge_sort(a[:m]); y=merge_sort(a[m:]); out=[]; i=j=0
    while i<len(x) and j<len(y):
        if x[i]<=y[j]:out.append(x[i]); i+=1
        else:out.append(y[j]); j+=1
    return out+x[i:]+y[j:]
def quick_sort(values):
    a=values.copy()
    def part(lo,hi):
        if lo>=hi:return
        p=a[(lo+hi)//2]; i=lo; j=lo; k=hi
        while j<=k:
            if a[j]<p:a[i],a[j]=a[j],a[i]; i+=1; j+=1
            elif a[j]>p:a[j],a[k]=a[k],a[j]; k-=1
            else:j+=1
        part(lo,i-1); part(k+1,hi)
    part(0,len(a)-1); return a
def counting_sort(values):
    if any(type(v)is not int for v in values):raise TypeError("integers only")
    if not values:return []
    low=min(values); counts=[0]*(max(values)-low+1)
    for v in values:counts[v-low]+=1
    return [i+low for i,c in enumerate(counts) for _ in range(c)]
