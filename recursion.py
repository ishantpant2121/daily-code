"""Recursion and recurrence examples."""
def factorial(n):
    if n<0:raise ValueError("n must be non-negative")
    out=1
    for x in range(2,n+1):out*=x
    return out
def fibonacci(n):
    if n<0:raise ValueError("n must be non-negative")
    a,b=0,1
    for _ in range(n):a,b=b,a+b
    return a
def sum_recursive(values):
    def visit(lo,hi):
        if lo>=hi:return 0
        if hi-lo==1:return values[lo]
        mid=(lo+hi)//2
        return visit(lo,mid)+visit(mid,hi)
    return visit(0,len(values))
