"""Subsets and permutations using backtracking."""
def subsets(values):
    out=[]; path=[]
    def visit(i):
        if i==len(values):out.append(path.copy()); return
        visit(i+1); path.append(values[i]); visit(i+1); path.pop()
    visit(0); return out
def permutations(values):
    out=[]; a=values.copy()
    def visit(i):
        if i==len(a):out.append(a.copy()); return
        for j in range(i,len(a)):
            a[i],a[j]=a[j],a[i]; visit(i+1); a[i],a[j]=a[j],a[i]
    visit(0); return out
