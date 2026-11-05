"""Disjoint Set Union using path compression and union by size."""
class DisjointSet:
    def __init__(self,size):
        if size<0:raise ValueError("size must be non-negative")
        self.parent=list(range(size)); self.sizes=[1]*size; self.components=size
    def find(self,x):
        if not 0<=x<len(self.parent):raise IndexError("element out of range")
        if self.parent[x]!=x:self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,a,b):
        a=self.find(a); b=self.find(b)
        if a==b:return False
        if self.sizes[a]<self.sizes[b]:a,b=b,a
        self.parent[b]=a; self.sizes[a]+=self.sizes[b]; self.components-=1; return True
    def connected(self,a,b):return self.find(a)==self.find(b)
