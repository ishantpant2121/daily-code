"""BFS, DFS and Dijkstra graph algorithms."""
from collections import deque
from heapq import heappop,heappush
def bfs(graph,start):
    seen={start}; q=deque([start]); out=[]
    while q:
        u=q.popleft(); out.append(u)
        for v in graph.get(u,[]):
            if v not in seen:seen.add(v); q.append(v)
    return out
def dfs(graph,start):
    seen=set(); stack=[start]; out=[]
    while stack:
        u=stack.pop()
        if u in seen:continue
        seen.add(u); out.append(u); stack.extend(reversed(graph.get(u,[])))
    return out
def dijkstra(graph,start):
    if any(w<0 for edges in graph.values() for _,w in edges):raise ValueError("negative edge weight")
    dist={start:0.0}; heap=[(0.0,start)]
    while heap:
        d,u=heappop(heap)
        if d!=dist.get(u):continue
        for v,w in graph.get(u,[]):
            nd=d+w
            if nd<dist.get(v,float("inf")):dist[v]=nd; heappush(heap,(nd,v))
    return dist
