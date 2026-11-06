"""BFS, DFS, shortest paths, and topological ordering."""

from collections import deque
from heapq import heappop, heappush


def bfs(graph, start):
    seen = {start}
    queue = deque([start])
    output = []
    while queue:
        node = queue.popleft()
        output.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return output


def dfs(graph, start):
    seen = set()
    stack = [start]
    output = []
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        output.append(node)
        stack.extend(reversed(graph.get(node, [])))
    return output


def dijkstra(graph, start):
    if any(weight < 0 for edges in graph.values() for _, weight in edges):
        raise ValueError("negative edge weight")
    distances = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        distance, node = heappop(heap)
        if distance != distances.get(node):
            continue
        for neighbor, weight in graph.get(node, []):
            candidate = distance + weight
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                heappush(heap, (candidate, neighbor))
    return distances


def topological_sort(graph):
    """Return a topological ordering of a directed acyclic graph.

    Nodes may appear only as neighbors; they are included as sink vertices.
    The input is not modified. Duplicate edges are handled correctly.

    Raises:
        ValueError: If the directed graph contains a cycle.

    Time: O(V + E) | Auxiliary space: O(V).
    """
    indegree = {}
    for node, neighbors in graph.items():
        indegree.setdefault(node, 0)
        for neighbor in neighbors:
            indegree[neighbor] = indegree.get(neighbor, 0) + 1

    ready = deque(
        node for node, degree in indegree.items() if degree == 0
    )
    order = []

    while ready:
        node = ready.popleft()
        order.append(node)
        for neighbor in graph.get(node, ()):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                ready.append(neighbor)

    if len(order) != len(indegree):
        raise ValueError("graph contains a cycle")
    return order
