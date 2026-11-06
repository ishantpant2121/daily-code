"""BFS, DFS, shortest paths, and topological ordering."""

from collections import deque
from itertools import count
from heapq import heappop, heappush

from union_find import DisjointSet


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



def connected_components(graph):
    """Return connected components of an undirected adjacency mapping.

    Vertices that occur only in neighbor lists are included as well.
    """
    nodes = list(graph)
    known = set(nodes)
    for neighbors in graph.values():
        for neighbor in neighbors:
            if neighbor not in known:
                known.add(neighbor)
                nodes.append(neighbor)

    visited = set()
    components = []
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [start]
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in graph.get(node, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components



def dijkstra_path(graph, start, goal):
    """Return (distance, path) for the cheapest route, or None if unreachable.

    Edge weights must be non-negative. A sequence counter keeps heap entries
    comparable even when node labels use different Python types.
    """
    if any(weight < 0 for edges in graph.values() for _, weight in edges):
        raise ValueError("negative edge weight")

    sequence = count()
    distances = {start: 0.0}
    previous = {}
    heap = [(0.0, next(sequence), start)]

    while heap:
        distance, _, node = heappop(heap)
        if distance != distances.get(node):
            continue
        if node == goal:
            break

        for neighbor, weight in graph.get(node, ()):
            candidate = distance + weight
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                previous[neighbor] = node
                heappush(heap, (candidate, next(sequence), neighbor))

    if goal not in distances:
        return None

    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return distances[goal], path



def kruskal_mst(nodes, edges):
    """Return (total_weight, selected_edges) for a connected weighted graph.

    Nodes are supplied explicitly so isolated vertices are not lost. Edges
    are (source, destination, weight) tuples. Negative weights are supported.
    Raises ValueError when an edge references an unknown node or the graph is
    disconnected.
    """
    nodes = list(dict.fromkeys(nodes))
    edges = list(edges)
    positions = {node: index for index, node in enumerate(nodes)}
    for source, destination, _ in edges:
        if source not in positions or destination not in positions:
            raise ValueError("edge references an unknown node")

    if not nodes:
        return 0, []

    sets = DisjointSet(len(nodes))
    selected = []
    total_weight = 0
    for source, destination, weight in sorted(edges, key=lambda edge: edge[2]):
        if source == destination:
            continue
        if sets.union(positions[source], positions[destination]):
            selected.append((source, destination, weight))
            total_weight += weight
            if len(selected) == len(nodes) - 1:
                break

    if len(selected) != len(nodes) - 1:
        raise ValueError("graph is disconnected")
    return total_weight, selected



def prim_mst(graph, start=None):
    """Return (total_weight, selected_edges) for an undirected connected graph.

    The graph maps each node to (neighbor, weight) pairs. A sequence counter
    prevents heap comparisons between unrelated node-label types. Raises
    ValueError for an unknown start node or a disconnected graph.
    """
    adjacency = {node: list(neighbors) for node, neighbors in graph.items()}
    nodes = list(adjacency)
    known = set(nodes)
    for neighbors in adjacency.values():
        for neighbor, _ in neighbors:
            if neighbor not in known:
                known.add(neighbor)
                nodes.append(neighbor)

    if not nodes:
        return 0, []
    if start is None:
        start = nodes[0]
    if start not in known:
        raise ValueError("start node is not in the graph")

    sequence = count()
    visited = {start}
    heap = []
    for neighbor, weight in adjacency.get(start, ()):
        heappush(heap, (weight, next(sequence), start, neighbor))

    selected = []
    total_weight = 0
    while heap and len(visited) < len(nodes):
        weight, _, source, destination = heappop(heap)
        if destination in visited:
            continue
        visited.add(destination)
        selected.append((source, destination, weight))
        total_weight += weight
        for neighbor, edge_weight in adjacency.get(destination, ()):
            if neighbor not in visited:
                heappush(
                    heap,
                    (edge_weight, next(sequence), destination, neighbor),
                )

    if len(visited) != len(nodes):
        raise ValueError("graph is disconnected")
    return total_weight, selected
