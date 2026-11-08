"""BFS, DFS, shortest paths, and topological ordering."""

from collections import deque
from heapq import heappop, heappush
from itertools import count
import math
from numbers import Real

from union_find import DisjointSet


def _validated_weight(value, *, allow_negative):
    """Convert an edge weight to a finite float and enforce its sign policy."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("edge weights must be real numbers")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError("edge weights must be finite real numbers") from None
    if not math.isfinite(number):
        raise ValueError("edge weights must be finite real numbers")
    if not allow_negative and number < 0:
        raise ValueError("edge weights must be non-negative")
    return number


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
    validated_graph = {
        node: [
            (neighbor, _validated_weight(weight, allow_negative=False))
            for neighbor, weight in edges
        ]
        for node, edges in graph.items()
    }
    sequence = count()
    distances = {start: 0.0}
    heap = [(0.0, next(sequence), start)]
    while heap:
        distance, _, node = heappop(heap)
        if distance != distances.get(node):
            continue
        for neighbor, weight in validated_graph.get(node, ()):
            candidate = distance + weight
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                heappush(heap, (candidate, next(sequence), neighbor))
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

    Edge weights must be finite, real, and non-negative. A sequence counter
    keeps heap entries comparable even when node labels use different types.
    """
    validated_graph = {
        node: [
            (neighbor, _validated_weight(weight, allow_negative=False))
            for neighbor, weight in edges
        ]
        for node, edges in graph.items()
    }
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

        for neighbor, weight in validated_graph.get(node, ()):
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
    are (source, destination, weight) tuples. Finite negative weights are
    supported. Unknown endpoints and disconnected graphs raise ValueError.
    """
    nodes = list(dict.fromkeys(nodes))
    edges = [
        (source, destination, _validated_weight(weight, allow_negative=True))
        for source, destination, weight in edges
    ]
    positions = {node: index for index, node in enumerate(nodes)}
    for source, destination, _ in edges:
        if source not in positions or destination not in positions:
            raise ValueError("edge references an unknown node")

    if not nodes:
        return 0, []

    sets = DisjointSet(len(nodes))
    selected = []
    total_weight = 0.0
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

    The graph maps each node to (neighbor, weight) pairs. Finite negative
    weights are supported. A sequence counter prevents heap comparisons
    between unrelated node-label types.
    """
    adjacency = {
        node: [
            (neighbor, _validated_weight(weight, allow_negative=True))
            for neighbor, weight in neighbors
        ]
        for node, neighbors in graph.items()
    }
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
    total_weight = 0.0
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


def bellman_ford(graph, start):
    """Return shortest distances from start, supporting negative edge weights.

    Unreachable vertices are omitted, matching dijkstra(). A reachable
    negative-weight cycle raises ValueError. Edge weights must be finite real
    numbers. Time: O(V * E); auxiliary space: O(V + E).
    """
    nodes = list(graph)
    known = set(nodes)
    edges = []
    for source, neighbors in graph.items():
        for destination, weight in neighbors:
            if destination not in known:
                known.add(destination)
                nodes.append(destination)
            edges.append(
                (source, destination, _validated_weight(weight, allow_negative=True))
            )
    if start not in known:
        nodes.append(start)
        known.add(start)

    distances = {start: 0.0}
    for _ in range(max(0, len(nodes) - 1)):
        updated = False
        for source, destination, weight in edges:
            if source not in distances:
                continue
            candidate = distances[source] + weight
            if candidate < distances.get(destination, float("inf")):
                distances[destination] = candidate
                updated = True
        if not updated:
            break

    for source, destination, weight in edges:
        if (
            source in distances
            and distances[source] + weight < distances.get(destination, float("inf"))
        ):
            raise ValueError("graph contains a reachable negative-weight cycle")
    return distances


def floyd_warshall(graph):
    """Return all-pairs shortest distances for a directed weighted graph.

    The result includes every vertex pair; unreachable pairs have infinity.
    Finite negative edges are supported, but any negative cycle raises
    ValueError. Node order follows first appearance in the input mapping and
    its neighbor lists. Time: O(V^3); space: O(V^2).
    """
    nodes = list(graph)
    known = set(nodes)
    for neighbors in graph.values():
        for neighbor, _ in neighbors:
            if neighbor not in known:
                known.add(neighbor)
                nodes.append(neighbor)

    distances = {
        source: {destination: float("inf") for destination in nodes}
        for source in nodes
    }
    for node in nodes:
        distances[node][node] = 0.0

    for source, neighbors in graph.items():
        for destination, weight in neighbors:
            numeric_weight = _validated_weight(weight, allow_negative=True)
            if numeric_weight < distances[source][destination]:
                distances[source][destination] = numeric_weight

    for middle in nodes:
        for source in nodes:
            first_leg = distances[source][middle]
            if math.isinf(first_leg):
                continue
            for destination in nodes:
                second_leg = distances[middle][destination]
                if math.isinf(second_leg):
                    continue
                candidate = first_leg + second_leg
                if candidate < distances[source][destination]:
                    distances[source][destination] = candidate

    if any(distances[node][node] < 0 for node in nodes):
        raise ValueError("graph contains a negative-weight cycle")
    return distances
