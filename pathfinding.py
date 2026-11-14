"""A* shortest-path search for weighted adjacency mappings."""

from heapq import heappop, heappush
from itertools import count
import math
from numbers import Real
from typing import Callable, Hashable, Mapping, Sequence, TypeVar

Node = TypeVar("Node", bound=Hashable)


def _finite_non_negative(value: Real, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{description} must be real numbers")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{description} must be finite and non-negative") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{description} must be finite and non-negative")
    return number


def a_star(
    graph: Mapping[Node, Sequence[tuple[Node, Real]]],
    start: Node,
    goal: Node,
    heuristic: Callable[[Node, Node], Real] | None = None,
) -> tuple[float, list[Node]] | None:
    """Return the cheapest (cost, path), or None when the goal is unreachable.

    Edge weights must be finite and non-negative. Supply an admissible
    heuristic for optimal A* behavior; omitting it reduces the search to
    Dijkstra's algorithm. A sequence counter avoids comparing node labels.
    """
    validated_graph = {}
    for node, edges in graph.items():
        validated_edges = []
        for neighbor, weight in edges:
            validated_edges.append(
                (neighbor, _finite_non_negative(weight, "edge weights"))
            )
        validated_graph[node] = validated_edges

    def estimate(node: Node) -> float:
        value = 0 if heuristic is None else heuristic(node, goal)
        return _finite_non_negative(value, "heuristic values")

    sequence = count()
    best_cost: dict[Node, float] = {start: 0.0}
    previous: dict[Node, Node] = {}
    frontier = [(estimate(start), next(sequence), 0.0, start)]

    while frontier:
        _, _, cost, node = heappop(frontier)
        if cost != best_cost.get(node):
            continue
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(previous[path[-1]])
            path.reverse()
            return cost, path

        for neighbor, weight in validated_graph.get(node, ()):
            candidate = cost + weight
            if not math.isfinite(candidate):
                raise ValueError("path cost exceeds finite float range")
            if candidate < best_cost.get(neighbor, math.inf):
                priority = candidate + estimate(neighbor)
                if not math.isfinite(priority):
                    raise ValueError("path priority exceeds finite float range")
                best_cost[neighbor] = candidate
                previous[neighbor] = node
                heappush(frontier, (priority, next(sequence), candidate, neighbor))
    return None
