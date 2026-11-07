"""A* shortest-path search for weighted adjacency mappings."""

from heapq import heappop, heappush
from itertools import count
import math
from numbers import Real
from typing import Callable, Hashable, Mapping, Sequence, TypeVar

Node = TypeVar("Node", bound=Hashable)


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
    for edges in graph.values():
        for _, weight in edges:
            if isinstance(weight, bool) or not isinstance(weight, Real):
                raise ValueError("edge weights must be real numbers")
            if not math.isfinite(float(weight)) or weight < 0:
                raise ValueError("edge weights must be finite and non-negative")

    def estimate(node: Node) -> float:
        value = 0 if heuristic is None else heuristic(node, goal)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("heuristic values must be real numbers")
        result = float(value)
        if not math.isfinite(result) or result < 0:
            raise ValueError("heuristic values must be finite and non-negative")
        return result

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

        for neighbor, weight in graph.get(node, ()):
            candidate = cost + float(weight)
            if candidate < best_cost.get(neighbor, math.inf):
                best_cost[neighbor] = candidate
                previous[neighbor] = node
                heappush(frontier, (candidate + estimate(neighbor), next(sequence), candidate, neighbor))
    return None
