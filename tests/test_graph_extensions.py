"""Regression tests for graph algorithms beyond traversal and shortest paths."""

import unittest

from graphs import connected_components, dijkstra_path, kruskal_mst, prim_mst


class ConnectedComponentsTests(unittest.TestCase):
    def test_finds_disconnected_components_and_isolated_vertices(self):
        graph = {"a": ["b"], "b": ["a"], "c": [], "d": ["e"], "e": ["d"]}
        components = {frozenset(component) for component in connected_components(graph)}
        self.assertEqual(
            components,
            {frozenset({"a", "b"}), frozenset({"c"}), frozenset({"d", "e"})},
        )

    def test_includes_vertices_present_only_as_neighbors(self):
        self.assertEqual(
            {frozenset(c) for c in connected_components({"source": ["sink"]})},
            {frozenset({"source", "sink"})},
        )

    def test_empty_graph_has_no_components(self):
        self.assertEqual(connected_components({}), [])


class DijkstraPathTests(unittest.TestCase):
    def test_returns_the_shortest_distance_and_path(self):
        graph = {
            "s": [("a", 1), ("b", 4)],
            "a": [("b", 1), ("t", 5)],
            "b": [("t", 1)],
            "t": [],
        }
        self.assertEqual(dijkstra_path(graph, "s", "t"), (3.0, ["s", "a", "b", "t"]))

    def test_handles_same_start_and_unreachable_goal(self):
        self.assertEqual(dijkstra_path({}, "same", "same"), (0.0, ["same"]))
        self.assertIsNone(dijkstra_path({"a": [], "b": []}, "a", "b"))

    def test_handles_incomparable_node_labels_and_rejects_negative_edges(self):
        graph = {"start": [(1, 1), ("other", 1)], 1: [], "other": []}
        self.assertEqual(dijkstra_path(graph, "start", 1), (1.0, ["start", 1]))
        with self.assertRaises(ValueError):
            dijkstra_path({"a": [("b", -1)]}, "a", "b")


class KruskalMSTTests(unittest.TestCase):
    def test_builds_a_minimum_spanning_tree(self):
        nodes = ["a", "b", "c", "d"]
        edges = [
            ("a", "b", 1), ("b", "c", 2), ("c", "d", 3),
            ("a", "d", 10), ("a", "c", 4), ("b", "d", 5),
        ]
        weight, selected = kruskal_mst(nodes, edges)
        self.assertEqual(weight, 6)
        self.assertEqual(len(selected), len(nodes) - 1)
        self.assertEqual(
            {frozenset((u, v)) for u, v, _ in selected},
            {frozenset(("a", "b")), frozenset(("b", "c")), frozenset(("c", "d"))},
        )

    def test_supports_negative_weights_and_empty_graphs(self):
        self.assertEqual(kruskal_mst([], []), (0, []))
        self.assertEqual(kruskal_mst([1, 2], [(1, 2, -3)])[0], -3)

    def test_rejects_disconnected_graphs_and_unknown_endpoints(self):
        with self.assertRaisesRegex(ValueError, "disconnected"):
            kruskal_mst([1, 2, 3], [(1, 2, 1)])
        with self.assertRaisesRegex(ValueError, "unknown node"):
            kruskal_mst([1, 2], [(1, 3, 1)])


class PrimMSTTests(unittest.TestCase):
    def test_builds_a_minimum_spanning_tree(self):
        graph = {
            "a": [("b", 1), ("c", 4)],
            "b": [("a", 1), ("c", 2), ("d", 5)],
            "c": [("a", 4), ("b", 2), ("d", 1)],
            "d": [("b", 5), ("c", 1)],
        }
        weight, selected = prim_mst(graph, start="a")
        self.assertEqual(weight, 4)
        self.assertEqual(len(selected), 3)

    def test_handles_empty_graph_and_rejects_invalid_start(self):
        self.assertEqual(prim_mst({}), (0, []))
        with self.assertRaises(ValueError):
            prim_mst({"a": []}, start="missing")

    def test_rejects_disconnected_graphs(self):
        with self.assertRaisesRegex(ValueError, "disconnected"):
            prim_mst({"a": [], "b": []})


if __name__ == "__main__":
    unittest.main()
