"""Regression tests for graph algorithms beyond traversal and shortest paths."""

import unittest

from graphs import connected_components, dijkstra_path


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


if __name__ == "__main__":
    unittest.main()
