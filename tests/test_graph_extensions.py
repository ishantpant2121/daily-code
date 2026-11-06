"""Regression tests for graph algorithms beyond traversal and shortest paths."""

import unittest

from graphs import connected_components


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


if __name__ == "__main__":
    unittest.main()
