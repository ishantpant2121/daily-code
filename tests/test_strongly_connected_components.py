"""Tests for iterative strongly connected components."""

import unittest

from graphs import strongly_connected_components


class StronglyConnectedComponentsTests(unittest.TestCase):
    def test_finds_components_in_a_directed_graph(self):
        graph = {
            "a": ["b"],
            "b": ["c", "e", "f"],
            "c": ["d", "g"],
            "d": ["c", "h"],
            "e": ["a", "f"],
            "f": ["g"],
            "g": ["f"],
            "h": ["d", "g"],
            "isolated": [],
        }
        components = {
            frozenset(component)
            for component in strongly_connected_components(graph)
        }
        self.assertEqual(
            components,
            {
                frozenset({"a", "b", "e"}),
                frozenset({"c", "d", "h"}),
                frozenset({"f", "g"}),
                frozenset({"isolated"}),
            },
        )

    def test_includes_vertices_that_only_appear_as_neighbors(self):
        components = strongly_connected_components({"source": ["sink"]})
        self.assertEqual(
            {frozenset(component) for component in components},
            {frozenset({"source"}), frozenset({"sink"})},
        )

    def test_handles_long_chains_without_recursion(self):
        graph = {index: [index + 1] for index in range(2500)}
        graph[2500] = []
        components = strongly_connected_components(graph)
        self.assertEqual(len(components), 2501)
        self.assertTrue(all(len(component) == 1 for component in components))

    def test_empty_graph(self):
        self.assertEqual(strongly_connected_components({}), [])


if __name__ == "__main__":
    unittest.main()
