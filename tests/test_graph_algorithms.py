"""Tests for directed-graph topological ordering."""

import unittest

from graphs import topological_sort


class TopologicalSortTests(unittest.TestCase):
    def assert_valid_order(self, graph, order):
        nodes = set(graph)
        for neighbors in graph.values():
            nodes.update(neighbors)
        self.assertEqual(set(order), nodes)
        self.assertEqual(len(order), len(nodes))

        positions = {node: index for index, node in enumerate(order)}
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                self.assertLess(positions[node], positions[neighbor])

    def test_orders_a_dag_with_multiple_sources_and_sink_only_nodes(self):
        graph = {
            "shop": ["cook"],
            "prepare": ["cook"],
            "cook": ["eat"],
        }
        order = topological_sort(graph)
        self.assert_valid_order(graph, order)

    def test_handles_disconnected_and_isolated_components(self):
        graph = {"a": ["b"], "c": [], "d": ["e"]}
        order = topological_sort(graph)
        self.assert_valid_order(graph, order)

    def test_handles_duplicate_edges(self):
        graph = {"source": ["sink", "sink"], "sink": []}
        self.assert_valid_order(graph, topological_sort(graph))

    def test_empty_graph(self):
        self.assertEqual(topological_sort({}), [])

    def test_rejects_cycles_and_self_loops(self):
        for graph in (
            {"a": ["b"], "b": ["a"]},
            {"a": ["a"]},
            {"a": ["b"], "b": ["c"], "c": ["a"]},
        ):
            with self.subTest(graph=graph):
                with self.assertRaisesRegex(ValueError, "cycle"):
                    topological_sort(graph)


if __name__ == "__main__":
    unittest.main()
