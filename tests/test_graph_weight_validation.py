"""Tests for strict weighted-edge validation in graph algorithms."""

import unittest

from graphs import dijkstra, dijkstra_path, kruskal_mst, prim_mst


class GraphWeightValidationTests(unittest.TestCase):
    def test_dijkstra_rejects_negative_non_finite_and_non_numeric_weights(self):
        invalid = (-1, True, float("nan"), float("inf"), "2", 10 ** 10000)
        for index, weight in enumerate(invalid):
            with self.subTest(index=index, value_type=type(weight).__name__):
                graph = {"a": [("b", weight)], "b": []}
                with self.assertRaises(ValueError):
                    dijkstra(graph, "a")
                with self.assertRaises(ValueError):
                    dijkstra_path(graph, "a", "b")

    def test_dijkstra_validates_unreachable_edges_too(self):
        graph = {"start": [], "isolated": [("other", float("nan"))]}
        with self.assertRaises(ValueError):
            dijkstra(graph, "start")

    def test_mst_algorithms_reject_non_finite_and_non_numeric_weights(self):
        invalid = (True, float("nan"), float("-inf"), "1")
        for index, weight in enumerate(invalid):
            with self.subTest(index=index, value_type=type(weight).__name__):
                with self.assertRaises(ValueError):
                    kruskal_mst([1, 2], [(1, 2, weight)])
                with self.assertRaises(ValueError):
                    prim_mst({1: [(2, weight)], 2: [(1, weight)]})

    def test_mst_algorithms_continue_to_support_finite_negative_weights(self):
        self.assertEqual(
            kruskal_mst(["a", "b"], [("a", "b", -2)])[0],
            -2.0,
        )
        self.assertEqual(
            prim_mst({"a": [("b", -2)], "b": [("a", -2)]})[0],
            -2.0,
        )


if __name__ == "__main__":
    unittest.main()
