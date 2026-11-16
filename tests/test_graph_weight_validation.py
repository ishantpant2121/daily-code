"""Tests for strict weighted-edge validation in graph algorithms."""

import unittest

from graphs import bellman_ford, dijkstra, dijkstra_path, floyd_warshall, kruskal_mst, prim_mst


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


    def test_shortest_path_algorithms_reject_unrepresentable_accumulated_costs(self):
        chain = {"s": [("a", 1e308)], "a": [("b", 1e308)], "b": []}
        with self.assertRaisesRegex(OverflowError, "finite floating-point range"):
            dijkstra(chain, "s")
        with self.assertRaisesRegex(OverflowError, "finite floating-point range"):
            dijkstra_path(chain, "s", "b")
        with self.assertRaisesRegex(OverflowError, "finite floating-point range"):
            bellman_ford(chain, "s")
        with self.assertRaisesRegex(OverflowError, "finite floating-point range"):
            floyd_warshall(chain)

    def test_bellman_ford_rejects_negative_accumulation_underflow(self):
        chain = {"s": [("a", -1e308)], "a": [("b", -1e308)], "b": []}
        with self.assertRaisesRegex(OverflowError, "finite floating-point range"):
            bellman_ford(chain, "s")


if __name__ == "__main__":
    unittest.main()
