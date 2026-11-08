"""Tests for Floyd-Warshall all-pairs shortest paths."""

import math
import unittest

from graphs import floyd_warshall


class FloydWarshallTests(unittest.TestCase):
    def test_computes_all_pairs_distances(self):
        graph = {
            "a": [("b", 3), ("c", 10)],
            "b": [("c", -2)],
            "c": [("d", 4)],
            "d": [],
        }
        result = floyd_warshall(graph)
        self.assertEqual(result["a"]["d"], 5.0)
        self.assertEqual(result["a"]["c"], 1.0)
        self.assertEqual(result["d"]["a"], math.inf)
        self.assertEqual(result["d"]["d"], 0.0)

    def test_includes_neighbor_only_vertices_and_chooses_lightest_parallel_edge(self):
        result = floyd_warshall({"source": [("sink", 5), ("sink", 2)]})
        self.assertEqual(result["source"]["sink"], 2.0)
        self.assertEqual(result["sink"]["source"], math.inf)

    def test_detects_negative_cycles(self):
        graph = {"a": [("b", 1)], "b": [("c", -3)], "c": [("a", 1)]}
        with self.assertRaisesRegex(ValueError, "negative-weight cycle"):
            floyd_warshall(graph)

    def test_rejects_non_finite_and_non_numeric_weights(self):
        for weight in (float("nan"), float("inf"), True, "2"):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    floyd_warshall({"a": [("b", weight)]})


if __name__ == "__main__":
    unittest.main()
