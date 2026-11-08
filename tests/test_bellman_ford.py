"""Tests for Bellman-Ford shortest paths."""

import unittest

from graphs import bellman_ford


class BellmanFordTests(unittest.TestCase):
    def test_handles_negative_edges_and_mixed_node_labels(self):
        graph = {
            "start": [(1, 4), ("middle", 5)],
            1: [("middle", -3)],
            "middle": [("end", 2)],
            "end": [],
        }
        self.assertEqual(
            bellman_ford(graph, "start"),
            {"start": 0.0, 1: 4.0, "middle": 1.0, "end": 3.0},
        )

    def test_omits_unreachable_vertices_and_ignores_unreachable_negative_cycles(self):
        graph = {
            "start": [("goal", 2)],
            "goal": [],
            "x": [("y", -2)],
            "y": [("x", 1)],
        }
        self.assertEqual(bellman_ford(graph, "start"), {"start": 0.0, "goal": 2.0})

    def test_detects_only_reachable_negative_cycles(self):
        graph = {"s": [("a", 1)], "a": [("b", -2)], "b": [("a", 1)]}
        with self.assertRaisesRegex(ValueError, "negative-weight cycle"):
            bellman_ford(graph, "s")

    def test_supports_an_absent_start_and_rejects_invalid_weights(self):
        self.assertEqual(bellman_ford({}, "new"), {"new": 0.0})
        for weight in (float("nan"), float("inf"), True):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    bellman_ford({"a": [("b", weight)]}, "a")


if __name__ == "__main__":
    unittest.main()
