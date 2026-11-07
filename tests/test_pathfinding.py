"""Tests for A* pathfinding."""

import unittest

from pathfinding import a_star


class AStarTests(unittest.TestCase):
    def test_finds_lowest_cost_path_with_heuristic(self):
        graph = {
            "s": [("a", 1), ("b", 4)],
            "a": [("b", 1), ("t", 7)],
            "b": [("t", 2)],
            "t": [],
        }
        distances = {"s": 3, "a": 2, "b": 1, "t": 0}
        self.assertEqual(
            a_star(graph, "s", "t", lambda node, goal: distances[node]),
            (4.0, ["s", "a", "b", "t"]),
        )

    def test_zero_heuristic_matches_dijkstra_and_handles_incomparable_nodes(self):
        graph = {
            "start": [(1, 2), ("other", 2)],
            1: [("end", 1)],
            "other": [("end", 1)],
            "end": [],
        }
        self.assertEqual(a_star(graph, "start", "end"), (3.0, ["start", 1, "end"]))

    def test_handles_same_start_and_unreachable_goal(self):
        self.assertEqual(a_star({}, "x", "x"), (0.0, ["x"]))
        self.assertIsNone(a_star({"x": [], "y": []}, "x", "y"))

    def test_rejects_negative_or_non_finite_edge_weights(self):
        for weight in (-1, float("inf"), float("nan"), True):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    a_star({"a": [("b", weight)]}, "a", "b")

    def test_rejects_invalid_heuristics(self):
        heuristics = (
            lambda node, goal: -1,
            lambda node, goal: float("nan"),
            lambda node, goal: "0",
        )
        for heuristic in heuristics:
            with self.subTest(heuristic=heuristic):
                with self.assertRaises(ValueError):
                    a_star({"a": [("b", 1)], "b": []}, "a", "b", heuristic)


if __name__ == "__main__":
    unittest.main()
