"""Unit tests for the evaluation histogram (rl/plotting.py): statistics, bins and the degenerate case."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl.plotting import _integer_edges, plot_reward_histogram, reward_stats


class RewardStatsTests(unittest.TestCase):
    def test_stats(self):
        s = reward_stats([1, 2, 2, 5])
        self.assertEqual((s["episodes"], s["min"], s["max"], s["median"], s["distinct"]), (4, 1.0, 5.0, 2.0, 3))
        self.assertEqual(s["most_common"], [2.0, 2])
        self.assertAlmostEqual(s["mean"], 2.5)

    def test_integer_edges_one_bin_per_integer_for_short_ranges(self):
        edges = _integer_edges(-21, 21)
        self.assertEqual(len(edges), 44)
        self.assertEqual((edges[0], edges[-1]), (-21.5, 21.5))
        np.testing.assert_allclose(np.diff(edges), 1.0)

    def test_integer_edges_round_width_for_long_ranges(self):
        edges = _integer_edges(0, 500)
        self.assertEqual(edges[0], -0.5)
        self.assertGreaterEqual(edges[-1], 500.5)
        self.assertLessEqual(len(edges) - 1, 26)
        self.assertTrue(np.allclose(np.diff(edges)[:-1], 20.0))


class HistogramTests(unittest.TestCase):
    def test_degenerate_and_spread_histograms_are_drawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, mean, std = plot_reward_histogram([500.0] * 500, Path(tmp) / "a.png", "all the same", value_range=(0, 500),
                                                    integer=True, note="every episode reached the cap")
            self.assertEqual((mean, std), (500.0, 0.0))
            self.assertGreater(path.stat().st_size, 20_000)
            rng = np.random.default_rng(0)
            path, mean, std = plot_reward_histogram(rng.integers(-21, 22, 500), Path(tmp) / "b.png", "spread", value_range=(-21, 21), integer=True)
            self.assertGreater(std, 0)
            self.assertTrue(path.exists())
            path, _, _ = plot_reward_histogram(rng.normal(10, 2, 200), Path(tmp) / "c.png", "no range")  # old call style
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()


class LengthTextTests(unittest.TestCase):
    def test_histogram_with_a_length_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _, _ = plot_reward_histogram([-21.0] * 20, Path(tmp) / "d.png", "lengths", value_range=(-21, 21), integer=True,
                                               length_text="every game lasted 764 steps (51 s of game time)")
            self.assertTrue(path.exists())

    def test_run_helpers_for_lengths(self):
        from rl.pg.run import eval_note, game_time, length_summary

        self.assertEqual(game_time("pong", 764), "51 s")
        self.assertEqual(game_time("cartpole", 500), "10 s")
        self.assertEqual(game_time("pong", 3000), "3 min 20 s")
        fields, text = length_summary("pong", [764] * 5)
        self.assertEqual((fields["mean_length"], fields["min_length"], fields["max_length"]), (764.0, 764, 764))
        self.assertIn("764 steps", text)
        self.assertIn("Each game lasted 764 steps", eval_note("pong", [-21.0] * 5, [764] * 5))
        self.assertIsNone(eval_note("pong", [-21.0, -20.0], [764, 800]))
