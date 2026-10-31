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


class ComparisonAndLayoutTests(unittest.TestCase):
    def test_baseline_comparison_is_drawn(self):
        from rl.plotting import plot_baseline_comparison

        rng = np.random.default_rng(1)
        runs = [{"label": "Part 1, no baseline", "rewards": rng.integers(10, 500, 300), "weights": rng.uniform(10, 20, 300)},
                {"label": "Part 2, value baseline V(s)", "rewards": rng.integers(10, 500, 300), "weights": rng.uniform(1, 5, 300)}]
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_baseline_comparison(runs, Path(tmp) / "c.png", "title", "subtitle", note="note")
            self.assertGreater(path.stat().st_size, 20_000)
            path = plot_baseline_comparison([{**run, "weights": None} for run in runs], Path(tmp) / "d.png", "no weights")
            self.assertTrue(path.exists())

    def test_compare_parts_needs_both_runs(self):
        import json
        from unittest import mock

        import rl.outputs
        import rl.pg.run as run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pg_dir = lambda part, env, tag=None: root / "pg" / f"part{part}_{env}"  # noqa: E731
            with mock.patch.object(run, "pg_dir", pg_dir), mock.patch.object(rl.outputs, "RESULTS", root):
                self.assertIsNone(run.compare_parts("cartpole"))
                for part, baseline in ((1, "none"), (2, "value")):
                    d = pg_dir(part, "cartpole")
                    d.mkdir(parents=True)
                    (d / "config.json").write_text(json.dumps({"gamma": 0.95, "lr": 0.01, "seed": 0, "episodes": 3,
                                                               "extra": {}, "baseline": baseline}))
                    (d / "train.json").write_text("{}")
                    (d / "progress.jsonl").write_text("".join(
                        json.dumps({"episode": i, "reward": 10.0 * i, "advantage_mean": 3.0, "advantage_std": 4.0}) + "\n"
                        for i in range(1, 4)))
                path = run.compare_parts("cartpole")
                self.assertEqual(path, root / "pg" / "compare_cartpole.png")
                self.assertTrue(path.exists())

    def test_few_values_and_single_value_histograms(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = [-21] * 213 + [-20] * 202 + [-19] * 80 + [-18] * 5
            path, mean, std = plot_reward_histogram(r, Path(tmp) / "e.png", "four values", value_range=(-21, 21), integer=True,
                                                    length_text="length: mean 1050 steps")
            self.assertAlmostEqual(mean, np.mean(r))
            self.assertTrue(path.exists())
