"""Unit tests for the episode recorder and viewer (rl/pg/watch.py).

A recording must be the episode the environment and the policy actually produced (the frames are the renders,
Pong's inputs are the handout's preprocess applied to those frames and differenced, the probabilities are the
policy's softmax, the returns are the reward-to-go), survive the round trip through episode.npz, embed into the
page as valid JSON, and draw the first-steps figure.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rl.pg.envs import PONG_ACTIONS, preprocess
from rl.pg.reinforce import discounted_returns
from rl.pg.watch import (ACTIONS, CROP, Recording, first_steps_figure, load_recording, policy_sources, record,
                         record_episode, render_env, render_html, save_recording, untrained, viewer_data)

torch.set_num_threads(2)


class CartPoleRecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = record("cartpole", seed=1)

    def test_shapes_and_alignment(self):
        rec = self.rec
        T = rec.steps
        self.assertGreater(T, 5)
        self.assertEqual(rec.frames.shape, (T + 1, 400, 600, 3))
        self.assertEqual(rec.observations.shape, (T + 1, 4))
        self.assertEqual(rec.actions.shape, (T,))
        self.assertEqual(rec.probs.shape, (T, 2))
        self.assertEqual(rec.rewards.shape, (T,))
        self.assertTrue(np.all(rec.rewards == 1.0))  # CartPole-v1 pays 1 per step
        self.assertEqual(rec.total_reward, T)
        self.assertFalse(rec.truncated)
        self.assertEqual(rec.env_id, "CartPole-v1")

    def test_probabilities_are_the_policys_softmax_and_actions_in_range(self):
        np.testing.assert_allclose(self.rec.probs.sum(axis=1), 1.0, atol=1e-5)
        self.assertTrue(set(np.unique(self.rec.actions)) <= {0, 1})
        np.testing.assert_array_equal(self.rec.env_actions, self.rec.actions)  # CartPole: policy index = env action

    def test_episode_ended_because_a_limit_was_crossed(self):
        x, _, theta, _ = self.rec.observations[-1]
        self.assertTrue(abs(x) > 2.4 or abs(theta) > 12 * np.pi / 180 or self.rec.steps == 500)

    def test_returns_are_reward_to_go(self):
        np.testing.assert_allclose(self.rec.returns(), discounted_returns(self.rec.rewards, self.rec.gamma))
        self.assertEqual(self.rec.gamma, 0.95)

    def test_same_seed_same_episode(self):
        again = record("cartpole", seed=1)
        np.testing.assert_array_equal(again.actions, self.rec.actions)
        np.testing.assert_array_equal(again.observations, self.rec.observations)

    def test_step_cap_truncates(self):
        rec = record("cartpole", seed=1, max_steps=3)
        self.assertEqual(rec.steps, 3)
        self.assertTrue(rec.truncated)
        self.assertEqual(len(rec.frames), 4)

    def test_greedy_takes_the_most_probable_action(self):
        rec = record("cartpole", seed=1, greedy=True)
        np.testing.assert_array_equal(rec.actions, rec.probs.argmax(axis=1))

    def test_viewer_data_and_page(self):
        data = viewer_data(self.rec)
        self.assertEqual(data["steps"], self.rec.steps)
        self.assertEqual(len(data["frames"]), self.rec.steps + 1)
        self.assertTrue(all(f.startswith("data:image/png;base64,") for f in data["frames"]))
        self.assertEqual(data["env_id"], "CartPole-v1")
        self.assertEqual(data["action_short"], list(ACTIONS["cartpole"][1]))
        json.dumps(data, allow_nan=False)
        html = render_html(data, theme="dark")
        self.assertIn('data-theme="dark"', html.split("<head>")[0])
        self.assertIn("const DATA = {", html)
        self.assertNotIn("__DATA__", html)
        short = viewer_data(self.rec, max_steps=4)
        self.assertEqual(short["steps"], 4)
        self.assertTrue(short["cut"])
        self.assertEqual(len(short["frames"]), 5)
        self.assertEqual(short["episode_steps"], self.rec.steps)

    def test_round_trip_and_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_recording(self.rec, Path(tmp) / "episode.npz")
            back = load_recording(path)
            self.assertEqual(back.policy_label, self.rec.policy_label)
            self.assertEqual(back.seed, self.rec.seed)
            np.testing.assert_array_equal(back.frames, self.rec.frames)
            np.testing.assert_array_equal(back.observations, self.rec.observations)
            np.testing.assert_array_equal(back.probs, self.rec.probs)
            self.assertIsNone(back.pre)
            fig = first_steps_figure(self.rec, Path(tmp) / "steps.png", "CartPole-v1 test")
            self.assertGreater(fig.stat().st_size, 10_000)


class PongRecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = record("pong", seed=1, max_steps=60)

    def test_shapes(self):
        rec = self.rec
        self.assertEqual(rec.steps, 60)
        self.assertEqual(rec.frames.shape, (61, 210, 160, 3))
        self.assertEqual(rec.pre.shape, (61, 80, 80))
        self.assertEqual(rec.inputs.shape, (61, 80, 80))
        self.assertEqual(rec.env_id, "ALE/Pong-v5")
        self.assertEqual(rec.gamma, 0.99)

    def test_only_right_and_left_are_sent_to_the_environment(self):
        self.assertTrue(set(np.unique(self.rec.env_actions)) <= set(PONG_ACTIONS))
        np.testing.assert_array_equal(self.rec.env_actions, np.array(PONG_ACTIONS)[self.rec.actions])

    def test_preprocessed_frames_are_the_handouts_preprocess_of_the_frames(self):
        for t in range(0, 61, 10):
            np.testing.assert_array_equal(self.rec.pre[t], preprocess(self.rec.frames[t].copy()).astype(np.uint8))
        self.assertTrue(set(np.unique(self.rec.pre)) <= {0, 1})

    def test_inputs_are_frame_differences(self):
        np.testing.assert_array_equal(self.rec.inputs[0], 0)  # no previous frame at t = 0
        for t in range(1, 61, 7):
            np.testing.assert_array_equal(self.rec.inputs[t], self.rec.pre[t].astype(np.int8) - self.rec.pre[t - 1].astype(np.int8))
        self.assertTrue(set(np.unique(self.rec.inputs)) <= {-1, 0, 1})

    def test_score_tracks_the_points(self):
        score = self.rec.score()
        self.assertEqual(score.shape, (61, 2))
        self.assertEqual(score[-1, 0] - score[-1, 1], self.rec.total_reward)

    def test_viewer_data_has_the_pipeline_images(self):
        data = viewer_data(self.rec)
        self.assertEqual(len(data["pre"]), 61)
        self.assertEqual(len(data["inputs"]), 61)
        self.assertEqual(data["crop"], list(CROP))
        self.assertEqual(data["input_counts"][0], [0, 0])
        self.assertEqual(len(data["score"]), 61)
        json.dumps(data, allow_nan=False)

    def test_round_trip_and_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            back = load_recording(save_recording(self.rec, Path(tmp) / "episode.npz"))
            np.testing.assert_array_equal(back.inputs, self.rec.inputs)
            np.testing.assert_array_equal(back.pre, self.rec.pre)
            self.assertIsNone(back.observations)
            fig = first_steps_figure(self.rec, Path(tmp) / "steps.png", "Pong-v5 test")
            self.assertGreater(fig.stat().st_size, 10_000)


class RecorderPlumbingTests(unittest.TestCase):
    def test_record_episode_matches_run_episode_sampling(self):
        """The recorder draws the same actions as rl.pg.reinforce.run_episode would from the same policy and seed."""
        from rl.pg.envs import make_adapter
        from rl.pg.reinforce import run_episode

        cfg, policy = untrained("cartpole", seed=5)
        env = render_env("cartpole")
        try:
            rec = record_episode(cfg, env, policy, seed=5)
            torch.manual_seed(5)
            plain = run_episode(env, make_adapter(cfg, env), policy, torch.device("cpu"), seed=5)
        finally:
            env.close()
        np.testing.assert_array_equal(rec.actions, plain.actions)
        np.testing.assert_allclose(rec.observations[:-1], plain.observations, atol=1e-6)

    def test_policy_sources_end_with_the_untrained_policy(self):
        sources = policy_sources("pong", 1)
        self.assertEqual(sources[-1]["kind"], "untrained")
        self.assertTrue(all(s["kind"] in ("episode", "checkpoint", "untrained") for s in sources))

    def test_recording_dataclass_properties(self):
        rec = Recording(env="cartpole", policy_label="x", seed=0, gamma=0.5, frames=np.zeros((1, 2, 2, 3), np.uint8),
                        actions=np.zeros(0, np.int16), env_actions=np.zeros(0, np.int16), probs=np.zeros((0, 2)), rewards=np.zeros(0))
        self.assertEqual(rec.steps, 0)
        self.assertEqual(rec.total_reward, 0.0)


if __name__ == "__main__":
    unittest.main()


class PointsTests(unittest.TestCase):
    def test_points_from_rewards(self):
        from rl.pg.watch import points

        pts = points([0, 0, -1, 0, 1, 0, 0, -1])
        self.assertEqual([(p["scorer"], p["step"], p["rally"], p["score"]) for p in pts],
                         [("opponent", 3, 3, [0, 1]), ("agent", 5, 2, [1, 1]), ("opponent", 8, 3, [1, 2])])
        self.assertEqual(points([0, 0, 0]), [])

    def test_viewer_data_carries_seconds_per_step_and_points(self):
        rec = record("pong", seed=1, max_steps=60)
        data = viewer_data(rec)
        self.assertAlmostEqual(data["seconds_per_step"], 4 / 60)
        self.assertIsInstance(data["points"], list)
        self.assertEqual(sum(1 for r in rec.rewards if r != 0), len(data["points"]))
        self.assertAlmostEqual(viewer_data(record("cartpole", seed=1, max_steps=3))["seconds_per_step"], 0.02)
