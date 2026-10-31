"""Unit tests for the policy statistics logged per training episode (entropy, fraction of "right" actions), the
per-step entropy bonus, the per-environment learning-rate defaults and the two-panel training curve."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rl.outputs import ProgressLogger
from rl.params import apply_overrides
from rl.pg.envs import right_index
from rl.pg.options import Options, options_for
from rl.pg.policy import build_policy
from rl.pg.reinforce import Episode, TrainConfig, policy_gradient_loss, run_episode, train
from rl.pg.run import DEFAULT_LR
from rl.pg.toy import BanditEnv
from rl.plotting import plot_training_curve

torch.set_num_threads(2)
CPU = torch.device("cpu")


class LoggingTests(unittest.TestCase):
    def test_episode_entropy_and_action_fraction_are_logged(self):
        env = BanditEnv(means=(0.0, 1.0), seed=0)
        cfg = TrainConfig(env="bandit", part=1, gamma=1.0, episodes=6, device="cpu")
        policy = build_policy(cfg, env)
        with tempfile.TemporaryDirectory() as tmp:
            logger = ProgressLogger(Path(tmp), print_every=10 ** 9)
            train(cfg, env, policy, None, logger)
            records = [json.loads(line) for line in logger.path.read_text().splitlines()]
        self.assertEqual(len(records), 6)
        for r in records:
            self.assertIn("entropy", r)
            self.assertIn("frac_right", r)
            self.assertIn("batch_entropy", r)
            self.assertTrue(0.0 <= r["entropy"] <= math.log(2) + 1e-6)
            self.assertTrue(0.0 <= r["frac_right"] <= 1.0)

    def test_run_episode_records_the_mean_entropy(self):
        env = BanditEnv(means=(0.0, 1.0), seed=0)
        cfg = TrainConfig(env="bandit", part=1, gamma=1.0, device="cpu")
        policy = build_policy(cfg, env)
        from rl.pg.envs import make_adapter

        ep = run_episode(env, make_adapter(cfg, env), policy, CPU, seed=0)
        with torch.no_grad():
            expected = torch.distributions.Categorical(logits=policy(torch.as_tensor(ep.observations))).entropy().mean().item()
        self.assertAlmostEqual(ep.entropy, expected, places=5)
        self.assertTrue(math.isnan(Episode(np.zeros((1, 1), np.float32), np.zeros(1, np.int64), np.ones(1, np.float32)).entropy))

    def test_right_index(self):
        self.assertEqual(right_index("pong", 2), 0)  # RIGHT is the policy's first action (environment action 2)
        self.assertEqual(right_index("cartpole", 2), 1)
        self.assertEqual(right_index("bandit", 3), 1)
        self.assertEqual(right_index("bandit", 1), 0)


class EntropyBonusTests(unittest.TestCase):
    def test_entropy_bonus_is_per_step(self):
        """loss(beta) = loss(0) - beta * sum_t H(pi(.|s_t)) / episodes: the bonus scales with the number of steps."""
        torch.manual_seed(0)
        policy = torch.nn.Linear(3, 2)
        obs = np.random.default_rng(0).normal(size=(7, 3)).astype(np.float32)
        episode = Episode(obs, np.array([0, 1, 1, 0, 1, 0, 0]), np.ones(7, np.float32))
        loss0, _, _ = policy_gradient_loss(policy, [episode], 0.9, None, Options(), CPU)
        beta = 0.5
        loss_b, stats, _ = policy_gradient_loss(policy, [episode], 0.9, None, apply_overrides(Options(), {"entropy_coef": beta}), CPU)
        with torch.no_grad():
            total_entropy = torch.distributions.Categorical(logits=policy(torch.as_tensor(obs))).entropy().sum().item()
        self.assertAlmostEqual(loss_b.item(), loss0.item() - beta * total_entropy, places=5)
        self.assertAlmostEqual(stats["batch_entropy"], total_entropy / 7, places=5)

    def test_defaults_per_environment(self):
        self.assertEqual(DEFAULT_LR, {"cartpole": 0.01, "pong": 1e-4})
        self.assertEqual(options_for(TrainConfig(env="pong", part=1, gamma=0.99)).entropy_coef, 0.01)
        self.assertEqual(options_for(TrainConfig(env="cartpole", part=1, gamma=0.95)).entropy_coef, 0.0)
        self.assertEqual(options_for(TrainConfig(env="pong", part=1, gamma=0.99, extra={"entropy_coef": 0})).entropy_coef, 0.0)


class TrainingCurveTests(unittest.TestCase):
    def test_two_panel_curve(self):
        rng = np.random.default_rng(0)
        n = 300
        rewards = rng.integers(10, 500, n).astype(float)
        entropy = np.linspace(math.log(2), 0.01, n).tolist()
        frac = np.linspace(0.5, 1.0, n).tolist()
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_training_curve(rewards, Path(tmp) / "a.png", "two panels", "subtitle", entropy=entropy, frac_right=frac,
                                       right_label="RIGHT (paddle up)", n_actions=2)
            self.assertGreater(path.stat().st_size, 20_000)
            path = plot_training_curve(rewards, Path(tmp) / "b.png", "one panel")  # the old call still works
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
