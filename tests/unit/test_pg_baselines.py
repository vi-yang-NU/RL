"""Unit tests for the Part 2 baselines (rl/pg/baselines.py): construction, arithmetic, learning, and the
statistical properties the theory promises (unbiased gradient, lower variance)."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from rl.pg.baselines import DEFAULT_BASELINE, MeanBaseline, TimeBaseline, ValueBaseline, make_baseline
from rl.pg.envs import make_adapter
from rl.pg.options import Options
from rl.pg.policy import MLP
from rl.pg.reinforce import TrainConfig, policy_gradient_loss, run_episode
from rl.pg.toy import BanditEnv

CPU = torch.device("cpu")


def cfg(**extra) -> TrainConfig:
    return TrainConfig(env="bandit", part=2, gamma=0.95, device="cpu", extra=extra)


class MakeBaselineTests(unittest.TestCase):
    def test_names(self):
        env = BanditEnv((0.0, 1.0))
        self.assertIsNone(make_baseline("none", cfg(), env))
        self.assertIsNone(make_baseline(None, cfg(), env))
        self.assertIsInstance(make_baseline("mean", cfg(), env), MeanBaseline)
        self.assertIsInstance(make_baseline("time", cfg(), env), TimeBaseline)
        self.assertIsInstance(make_baseline("value", cfg(), env), ValueBaseline)
        self.assertIsInstance(make_baseline(DEFAULT_BASELINE, cfg(), env), ValueBaseline)
        with self.assertRaises(ValueError):
            make_baseline("critic", cfg(), env)
        with self.assertRaises(ValueError):
            make_baseline("mean", cfg(baseline_decay=1.0), env)


class MeanBaselineTests(unittest.TestCase):
    def test_exponential_moving_average(self):
        b = MeanBaseline(decay=0.5)
        obs, t = torch.zeros(3, 1), torch.arange(3)
        torch.testing.assert_close(b.predict(obs, t), torch.zeros(3))
        b.update(obs, torch.tensor([1.0, 2.0, 3.0]), t)
        self.assertEqual(b.value, 2.0)  # first update takes the batch mean
        b.update(obs, torch.tensor([4.0, 4.0]), t[:2])
        self.assertEqual(b.value, 0.5 * 2.0 + 0.5 * 4.0)
        torch.testing.assert_close(b.predict(obs, t), torch.full((3,), 3.0))


class TimeBaselineTests(unittest.TestCase):
    def test_running_mean_per_time_step(self):
        b = TimeBaseline()
        obs = torch.zeros(3, 1)
        b.update(obs, torch.tensor([3.0, 2.0, 1.0]), torch.arange(3))
        b.update(obs[:1], torch.tensor([1.0]), torch.arange(1))
        torch.testing.assert_close(b.predict(torch.zeros(4, 1), torch.tensor([0, 1, 2, 5])), torch.tensor([2.0, 2.0, 1.0, 0.0]))


class ValueBaselineTests(unittest.TestCase):
    def test_fits_the_returns(self):
        torch.manual_seed(0)
        env = BanditEnv((0.0, 1.0))
        obs = torch.randn(64, 1)
        returns = 3.0 * obs[:, 0] + 1.0
        first = ValueBaseline(cfg(baseline_hidden=[16]), env)
        self.assertEqual(first.predict(obs, torch.arange(64)).shape, (64,))
        loss0 = first.update(obs, returns, torch.arange(64))["value_loss"]
        trained = ValueBaseline(cfg(baseline_hidden=[16], baseline_epochs=400, baseline_lr=0.02), env)
        loss1 = trained.update(obs, returns, torch.arange(64))["value_loss"]
        self.assertLess(loss1, 0.1 * loss0)
        self.assertLess(torch.mean((trained.predict(obs, torch.arange(64)) - returns) ** 2).item(), 0.1 * loss0)
        self.assertIn("net.0.weight", trained.state_dict())


class UnbiasednessTests(unittest.TestCase):
    def test_baseline_keeps_the_mean_gradient_and_lowers_its_variance(self):
        """Bernoulli bandit, fixed policy: the per-episode gradient estimates with b = 0 and with b = 0.5
        have the same mean (within sampling error) and the baseline halves the variance."""
        torch.manual_seed(0)
        env = BanditEnv((0.3, 0.7), bernoulli=True, seed=0)
        policy = MLP(1, 2, hidden=())
        adapter = make_adapter(cfg(), env)
        with_b = MeanBaseline()
        with_b.value = 0.5
        samples = {0.0: [], 0.5: []}
        env.reset(seed=0)
        for _ in range(1500):
            episode = run_episode(env, adapter, policy, CPU)
            for b, baseline in ((0.0, None), (0.5, with_b)):
                policy.zero_grad()
                loss, _, _ = policy_gradient_loss(policy, [episode], 0.95, baseline, Options(), CPU)
                loss.backward()
                samples[b].append(policy.net[0].bias.grad[1].item())
        g0, g1 = np.array(samples[0.0]), np.array(samples[0.5])
        se = np.sqrt(g0.var(ddof=1) / len(g0) + g1.var(ddof=1) / len(g1))
        self.assertLess(abs(g0.mean() - g1.mean()), 4 * se)
        self.assertLess(g1.var(), 0.75 * g0.var())
        self.assertGreater(abs(g0.mean()), 0)


if __name__ == "__main__":
    unittest.main()
