"""Unit tests for the REINFORCE training loop (rl/pg/reinforce.py) on the toy environments, plus edge cases."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl.dp import solvers
from rl.outputs import ProgressLogger
from rl.pg.baselines import MeanBaseline, make_baseline
from rl.pg.envs import make_adapter
from rl.pg.options import Options, options_for
from rl.pg.policy import MLP, action_probabilities, build_policy
from rl.pg.reinforce import Episode, TrainConfig, discounted_returns, evaluate, policy_gradient_loss, run_episode, train
from rl.pg.toy import BanditEnv, CorridorEnv, corridor_mdp

CPU = torch.device("cpu")


def make_cfg(env: str, **changes: Any) -> TrainConfig:
    base = dict(env=env, part=1, gamma=0.95, lr=0.1, episodes=100, seed=0, device="cpu", extra={})
    base.update(changes)
    return TrainConfig(**base)


def train_in_tempdir(cfg: TrainConfig, env: Any, policy: Any, baseline: Any = None) -> tuple[list, list]:
    with tempfile.TemporaryDirectory() as tmp:
        logger = ProgressLogger(Path(tmp), print_every=10 ** 9)
        rewards = train(cfg, env, policy, baseline, logger)
        lines = [json.loads(line) for line in logger.path.read_text().splitlines()]
    return rewards, lines


class BanditTrainingTests(unittest.TestCase):
    def test_learns_the_best_arm(self):
        # Rewards of both signs: the bad arm is pushed down, the good arm up. (With positive-only rewards and
        # no baseline, REINFORCE + Adam can lock onto whichever arm is sampled first; see tests/test_pg_part2.py.)
        env = BanditEnv((-1.0, 1.0, 0.0))
        cfg = make_cfg("bandit", episodes=150)
        policy = build_policy(cfg, env)
        rewards, lines = train_in_tempdir(cfg, env, policy)
        probs = action_probabilities(policy, torch.ones(1, 1))[0]
        self.assertGreater(float(probs[1]), 0.9)
        self.assertGreater(np.mean(rewards[-20:]), 0.9)
        self.assertEqual(len(rewards), 150)
        self.assertEqual(len(lines), 150)
        self.assertEqual(lines[0]["episode"], 1)
        self.assertIn("loss", lines[0])

    def test_gradient_matches_the_softmax_score_function(self):
        """One step, action a, return G, baseline b: d loss / d logits = -(G - b) (e_a - pi)
        (the score of the softmax, lean/RLVerify/Baseline.lean `softmax_score`)."""
        torch.manual_seed(0)
        policy = MLP(1, 3, hidden=())
        obs = np.ones((1, 1), np.float32)
        episode = Episode(obs, np.array([2]), np.array([2.0], np.float32))
        pi = action_probabilities(policy, torch.ones(1, 1))[0].numpy()
        one_hot = np.eye(3)[2]
        for baseline, b in ((None, 0.0), (MeanBaseline(), 0.75)):
            if baseline is not None:
                baseline.value = b
            policy.zero_grad()
            loss, stats, _ = policy_gradient_loss(policy, [episode], 0.95, baseline, Options(), CPU)
            loss.backward()
            np.testing.assert_allclose(policy.net[0].bias.grad.numpy(), -(2.0 - b) * (one_hot - pi), atol=1e-6)
            np.testing.assert_allclose(policy.net[0].weight.grad.numpy()[:, 0], -(2.0 - b) * (one_hot - pi), atol=1e-6)
            self.assertAlmostEqual(stats["advantage_mean"], 2.0 - b, places=6)

    def test_training_is_reproducible(self):
        results = []
        for _ in range(2):
            env = BanditEnv((0.3, 0.7), bernoulli=True)
            cfg = make_cfg("bandit", episodes=40, seed=3)
            rewards, _ = train_in_tempdir(cfg, env, build_policy(cfg, env))
            results.append(rewards)
        self.assertEqual(results[0], results[1])


class CorridorTrainingTests(unittest.TestCase):
    def test_reaches_the_optimum_computed_by_dynamic_programming(self):
        length, gamma = 5, 0.9
        env = CorridorEnv(length=length, slip=0.0, max_steps=30)
        cfg = make_cfg("corridor", gamma=gamma, episodes=300, lr=0.05, extra={"eval_greedy": True})
        policy = build_policy(cfg, env)
        rewards, _ = train_in_tempdir(cfg, env, policy)
        self.assertGreater(np.mean(rewards[-30:]), 0.95)
        optimal = -solvers.value_iteration(corridor_mdp(length, 0.0, gamma), tol=1e-12).values[0]
        self.assertAlmostEqual(optimal, gamma ** (length - 2), places=10)
        adapter = make_adapter(cfg, env)
        episode = run_episode(env, adapter, policy, CPU, greedy=True)
        self.assertEqual(episode.length, length - 1)
        self.assertAlmostEqual(discounted_returns(episode.rewards, gamma)[0], optimal, places=6)
        self.assertEqual(evaluate(cfg, env, policy, 7), [1.0] * 7)

    def test_gamma_zero_credits_only_the_immediate_reward(self):
        torch.manual_seed(0)
        policy = MLP(3, 2, hidden=(8,))
        obs = np.eye(3, dtype=np.float32)
        long = Episode(obs, np.array([1, 1, 1]), np.array([0.0, 0.0, 1.0], np.float32))
        last = Episode(obs[2:], np.array([1]), np.array([1.0], np.float32))
        grads = []
        for episode in (long, last):
            policy.zero_grad()
            loss, _, _ = policy_gradient_loss(policy, [episode], 0.0, None, Options(), CPU)
            loss.backward()
            grads.append(torch.cat([p.grad.flatten() for p in policy.parameters()]).clone())
        torch.testing.assert_close(grads[0], grads[1])
        self.assertGreater(float(grads[0].abs().sum()), 0.0)


class EdgeCaseTests(unittest.TestCase):
    def test_constant_returns_with_normalisation_do_not_produce_nan(self):
        policy = MLP(1, 2, hidden=())
        episode = Episode(np.ones((4, 1), np.float32), np.array([0, 1, 0, 1]), np.ones(4, np.float32))
        # gamma = 0 makes every return equal to its (constant) reward, so the advantages have zero spread
        loss, stats, _ = policy_gradient_loss(policy, [episode], 0.0, None, Options(normalize=True), CPU)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.parameters()))
        self.assertEqual(stats["advantage_std"], 0.0)

    def test_perfect_baseline_gives_zero_gradient(self):
        env = BanditEnv((1.0, 1.0))  # every episode returns exactly 1
        cfg = make_cfg("bandit", part=2, baseline="mean", episodes=2, extra={"baseline_decay": 0.0})
        policy = build_policy(cfg, env)
        baseline = make_baseline("mean", cfg, env)
        train_in_tempdir(make_cfg("bandit", part=2, baseline="mean", episodes=1, extra={"baseline_decay": 0.0}), env, policy, baseline)
        self.assertEqual(baseline.value, 1.0)
        before = [p.detach().clone() for p in policy.parameters()]
        rewards, lines = train_in_tempdir(cfg, env, policy, baseline)
        for p, q in zip(before, policy.parameters()):
            torch.testing.assert_close(p, q.detach())
        self.assertEqual(lines[-1]["grad_norm"], 0.0)

    def test_truncation_and_step_caps(self):
        env = CorridorEnv(length=6, max_steps=3)
        policy = MLP(6, 2, hidden=())
        with torch.no_grad():
            policy.net[0].bias.copy_(torch.tensor([10.0, -10.0]))  # always go left
        adapter = make_adapter(make_cfg("corridor"), env)
        episode = run_episode(env, adapter, policy, CPU, greedy=True)
        self.assertEqual(episode.length, 3)  # truncated by the environment
        self.assertEqual(episode.total_reward, 0.0)
        with self.assertRaises(RuntimeError):
            env.step(0)  # stepping a finished episode is an error in the toy environment
        capped = run_episode(env, adapter, policy, CPU, greedy=True, max_steps=2)
        self.assertEqual(capped.length, 2)  # cut short by the max_steps option, the environment is not done

    def test_batch_larger_than_the_run_still_updates_once(self):
        env = BanditEnv((0.0, 1.0))
        cfg = make_cfg("bandit", episodes=3, extra={"batch_episodes": 10})
        policy = build_policy(cfg, env)
        before = [p.detach().clone() for p in policy.parameters()]
        rewards, lines = train_in_tempdir(cfg, env, policy)
        self.assertEqual(len(rewards), 3)
        self.assertNotIn("loss", lines[0])
        self.assertIn("loss", lines[-1])
        self.assertTrue(any(not torch.equal(p, q.detach()) for p, q in zip(before, policy.parameters())))

    def test_empty_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            policy_gradient_loss(MLP(1, 2), [], 0.9, None, Options(), CPU)

    def test_evaluation_count_and_greedy_determinism(self):
        env = CorridorEnv(length=4, slip=0.3, max_steps=20)
        cfg = make_cfg("corridor", extra={"eval_greedy": True})
        policy = build_policy(cfg, env)
        first = evaluate(cfg, env, policy, 5)
        second = evaluate(cfg, env, policy, 5)
        self.assertEqual(len(first), 5)
        self.assertEqual(first, second)
        sampled = evaluate(make_cfg("corridor"), env, policy, 4)
        self.assertEqual(len(sampled), 4)

    def test_gradient_clipping_and_entropy_options_run(self):
        env = BanditEnv((0.0, 1.0))
        cfg = make_cfg("bandit", episodes=5, extra={"max_grad_norm": 0.5, "entropy_coef": 0.01, "discount_grad": True, "reward_to_go": False})
        rewards, lines = train_in_tempdir(cfg, env, build_policy(cfg, env))
        self.assertEqual(len(rewards), 5)
        self.assertLessEqual(lines[-1]["grad_norm"], 0.5 + 1e-6)
        self.assertEqual(options_for(cfg).max_grad_norm, 0.5)


if __name__ == "__main__":
    unittest.main()
