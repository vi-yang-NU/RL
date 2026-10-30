"""Unit tests for the toy environments used to validate the policy-gradient code (rl/pg/toy.py)."""
from __future__ import annotations

import unittest

import numpy as np

from rl.dp import solvers
from rl.pg.toy import BanditEnv, CorridorEnv, corridor_mdp, make_toy_env


class BanditEnvTests(unittest.TestCase):
    def test_one_step_episodes_and_rewards(self):
        env = BanditEnv((0.2, 0.9, 0.5))
        obs, info = env.reset(seed=0)
        np.testing.assert_array_equal(obs, [1.0])
        self.assertEqual(env.best_arm, 1)
        obs, reward, terminated, truncated, _ = env.step(1)
        self.assertEqual((reward, terminated, truncated), (0.9, True, False))
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(3)

    def test_bernoulli_arms_have_the_right_frequencies(self):
        env = BanditEnv((0.3, 0.7), bernoulli=True, seed=1)
        env.reset(seed=1)
        wins = []
        for _ in range(4000):
            env.reset()
            wins.append(env.step(1)[1])
        self.assertAlmostEqual(float(np.mean(wins)), 0.7, delta=0.03)
        self.assertTrue(set(wins) <= {0.0, 1.0})

    def test_invalid_arms(self):
        with self.assertRaises(ValueError):
            BanditEnv((1.0,))
        with self.assertRaises(ValueError):
            BanditEnv((0.5, 1.5), bernoulli=True)
        with self.assertRaises(ValueError):
            make_toy_env("maze")


class CorridorEnvTests(unittest.TestCase):
    def test_walk_to_the_goal(self):
        env = CorridorEnv(length=4, max_steps=10)
        obs, _ = env.reset(seed=0)
        np.testing.assert_array_equal(obs, [1, 0, 0, 0])
        obs, r, term, trunc, _ = env.step(0)  # left into the wall: stays at 0
        np.testing.assert_array_equal(obs, [1, 0, 0, 0])
        self.assertEqual((r, term, trunc), (0.0, False, False))
        env.step(1)
        env.step(1)
        obs, r, term, trunc, _ = env.step(1)
        np.testing.assert_array_equal(obs, [0, 0, 0, 1])
        self.assertEqual((r, term, trunc), (1.0, True, False))
        with self.assertRaises(RuntimeError):
            env.step(1)

    def test_truncation_and_step_rewards(self):
        env = CorridorEnv(length=5, max_steps=3, step_reward=-0.05, right_reward=-0.1)
        env.reset(seed=0)
        self.assertAlmostEqual(env.step(0)[1], -0.05)
        self.assertAlmostEqual(env.step(1)[1], -0.15)
        obs, r, term, trunc, _ = env.step(1)
        self.assertEqual((term, trunc), (False, True))
        with self.assertRaises(ValueError):
            CorridorEnv(length=1)
        with self.assertRaises(ValueError):
            CorridorEnv(slip=1.5)

    def test_slip_reverses_moves_with_the_given_probability(self):
        env = CorridorEnv(length=50, slip=0.25, max_steps=10 ** 6)
        env.reset(seed=3)
        env.state = 25
        reversed_moves = 0
        for _ in range(4000):
            before = env.state
            env.step(1)
            reversed_moves += env.state < before
            env.state = 25
        self.assertAlmostEqual(reversed_moves / 4000, 0.25, delta=0.03)


class CorridorMdpTests(unittest.TestCase):
    def test_mdp_matches_the_environment(self):
        for slip, right_reward in ((0.0, 0.0), (0.2, -0.1)):
            mdp = corridor_mdp(5, slip, 0.9, step_reward=-0.05, right_reward=right_reward)
            self.assertEqual((mdp.n_states, mdp.n_actions), (5, 2))
            np.testing.assert_allclose(mdp.transitions.sum(axis=2), 1.0)
            self.assertEqual(mdp.transitions[1, 4, 4], 1.0)  # goal is absorbing
            self.assertEqual(mdp.costs[4, 1], 0.0)
            # expected one-step cost from state 3 moving right: -(step + right) - P(reach goal) * 1
            self.assertAlmostEqual(mdp.costs[3, 1], 0.05 - right_reward - (1 - slip) * 1.0)
            self.assertAlmostEqual(mdp.costs[3, 0], 0.05 - slip * 1.0)
            # Monte Carlo from the environment agrees with exact policy evaluation of "always right"
            env = CorridorEnv(length=5, slip=slip, max_steps=200, step_reward=-0.05, right_reward=right_reward)
            env.reset(seed=0)
            returns = []
            for _ in range(2000):
                env.reset()
                total, discount, done = 0.0, 1.0, False
                while not done:
                    _, r, term, trunc, _ = env.step(1)
                    total += discount * r
                    discount *= 0.9
                    done = term or trunc
                returns.append(total)
            exact = -solvers.evaluate_policy(mdp, np.ones(5, dtype=int))[0]
            self.assertAlmostEqual(float(np.mean(returns)), exact, delta=4 * np.std(returns) / np.sqrt(2000) + 1e-9)

    def test_optimal_policy_depends_on_the_discount(self):
        """The discount experiment in tests/test_pg_part1.py relies on this flip."""
        myopic = solvers.value_iteration(corridor_mdp(5, 0.0, 0.0, right_reward=-0.1), tol=1e-12)
        patient = solvers.value_iteration(corridor_mdp(5, 0.0, 0.9, right_reward=-0.1), tol=1e-12)
        self.assertEqual(myopic.policy[0], 0)
        self.assertTrue(np.all(patient.policy[:-1] == 1))
        self.assertAlmostEqual(-patient.values[0], -0.1 * (1 + 0.9 + 0.81 + 0.729) + 0.9 ** 3, places=10)


if __name__ == "__main__":
    unittest.main()
