"""Unit tests for the discounted-return arithmetic in rl/pg/reinforce.py.

`discounted_returns` is the backward recursion G_t = r_t + gamma G_{t+1}; lean/RLVerify/Returns.lean proves
it equals the definition G_t = sum_k gamma^k r_{t+k}. These tests check the floating-point implementation.
"""
from __future__ import annotations

import unittest

import numpy as np

from rl.pg.options import Options
from rl.pg.reinforce import Episode, discounted_returns, step_weights


class DiscountedReturnTests(unittest.TestCase):
    def test_matches_the_definition(self):
        rng = np.random.default_rng(0)
        for gamma in (0.0, 0.5, 0.95, 0.99, 1.0):
            r = rng.normal(size=37)
            expected = [sum(gamma ** k * r[t + k] for k in range(len(r) - t)) for t in range(len(r))]
            np.testing.assert_allclose(discounted_returns(r, gamma), expected, rtol=1e-12, atol=1e-12, err_msg=f"gamma {gamma}")

    def test_special_cases(self):
        self.assertEqual(discounted_returns([], 0.9).shape, (0,))
        np.testing.assert_allclose(discounted_returns([2.5], 0.9), [2.5])
        r = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(discounted_returns(r, 0.0), r)  # no discounting of the future at all
        np.testing.assert_allclose(discounted_returns(r, 1.0), [10.0, 9.0, 7.0, 4.0])  # reverse cumulative sum
        np.testing.assert_allclose(discounted_returns(np.zeros(5), 0.9), np.zeros(5))

    def test_constant_reward_closed_form(self):
        """CartPole gives reward 1 per step: G_t = (1 - gamma^(T-t)) / (1 - gamma)."""
        T, gamma = 500, 0.95
        G = discounted_returns(np.ones(T), gamma)
        steps_left = T - np.arange(T)
        np.testing.assert_allclose(G, (1 - gamma ** steps_left) / (1 - gamma), rtol=1e-12)
        self.assertLess(G.max(), 1 / (1 - gamma))

    def test_bound(self):
        rng = np.random.default_rng(1)
        r = rng.uniform(-3, 3, size=200)
        gamma = 0.99
        self.assertLessEqual(np.abs(discounted_returns(r, gamma)).max(), np.abs(r).max() / (1 - gamma) + 1e-9)


class StepWeightTests(unittest.TestCase):
    def setUp(self):
        self.episode = Episode(np.zeros((4, 1), np.float32), np.zeros(4, np.int64), np.array([1.0, 0.0, 2.0, 1.0], np.float32))
        self.gamma = 0.5
        self.G = discounted_returns(self.episode.rewards, self.gamma)

    def test_reward_to_go_is_the_default(self):
        np.testing.assert_allclose(step_weights(self.episode, self.gamma, Options()), self.G)

    def test_whole_episode_return(self):
        w = step_weights(self.episode, self.gamma, Options(reward_to_go=False))
        np.testing.assert_allclose(w, np.full(4, self.G[0]))

    def test_discount_grad_adds_gamma_to_the_t(self):
        w = step_weights(self.episode, self.gamma, Options(discount_grad=True))
        np.testing.assert_allclose(w, self.G * self.gamma ** np.arange(4))
        both = step_weights(self.episode, self.gamma, Options(discount_grad=True, reward_to_go=False))
        np.testing.assert_allclose(both, self.G[0] * self.gamma ** np.arange(4))


if __name__ == "__main__":
    unittest.main()
