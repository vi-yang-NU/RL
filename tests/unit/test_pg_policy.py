"""Unit tests for the policy networks, options and environment adapters (rl/pg/policy.py, options.py, envs.py)."""
from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import torch

from rl.pg.envs import PONG_ACTIONS, Adapter, PongAdapter, make_adapter, make_env, preprocess
from rl.pg.options import Options, options_for
from rl.pg.policy import CNN, MLP, action_probabilities, build_policy, build_value_net
from rl.pg.reinforce import TrainConfig
from rl.pg.toy import BanditEnv, CorridorEnv


class FakePong:
    """Produces Pong-shaped frames (210x160x3 uint8) with the two background colours and a moving ball."""

    def __init__(self, steps: int = 3):
        from gymnasium import spaces

        self.observation_space = spaces.Box(0, 255, shape=(210, 160, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(6)
        self.steps, self.t, self.actions = steps, 0, []

    def frame(self) -> np.ndarray:
        img = np.full((210, 160, 3), 144, dtype=np.uint8)
        img[100:] = 109
        img[51 + 2 * self.t, 40 + 2 * self.t] = 236  # the ball (on a row and column the downsampling keeps)
        img[60:80, 16] = 213  # a paddle
        return img

    def reset(self, seed: Any = None, options: Any = None):
        self.t = 0
        return self.frame(), {}

    def step(self, action: int):
        self.actions.append(action)
        self.t += 1
        return self.frame(), 0.0, self.t >= self.steps, False, {}


def cfg(env: str = "cartpole", **extra: Any) -> TrainConfig:
    return TrainConfig(env=env, part=1, gamma=0.95, extra=extra)


class OptionsTests(unittest.TestCase):
    def test_defaults_and_environment_defaults(self):
        self.assertEqual(options_for(cfg("cartpole")).hidden, (64,))
        self.assertEqual(options_for(cfg("pong")).hidden, (200,))
        self.assertEqual(options_for(cfg("bandit")).batch_episodes, 1)
        self.assertFalse(options_for(cfg("cartpole")).normalize)

    def test_overrides_and_coercion(self):
        o = options_for(cfg("pong", hidden=[128, 64], normalize="true", batch_episodes=4.0, arch="cnn"))
        self.assertEqual(o.hidden, (128, 64))
        self.assertTrue(o.normalize)
        self.assertEqual(o.batch_episodes, 4)
        self.assertIsInstance(o.batch_episodes, int)
        self.assertEqual(o.arch, "cnn")

    def test_unknown_option_is_rejected(self):
        with self.assertRaises(ValueError):
            options_for(cfg("cartpole", hiden=[8]))

    def test_describe_lists_every_field(self):
        self.assertIn("reward_to_go", Options().describe())


class PolicyNetworkTests(unittest.TestCase):
    def test_cartpole_policy_shapes_and_probabilities(self):
        env = make_env("cartpole")
        try:
            policy = build_policy(cfg("cartpole"), env)
            self.assertIsInstance(policy, MLP)
            self.assertEqual((policy.in_dim, policy.out_dim), (4, 2))
            x = torch.randn(5, 4)
            self.assertEqual(policy(x).shape, (5, 2))
            probs = action_probabilities(policy, x)
            np.testing.assert_allclose(probs.sum(dim=1).numpy(), 1.0, rtol=1e-6)
            self.assertTrue(bool((probs > 0).all()))
            value = build_value_net(cfg("cartpole"), env)
            self.assertEqual(value(x).shape, (5, 1))
        finally:
            env.close()

    def test_hidden_layers_change_the_parameter_count(self):
        env = BanditEnv((0.0, 1.0))
        small = build_policy(cfg("bandit", hidden=[]), env)
        self.assertEqual(sum(p.numel() for p in small.parameters()), 1 * 2 + 2)
        deep = build_policy(cfg("bandit", hidden=[32, 32]), env)
        self.assertEqual(sum(p.numel() for p in deep.parameters()), (1 * 32 + 32) + (32 * 32 + 32) + (32 * 2 + 2))
        corridor = build_policy(cfg("corridor"), CorridorEnv(length=7))
        self.assertEqual((corridor.in_dim, corridor.out_dim), (7, 2))

    def test_pong_networks(self):
        fake = FakePong()
        mlp = build_policy(cfg("pong"), fake)
        self.assertEqual((mlp.in_dim, mlp.out_dim), (6400, 2))
        self.assertEqual(mlp(torch.zeros(3, 6400)).shape, (3, 2))
        cnn = build_policy(cfg("pong", arch="cnn"), fake)
        self.assertIsInstance(cnn, CNN)
        self.assertEqual(cnn(torch.zeros(3, 1, 80, 80)).shape, (3, 2))
        self.assertEqual(cnn(torch.zeros(3, 6400)).shape, (3, 2))  # flattened input is reshaped
        self.assertEqual(build_value_net(cfg("pong", arch="cnn"), fake)(torch.zeros(2, 1, 80, 80)).shape, (2, 1))

    def test_invalid_architectures(self):
        with self.assertRaises(ValueError):
            build_policy(cfg("cartpole", arch="cnn"), make_env("cartpole"))
        with self.assertRaises(ValueError):
            build_policy(cfg("pong", arch="transformer"), FakePong())


class AdapterTests(unittest.TestCase):
    def test_vector_adapter(self):
        env = CorridorEnv(length=4)
        adapter = make_adapter(cfg("corridor"), env)
        self.assertIsInstance(adapter, Adapter)
        self.assertEqual((adapter.obs_shape, adapter.n_actions), ((4,), 2))
        obs = adapter.reset(env, seed=0)
        np.testing.assert_array_equal(obs, [1, 0, 0, 0])
        self.assertEqual(obs.dtype, np.float32)
        obs, reward, done, info = adapter.step(env, 1)
        np.testing.assert_array_equal(obs, [0, 1, 0, 0])
        self.assertEqual((reward, done), (0.0, False))
        self.assertFalse(info["terminated"])
        with self.assertRaises(ValueError):
            adapter.action(2)
        with self.assertRaises(ValueError):
            Adapter(FakePong())  # image observations need the Pong adapter

    def test_preprocess_gives_a_binary_80x80_image(self):
        fake = FakePong()
        frame = fake.frame()
        original = frame.copy()
        out = preprocess(frame.copy())
        self.assertEqual(out.shape, (80, 80))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(set(np.unique(out)) <= {0.0, 1.0})
        self.assertGreater(out.sum(), 0)  # the ball and paddle survive
        np.testing.assert_array_equal(frame, original)  # preprocess received a copy

    def test_pong_adapter_frame_difference_and_action_mapping(self):
        fake = FakePong(steps=3)
        adapter = make_adapter(cfg("pong"), fake)
        self.assertIsInstance(adapter, PongAdapter)
        self.assertEqual((adapter.obs_shape, adapter.n_actions), ((6400,), 2))
        first = adapter.reset(fake)
        np.testing.assert_array_equal(first, 0.0)  # nothing to difference against yet
        obs, _, done, _ = adapter.step(fake, 0)
        self.assertEqual(obs.shape, (6400,))
        self.assertTrue(set(np.unique(obs)) <= {-1.0, 0.0, 1.0})
        self.assertGreater(np.abs(obs).sum(), 0)  # the ball moved
        self.assertEqual(fake.actions, [PONG_ACTIONS[0]])
        adapter.step(fake, 1)
        self.assertEqual(fake.actions, [2, 3])
        with self.assertRaises(ValueError):
            adapter.action(2)
        np.testing.assert_array_equal(adapter.reset(fake), 0.0)  # a reset forgets the previous frame
        self.assertIsNotNone(adapter.previous)  # ...and remembers the new first frame for the next difference

    def test_pong_adapter_without_frame_difference_and_for_the_cnn(self):
        fake = FakePong()
        raw = make_adapter(cfg("pong", frame_diff=False), fake)
        obs = raw.reset(fake)
        self.assertTrue(set(np.unique(obs)) <= {0.0, 1.0})
        self.assertGreater(obs.sum(), 0)
        cnn = make_adapter(cfg("pong", arch="cnn"), fake)
        self.assertEqual(cnn.reset(fake).shape, (1, 80, 80))

    def test_real_pong_uses_only_right_and_left(self):
        env = make_env("pong")
        try:
            self.assertEqual(env.action_space.n, 6)
            adapter = make_adapter(cfg("pong"), env)
            adapter.reset(env, seed=0)
            for a in (0, 1, 0):
                obs, reward, done, _ = adapter.step(env, a)
                self.assertEqual(obs.shape, (6400,))
            self.assertEqual([adapter.action(a) for a in (0, 1)], [2, 3])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
