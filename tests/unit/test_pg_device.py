"""Device placement: everything the policy interacts with must live on the configured device.

The checks run on a GPU when one is available (CUDA on deepdish, MPS on an Apple laptop) and are skipped
otherwise; a module or tensor left on the CPU would make the forward pass raise a device-mismatch error, so
a training run that completes on the GPU is the evidence that the code moves everything there.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from rl.outputs import ProgressLogger
from rl.pg.baselines import make_baseline
from rl.pg.envs import make_adapter
from rl.pg.options import options_for
from rl.pg.policy import build_policy
from rl.pg.reinforce import TrainConfig, check_device, evaluate, policy_gradient_loss, run_episode, train
from rl.pg.toy import BanditEnv, CorridorEnv


def gpu_device() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


class DevicePlacementTests(unittest.TestCase):
    def setUp(self):
        self.device = gpu_device()
        if self.device is None:
            self.skipTest("no CUDA or MPS device on this machine")

    def test_training_and_evaluation_run_on_the_gpu(self):
        env = CorridorEnv(length=4, max_steps=10)
        cfg = TrainConfig(env="corridor", part=2, gamma=0.9, lr=0.05, episodes=4, baseline="value", device=self.device)
        policy = build_policy(cfg, env).to(torch.device(self.device))
        baseline = make_baseline("value", cfg, env)
        with tempfile.TemporaryDirectory() as tmp:
            rewards = train(cfg, env, policy, baseline, ProgressLogger(Path(tmp), print_every=10 ** 9))
        self.assertEqual(len(rewards), 4)
        self.assertTrue(all(p.device.type == self.device for p in policy.parameters()))
        self.assertTrue(all(p.device.type == self.device for p in baseline.net.parameters()))
        self.assertEqual(len(evaluate(cfg, env, policy, 3)), 3)
        device = torch.device(self.device)
        episode = run_episode(env, make_adapter(cfg, env), policy, device)
        loss, _, (obs, returns, t) = policy_gradient_loss(policy, [episode], 0.9, baseline, options_for(cfg), device)
        self.assertEqual(loss.device.type, self.device)
        self.assertEqual({obs.device.type, returns.device.type, t.device.type}, {self.device})
        self.assertEqual(baseline.predict(obs, t).device.type, self.device)

    def test_a_policy_left_on_the_cpu_is_reported_and_train_moves_it(self):
        env = BanditEnv((0.0, 1.0))
        cfg = TrainConfig(env="bandit", part=1, gamma=0.9, device=self.device)
        policy = build_policy(cfg, env)  # not moved to the device
        with self.assertRaises(RuntimeError):
            check_device(policy, None, torch.device(self.device))
        with tempfile.TemporaryDirectory() as tmp:
            train(cfg, env, policy, None, ProgressLogger(Path(tmp), print_every=10 ** 9))  # train moves it first
        self.assertTrue(all(p.device.type == self.device for p in policy.parameters()))


class CpuPlacementTests(unittest.TestCase):
    def test_check_device_accepts_matching_modules(self):
        env = BanditEnv((0.0, 1.0))
        cfg = TrainConfig(env="bandit", part=2, gamma=0.9, device="cpu")
        check_device(build_policy(cfg, env), make_baseline("value", cfg, env), torch.device("cpu"))
        check_device(build_policy(cfg, env), make_baseline("mean", cfg, env), torch.device("cpu"))

    def test_episode_length_cap_defaults(self):
        self.assertEqual(options_for(TrainConfig(env="cartpole", part=1, gamma=0.95)).max_steps, 1000)
        self.assertEqual(options_for(TrainConfig(env="pong", part=1, gamma=0.99)).max_steps, 10_000)
        self.assertEqual(options_for(TrainConfig(env="pong", part=1, gamma=0.99, extra={"max_steps": 0})).max_steps, 0)


if __name__ == "__main__":
    unittest.main()
