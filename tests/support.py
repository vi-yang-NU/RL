"""Helpers shared by the validation experiments in tests/test_pg_part1.py and tests/test_pg_part2.py.

(Not a test module: the harness only imports tests/test_*.py.)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl.outputs import ProgressLogger
from rl.pg.baselines import make_baseline
from rl.pg.envs import make_adapter
from rl.pg.options import options_for
from rl.pg.policy import action_probabilities, build_policy
from rl.pg.reinforce import TrainConfig, policy_gradient_loss, run_episode, seed_everything, train

CPU = torch.device("cpu")
torch.set_num_threads(2)  # tiny networks: more intra-op threads only add synchronisation overhead (see Options.cpu_threads)


def pg_config(env: str, *, part: int = 1, gamma: float = 0.95, lr: float = 0.01, episodes: int = 200, seed: int = 0,
              baseline: str = "none", **extra: Any) -> TrainConfig:
    return TrainConfig(env=env, part=part, gamma=gamma, lr=lr, episodes=episodes, baseline=baseline, seed=seed,
                       device="cpu", extra=dict(extra))


def quick_train(cfg: TrainConfig, env: Any, policy: Any = None, baseline: Any = None) -> dict[str, Any]:
    """Train on `env` with the progress log in a temporary folder. Returns policy, rewards, log records, baseline."""
    policy = policy if policy is not None else build_policy(cfg, env)
    if baseline is None:
        baseline = make_baseline(cfg.baseline, cfg, env)
    with tempfile.TemporaryDirectory() as tmp:
        logger = ProgressLogger(Path(tmp), print_every=10 ** 9)
        rewards = train(cfg, env, policy, baseline, logger)
        records = [json.loads(line) for line in logger.path.read_text().splitlines()]
    return {"policy": policy, "rewards": rewards, "records": records, "baseline": baseline}


def flat_grad(policy: torch.nn.Module) -> np.ndarray:
    return torch.cat([p.grad.detach().flatten() for p in policy.parameters()]).numpy().copy()


def gradient_samples(policy: torch.nn.Module, env: Any, cfg: TrainConfig, n: int, baseline: Any = None,
                     options: Any = None, seed: int = 0) -> np.ndarray:
    """`n` single-episode policy-gradient estimates (flattened over all parameters) for a fixed policy and a
    fixed baseline (the baseline is not updated), to compare their means and variances."""
    opts = options if options is not None else options_for(cfg)
    adapter = make_adapter(cfg, env, opts)
    seed_everything(seed)
    samples = []
    for i in range(n):
        episode = run_episode(env, adapter, policy, CPU, max_steps=opts.max_steps, seed=seed if i == 0 else None)
        policy.zero_grad()
        loss, _, _ = policy_gradient_loss(policy, [episode], cfg.gamma, baseline, opts, CPU)
        loss.backward()
        samples.append(flat_grad(policy))
    return np.array(samples)


def probabilities(policy: torch.nn.Module, observations: np.ndarray) -> np.ndarray:
    """pi(. | s) for a batch of observations as a numpy array."""
    return action_probabilities(policy, torch.as_tensor(np.asarray(observations, dtype=np.float32))).numpy()


def mean_and_stderr(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
