"""Baselines for variance reduction (Part 2).

A baseline b is subtracted from the return G_t that weights grad log pi(a_t|s_t). As long as b does not depend
on the action a_t, the expected gradient is unchanged (lean/RLVerify/Baseline.lean, `baseline_unbiased`), and
a baseline close to E[G_t | s_t] shrinks the variance of the estimate.

Every baseline has `predict(observations, timesteps) -> baseline per step` and
`update(observations, returns, timesteps) -> statistics`, called by rl/pg/reinforce.py after each policy update.

    none    no baseline (Part 1)
    mean    constant baseline: exponential moving average of the returns G_t seen so far
    time    b(t): running mean of G_t at each time step t (non-constant in time, constant across states)
    value   b(s) = V(s): a state-value network fitted to the returns by least squares (the default)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rl.pg.options import options_for

# Baseline that `python run.py pg --part 2` uses when --baseline is not given.
DEFAULT_BASELINE: str | None = "value"


class MeanBaseline:
    """Constant baseline: an exponential moving average of all observed returns G_t."""

    name = "mean"

    def __init__(self, decay: float = 0.9):
        if not 0.0 <= decay < 1.0:
            raise ValueError("baseline_decay must be in [0, 1)")
        self.decay = float(decay)
        self.value: float | None = None

    def predict(self, observations: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.full((len(timesteps),), self.value or 0.0, dtype=torch.float32, device=observations.device)

    def update(self, observations: torch.Tensor, returns: torch.Tensor, timesteps: torch.Tensor) -> dict[str, float]:
        batch_mean = float(returns.mean().item())
        self.value = batch_mean if self.value is None else self.decay * self.value + (1.0 - self.decay) * batch_mean
        return {"baseline": self.value}


class TimeBaseline:
    """b(t): the running mean of the returns observed at time step t."""

    name = "time"

    def __init__(self) -> None:
        self.sums = np.zeros(0)
        self.counts = np.zeros(0)

    def _grow(self, n: int) -> None:
        if n > len(self.sums):
            self.sums = np.concatenate([self.sums, np.zeros(n - len(self.sums))])
            self.counts = np.concatenate([self.counts, np.zeros(n - len(self.counts))])

    def predict(self, observations: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t = timesteps.detach().cpu().numpy().astype(int)
        self._grow(int(t.max()) + 1 if len(t) else 0)
        means = np.divide(self.sums, self.counts, out=np.zeros_like(self.sums), where=self.counts > 0)
        return torch.as_tensor(means[t], dtype=torch.float32, device=observations.device)

    def update(self, observations: torch.Tensor, returns: torch.Tensor, timesteps: torch.Tensor) -> dict[str, float]:
        t = timesteps.detach().cpu().numpy().astype(int)
        self._grow(int(t.max()) + 1 if len(t) else 0)
        np.add.at(self.sums, t, returns.detach().cpu().numpy().astype(float))
        np.add.at(self.counts, t, 1.0)
        return {"baseline_t0": float(self.sums[0] / self.counts[0]) if self.counts[0] else 0.0}


class ValueBaseline:
    """b(s) = V(s), a state-value network trained to predict the return G_t from the observation s_t."""

    name = "value"

    def __init__(self, cfg: Any, env: Any):
        from rl.pg.policy import build_value_net

        opts = options_for(cfg)
        self.epochs = max(1, int(opts.baseline_epochs))
        self.net = build_value_net(cfg, env, opts.baseline_hidden or None).to(torch.device(cfg.device))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(opts.baseline_lr))

    def predict(self, observations: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.net(observations).squeeze(-1)

    def update(self, observations: torch.Tensor, returns: torch.Tensor, timesteps: torch.Tensor) -> dict[str, float]:
        loss = torch.zeros(())
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(self.net(observations).squeeze(-1), returns.to(torch.float32))
            loss.backward()
            self.optimizer.step()
        return {"value_loss": float(loss.item())}

    def state_dict(self) -> dict[str, Any]:
        return self.net.state_dict()


BASELINES = {"mean": MeanBaseline, "time": TimeBaseline, "value": ValueBaseline}


def make_baseline(name: str, cfg: Any, env: Any) -> Any:
    """Return the baseline object the training loop uses, or None for "none" (Part 1)."""
    if name in (None, "none", ""):
        return None
    if name == "mean":
        return MeanBaseline(options_for(cfg).baseline_decay)
    if name == "time":
        return TimeBaseline()
    if name == "value":
        return ValueBaseline(cfg, env)
    raise ValueError(f"unknown baseline {name!r}; choose from {['none', *BASELINES]}")
