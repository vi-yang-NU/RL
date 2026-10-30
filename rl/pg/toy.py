"""Tiny environments with known optimal policies, used to validate the policy-gradient code.

Both follow the gymnasium API (`reset(seed=...)` -> (obs, info), `step(a)` -> (obs, reward, terminated,
truncated, info)), have a Box observation space and a Discrete action space, so `rl.pg.reinforce.train` runs on
them unchanged with cfg.env = "bandit" / "corridor" (anything but "pong" uses the vector adapter).

* `BanditEnv`: one decision, k arms with given mean rewards (deterministic, or Bernoulli with `bernoulli=True`).
  The optimal policy puts all probability on the best arm; the gradient has a closed form, so the test can
  compare the implementation's gradient with the analytic one.
* `CorridorEnv`: states 0..length-1 (one-hot observation), start at 0; action 1 moves right, 0 moves left (with
  probability `slip` the move is reversed). Reaching the last state gives reward 1 and ends the episode; every
  step gives `step_reward`, and choosing "right" additionally gives `right_reward` (a small negative value makes
  the myopic agent prefer to stay, which isolates the effect of the discount factor). `corridor_mdp` builds the
  same problem as a TabularMDP, so the DP solvers of Section 2 give the exact optimal values and policy that
  REINFORCE should approach.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from rl.dp.shuttle import TabularMDP


def _spaces():
    from gymnasium import spaces

    return spaces


class BanditEnv:
    """k-armed bandit as a one-step episode. Observation: the constant vector [1.0]."""

    metadata: dict[str, Any] = {}

    def __init__(self, means: tuple = (0.0, 1.0), bernoulli: bool = False, seed: int = 0):
        spaces = _spaces()
        self.means = np.asarray(means, dtype=float)
        if self.means.ndim != 1 or len(self.means) < 2:
            raise ValueError("means must list at least two arms")
        if bernoulli and (self.means.min() < 0 or self.means.max() > 1):
            raise ValueError("Bernoulli arms need means in [0, 1]")
        self.bernoulli = bernoulli
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.means))
        self.rng = np.random.default_rng(seed)
        self.done = True

    @property
    def best_arm(self) -> int:
        return int(self.means.argmax())

    def reset(self, seed: int | None = None, options: Any = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.done = False
        return np.ones(1, dtype=np.float32), {}

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() after the episode ended; call reset()")
        if not 0 <= int(action) < len(self.means):
            raise ValueError(f"invalid arm {action}")
        self.done = True
        mean = float(self.means[int(action)])
        reward = float(self.rng.random() < mean) if self.bernoulli else mean
        return np.ones(1, dtype=np.float32), reward, True, False, {}

    def close(self) -> None:
        pass


class CorridorEnv:
    """Walk right along a corridor of `length` cells to collect the reward at the end."""

    metadata: dict[str, Any] = {}

    def __init__(self, length: int = 5, slip: float = 0.0, max_steps: int = 50, step_reward: float = 0.0,
                 right_reward: float = 0.0, seed: int = 0):
        spaces = _spaces()
        if length < 2:
            raise ValueError("length must be at least 2")
        if not 0.0 <= slip <= 1.0:
            raise ValueError("slip must be a probability")
        self.length, self.slip, self.max_steps, self.step_reward = int(length), float(slip), int(max_steps), float(step_reward)
        self.right_reward = float(right_reward)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.length,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.rng = np.random.default_rng(seed)
        self.state = 0
        self.steps = 0
        self.done = True

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.length, dtype=np.float32)
        obs[self.state] = 1.0
        return obs

    def reset(self, seed: int | None = None, options: Any = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state, self.steps, self.done = 0, 0, False
        return self._obs(), {}

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() after the episode ended; call reset()")
        if int(action) not in (0, 1):
            raise ValueError(f"invalid action {action}")
        move = 1 if int(action) == 1 else -1
        if self.slip and self.rng.random() < self.slip:
            move = -move
        self.state = int(np.clip(self.state + move, 0, self.length - 1))
        self.steps += 1
        terminated = self.state == self.length - 1
        truncated = not terminated and self.steps >= self.max_steps
        reward = (1.0 if terminated else 0.0) + self.step_reward + (self.right_reward if int(action) == 1 else 0.0)
        self.done = terminated or truncated
        return self._obs(), reward, terminated, truncated, {}

    def close(self) -> None:
        pass


def corridor_mdp(length: int = 5, slip: float = 0.0, gamma: float = 0.9, step_reward: float = 0.0, right_reward: float = 0.0) -> TabularMDP:
    """The corridor as a TabularMDP with costs = -rewards (the goal state is absorbing with zero cost)."""
    S, goal = int(length), int(length) - 1
    P = np.zeros((2, S, S))
    C = np.zeros((S, 2))
    for s in range(S):
        if s == goal:
            P[:, s, s] = 1.0
            continue
        for a, move in ((0, -1), (1, 1)):
            C[s, a] -= step_reward + (right_reward if a == 1 else 0.0)
            for direction, p in ((move, 1.0 - slip), (-move, slip)):
                if p == 0.0:
                    continue
                nxt = int(np.clip(s + direction, 0, S - 1))
                P[a, s, nxt] += p
                C[s, a] -= p * (1.0 if nxt == goal else 0.0)
    return TabularMDP(P, C, gamma, states=np.arange(S), action_labels=("Left", "Right"), name="corridor")


def make_toy_env(name: str, **kwargs: Any) -> Any:
    if name == "bandit":
        return BanditEnv(**kwargs)
    if name == "corridor":
        return CorridorEnv(**kwargs)
    raise ValueError(f"unknown toy environment {name!r}")
