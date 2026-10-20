"""Basic policy gradient: training and evaluation loops."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rl.outputs import ProgressLogger


@dataclass
class TrainConfig:
    env: str  # "cartpole" or "pong"
    part: int  # 1 (no baseline) or 2 (with baseline)
    gamma: float
    lr: float = 0.01  # Adam learning rate from the handout; note any change in the report
    episodes: int = 1000
    baseline: str = "none"
    seed: int = 0
    device: str = "cpu"
    eval_episodes: int = 500
    extra: dict[str, Any] = field(default_factory=dict)  # anything passed with --set key=value


def train(cfg: TrainConfig, env: Any, policy: Any, baseline: Any, logger: ProgressLogger) -> list[float]:
    """Train `policy` and return the reward of every training episode, in order.

    Call logger.log(episode, reward, ...) once per episode so progress shows up in the job
    log and in the dashboard's live training curve.

    TODO(Part 1 / Part 2).
    """
    raise NotImplementedError("train (rl/pg/reinforce.py)")


def evaluate(cfg: TrainConfig, env: Any, policy: Any, episodes: int) -> list[float]:
    """Roll out `episodes` episodes with the trained policy and return their rewards.

    TODO(Part 1).
    """
    raise NotImplementedError("evaluate (rl/pg/reinforce.py)")
