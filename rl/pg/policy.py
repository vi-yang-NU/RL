"""Policy networks (PyTorch) for CartPole-v1 and Pong-v5."""
from __future__ import annotations

from typing import Any

import torch.nn as nn


def build_policy(cfg: Any, env: Any) -> nn.Module:
    """Return the neural network that models the policy for `cfg.env`.

    TODO(Part 1): a simple network that takes the state (CartPole) or game frames (Pong)
    and outputs a distribution over the allowed actions.
    """
    raise NotImplementedError(f"build_policy for {cfg.env} (rl/pg/policy.py)")
