"""Policy networks (PyTorch) for CartPole-v1 and Pong-v5.

The network maps an observation to one logit per allowed action and ends with a softmax layer: `forward`
returns the logits (what `torch.distributions.Categorical(logits=...)` and `log_softmax` consume, which is the
numerically stable way to apply the softmax) and `probabilities` returns the softmax itself.
`build_policy(cfg, env)` chooses the architecture from the environment and the options in cfg.extra:

* CartPole-v1 (and any vector observation): MLP, obs_dim -> hidden -> ... -> n_actions (default hidden (64,)).
* Pong-v5: MLP on the flattened 80x80 frame difference, 6400 -> 200 -> 2 (`--set arch=cnn` for a small
  convolutional network instead; `--set hidden=[...]` changes the layer sizes).

`build_value_net` builds the same architecture with a single output for the value baseline (Part 2).
"""
from __future__ import annotations

from math import prod
from typing import Any, Sequence

import torch
import torch.nn as nn

from rl.pg.envs import PONG_FRAME, make_adapter
from rl.pg.options import options_for


class PolicyNet(nn.Module):
    """Common interface: `forward` gives logits, `probabilities` the softmax over the allowed actions."""

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)

    def log_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.forward(x), dim=-1)


class MLP(PolicyNet):
    """Fully connected network with ReLU hidden layers; the input is flattened."""

    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int] = (64,)):
        super().__init__()
        layers: list[nn.Module] = []
        last = int(in_dim)
        for size in hidden:
            layers += [nn.Linear(last, int(size)), nn.ReLU()]
            last = int(size)
        layers.append(nn.Linear(last, int(out_dim)))
        self.net = nn.Sequential(*layers)
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


class CNN(PolicyNet):
    """Two convolutions on a 1x80x80 frame, then a fully connected layer (hidden[0] units) and the output."""

    def __init__(self, out_dim: int, hidden: Sequence[int] = (256,), frame: tuple = PONG_FRAME):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_features = self.features(torch.zeros(1, 1, *frame)).shape[1]
        width = int(hidden[0]) if hidden else 256
        self.head = nn.Sequential(nn.Linear(n_features, width), nn.ReLU(), nn.Linear(width, int(out_dim)))
        self.frame = tuple(frame)
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.reshape(x.shape[0], 1, *self.frame)))


def _build(cfg: Any, env: Any, out_dim: int, hidden: Sequence[int] | None = None) -> nn.Module:
    opts = options_for(cfg)
    adapter = make_adapter(cfg, env, opts)
    hidden = tuple(opts.hidden if hidden is None else hidden)
    if opts.arch == "cnn":
        if getattr(cfg, "env", "") != "pong":
            raise ValueError("arch=cnn is only available for Pong (image observations)")
        return CNN(out_dim, hidden)
    if opts.arch != "mlp":
        raise ValueError(f"arch must be 'mlp' or 'cnn', got {opts.arch!r}")
    return MLP(prod(adapter.obs_shape), out_dim, hidden)


def build_policy(cfg: Any, env: Any) -> nn.Module:
    """The network that models the policy for cfg.env: observation -> logits over the allowed actions."""
    opts = options_for(cfg)
    return _build(cfg, env, make_adapter(cfg, env, opts).n_actions)


def build_value_net(cfg: Any, env: Any, hidden: Sequence[int] | None = None) -> nn.Module:
    """A state-value network with the policy's architecture and one output (the value baseline)."""
    return _build(cfg, env, 1, hidden)


def action_probabilities(policy: nn.Module, observations: torch.Tensor) -> torch.Tensor:
    """pi(. | s) for a batch of observations."""
    with torch.no_grad():
        return torch.softmax(policy(observations), dim=-1)
