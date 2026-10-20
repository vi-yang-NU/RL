"""Solvers for the shuttle MDP, Section 2 (a)-(c).

Each solver returns a SolverResult. rl/dp/run.py saves it to results/dp/<method>.json and
draws the plot the handout asks for from `states` and `values` (or `policy`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SolverResult:
    states: np.ndarray  # x-axis of the plots: number of customers waiting
    values: np.ndarray | None = None  # optimal value per state (enumeration: at time 0)
    policy: np.ndarray | None = None  # optimal action per state
    iterations: int | None = None
    history: dict[str, list[float]] = field(default_factory=dict)  # optional per-iteration diagnostics, plotted if set
    info: dict[str, Any] = field(default_factory=dict)  # anything else worth saving

    def to_json(self) -> dict[str, Any]:
        return {
            "states": self.states,
            "values": self.values,
            "policy": self.policy,
            "iterations": self.iterations,
            "history": self.history,
            "info": self.info,
        }


def enumeration(mdp: Any, horizon: int) -> SolverResult:
    """(a) Solve by enumeration over a finite horizon of `horizon` periods.

    TODO(Section 2a): return a SolverResult whose `values` is the optimal value function at time 0.
    """
    raise NotImplementedError("enumeration (rl/dp/solvers.py)")


def value_iteration(mdp: Any, tol: float = 1e-6, max_iter: int = 100_000) -> SolverResult:
    """(b) Solve the infinite-horizon problem (T = ∞) by value iteration.

    TODO(Section 2b): return a SolverResult whose `values` is the optimal value function.
    """
    raise NotImplementedError("value_iteration (rl/dp/solvers.py)")


def policy_iteration(mdp: Any, max_iter: int = 1_000) -> SolverResult:
    """(c) Solve by policy iteration.

    TODO(Section 2c): return a SolverResult whose `policy` is the optimal policy.
    """
    raise NotImplementedError("policy_iteration (rl/dp/solvers.py)")
