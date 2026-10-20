"""Shuttle dispatching problem (Section 2): the given parameters and the MDP built from them."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ShuttleParams:
    """Problem constants from the handout.

    Validation experiments can override fields, e.g. ShuttleParams(capacity=5).
    """

    capacity: int = 15  # K: customers a dispatched shuttle can carry
    dispatch_cost: float = 100.0  # c_f: cost of dispatching a shuttle
    holding_cost: float = 2.0  # c_h: cost per customer left waiting per period (after dispatch)
    arrivals: tuple[int, ...] = (1, 2, 3, 4, 5)  # A_t ~ Unif{1, 2, 3, 4, 5}
    max_customers: int = 200  # the station never holds more than this many people
    gamma: float = 0.95  # discount rate
    horizon: int = 500  # T for enumeration, part (a)

    def describe(self) -> str:
        """One-line summary used as a plot subtitle."""
        lo, hi = min(self.arrivals), max(self.arrivals)
        return (
            f"K = {self.capacity}, c_f = {self.dispatch_cost:g}, c_h = {self.holding_cost:g}, "
            f"A_t ~ Unif{{{lo},…,{hi}}}, max {self.max_customers} waiting, γ = {self.gamma}"
        )


def build_mdp(params: ShuttleParams) -> Any:
    """Model the shuttle dispatching problem as a Markov Decision Process.

    TODO(Section 2): define the states, actions, transition probabilities and one-step
    costs/rewards, and return them in whatever structure your solvers take.
    """
    raise NotImplementedError("build_mdp: model the shuttle problem as an MDP (rl/dp/shuttle.py)")
