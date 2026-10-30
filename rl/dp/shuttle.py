"""Shuttle dispatching problem (Section 2): the parameters, and the finite MDP built from them.

Model (the assumptions are repeated in the report):

* **Decision epochs** t = 0, 1, 2, ...  **State** s = number of customers waiting when the agent decides,
  0 <= s <= max_customers (people arriving when the station is full are turned away, so the state is capped).
* **Actions** a in {0, ..., max_dispatch}: how many shuttles to dispatch (the handout's problem has
  max_dispatch = 1, i.e. "dispatch" or "do not dispatch"). A dispatched shuttle takes min(s, K) customers.
* **One-step cost** c(s, a) = a * c_f + c_h * (customers left waiting). By default those are the customers
  left after the dispatch (the handout: "incurred after dispatch"); `holding_charged` switches to charging
  everyone present before the dispatch, or everyone waiting at the end of the period (after the arrivals).
* **Transition** s' = min(s - served + A_t, max_customers), where A_t is drawn from `arrivals` with
  probabilities `arrival_probs` (uniform by default).
* **Objective** minimise the expected discounted cost, discount factor gamma.

`build_mdp` turns `ShuttleParams` into a `TabularMDP` (dense transition and cost arrays). The solvers in
rl/dp/solvers.py work on any `TabularMDP`, so the validation experiments can also solve other small problems
(for example the corridor environment used to cross-check the policy-gradient code).

Formal counterpart: lean/RLVerify/Shuttle.lean defines the same model over exact rationals and proves its
structural properties (state stays within bounds, costs are non-negative, the Bellman operator is monotone
and a gamma-contraction); lean/reference/shuttle_reference.json holds exact values the unit tests compare with.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

HOLDING_CHARGED = ("after_dispatch", "before_dispatch", "end_of_period")


@dataclass(frozen=True)
class ShuttleParams:
    """Problem constants from the handout. Any field can be changed, e.g. ShuttleParams(capacity=5) or
    `python run.py dp --set capacity=5 --set arrivals=[1,2,3]`."""

    capacity: int = 15  # K: customers one dispatched shuttle can carry
    dispatch_cost: float = 100.0  # c_f: cost of dispatching one shuttle
    holding_cost: float = 2.0  # c_h: cost per customer left waiting per period
    arrivals: tuple = (1, 2, 3, 4, 5)  # support of A_t
    arrival_probs: Any = None  # probabilities of `arrivals`; None = uniform (A_t ~ Unif{1, ..., 5})
    max_customers: int = 200  # the station never holds more than this many people
    gamma: float = 0.95  # discount rate
    horizon: int = 500  # T for enumeration, part (a)
    max_dispatch: int = 1  # shuttles that may be dispatched per period (actions 0..max_dispatch)
    holding_charged: str = "after_dispatch"  # which customers pay c_h: see the module docstring

    def __post_init__(self) -> None:
        object.__setattr__(self, "arrivals", tuple(int(a) for a in self.arrivals))
        if self.arrival_probs is not None:
            object.__setattr__(self, "arrival_probs", tuple(float(p) for p in self.arrival_probs))
        self.validate()

    def validate(self) -> None:
        def nonneg_int(name: str, value: Any) -> None:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")

        def nonneg_finite(name: str, value: Any) -> None:
            if not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite number, got {value!r}")

        nonneg_int("capacity", self.capacity)
        nonneg_finite("dispatch_cost", self.dispatch_cost)
        nonneg_finite("holding_cost", self.holding_cost)
        nonneg_int("max_customers", self.max_customers)
        nonneg_int("horizon", self.horizon)
        nonneg_int("max_dispatch", self.max_dispatch)
        if self.max_dispatch < 1:
            raise ValueError("max_dispatch must be at least 1 (otherwise there is no decision to make)")
        if not self.arrivals:
            raise ValueError("arrivals must list at least one possible number of arrivals")
        for a in self.arrivals:
            nonneg_int("each arrival count", a)
        if self.arrival_probs is not None:
            if len(self.arrival_probs) != len(self.arrivals):
                raise ValueError("arrival_probs must have one probability per entry of arrivals")
            if any(p < 0 for p in self.arrival_probs) or not math.isclose(sum(self.arrival_probs), 1.0, abs_tol=1e-9):
                raise ValueError("arrival_probs must be non-negative and sum to 1")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if self.holding_charged not in HOLDING_CHARGED:
            raise ValueError(f"holding_charged must be one of {HOLDING_CHARGED}, got {self.holding_charged!r}")

    def with_(self, **changes: Any) -> "ShuttleParams":
        """A copy with some fields changed: params.with_(capacity=5, gamma=0.9)."""
        return replace(self, **changes)

    def arrival_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        """(values, probabilities) of A_t; repeated values are merged."""
        values = np.asarray(self.arrivals, dtype=int)
        probs = np.full(len(values), 1.0 / len(values)) if self.arrival_probs is None else np.asarray(self.arrival_probs, float)
        merged: dict[int, float] = {}
        for v, p in zip(values, probs):
            merged[int(v)] = merged.get(int(v), 0.0) + float(p)
        values = np.array(sorted(merged), dtype=int)
        probs = np.array([merged[v] for v in values])
        return values, probs / probs.sum()

    @property
    def mean_arrivals(self) -> float:
        values, probs = self.arrival_distribution()
        return float(values @ probs)

    @property
    def is_uniform(self) -> bool:
        values, probs = self.arrival_distribution()
        return bool(np.allclose(probs, 1.0 / len(values)))

    @property
    def n_states(self) -> int:
        return self.max_customers + 1

    def describe(self) -> str:
        """One-line summary used as a plot subtitle."""
        values, _ = self.arrival_distribution()
        contiguous = list(values) == list(range(values[0], values[-1] + 1))
        if self.is_uniform and contiguous:
            arrivals = f"A_t ~ Unif{{{values[0]},…,{values[-1]}}}" if len(values) > 1 else f"A_t = {values[0]}"
        else:
            arrivals = f"A_t ∈ {{{', '.join(map(str, values))}}}" + ("" if self.is_uniform else " (non-uniform)")
        text = (
            f"K = {self.capacity}, c_f = {self.dispatch_cost:g}, c_h = {self.holding_cost:g}, {arrivals}, "
            f"max {self.max_customers} waiting, γ = {self.gamma:g}"
        )
        if self.max_dispatch != 1:
            text += f", up to {self.max_dispatch} shuttles per period"
        if self.holding_charged != "after_dispatch":
            text += f", holding charged {self.holding_charged.replace('_', ' ')}"
        return text


@dataclass(frozen=True)
class TabularMDP:
    """A finite MDP with costs to minimise.

    transitions[a, s, s'] = P(s' | s, a), costs[s, a] = expected one-step cost, gamma = discount factor.
    `states` labels the states on the plots' x-axis and `action_labels` the actions on the policy plot.
    """

    transitions: np.ndarray
    costs: np.ndarray
    gamma: float
    states: np.ndarray
    action_labels: tuple
    name: str = "mdp"

    def __post_init__(self) -> None:
        P = np.asarray(self.transitions, dtype=float)
        C = np.asarray(self.costs, dtype=float)
        object.__setattr__(self, "transitions", P)
        object.__setattr__(self, "costs", C)
        object.__setattr__(self, "states", np.asarray(self.states))
        object.__setattr__(self, "action_labels", tuple(self.action_labels))
        if P.ndim != 3 or P.shape[1] != P.shape[2]:
            raise ValueError(f"transitions must have shape (actions, states, states), got {P.shape}")
        A, S, _ = P.shape
        if C.shape != (S, A):
            raise ValueError(f"costs must have shape (states, actions) = {(S, A)}, got {C.shape}")
        if len(self.states) != S or len(self.action_labels) != A:
            raise ValueError("states and action_labels must match the transition array")
        if not np.all(np.isfinite(C)):
            raise ValueError("costs must be finite")
        if np.any(P < -1e-12):
            raise ValueError("transition probabilities must be non-negative")
        rows = P.sum(axis=2)
        if not np.allclose(rows, 1.0, atol=1e-9):
            bad = np.argwhere(~np.isclose(rows, 1.0, atol=1e-9))[0]
            raise ValueError(f"transition probabilities must sum to 1 in every row; row (a={bad[0]}, s={bad[1]}) sums to {rows[tuple(bad)]:.6g}")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")

    @property
    def n_states(self) -> int:
        return self.transitions.shape[1]

    @property
    def n_actions(self) -> int:
        return self.transitions.shape[0]

    def expected_next(self, values: np.ndarray) -> np.ndarray:
        """E[V(s') | s, a] for every (s, a): shape (states, actions)."""
        return np.einsum("asj,j->sa", self.transitions, np.asarray(values, dtype=float))

    def q_values(self, values: np.ndarray) -> np.ndarray:
        """Q(s, a) = c(s, a) + gamma * E[V(s') | s, a] (the Bellman operator before minimising over a)."""
        return self.costs + self.gamma * self.expected_next(values)

    def bellman(self, values: np.ndarray) -> np.ndarray:
        """(T V)(s) = min_a Q(s, a)."""
        return self.q_values(values).min(axis=1)

    def policy_transitions(self, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(P_pi, c_pi): the Markov chain and cost vector induced by a deterministic policy."""
        policy = np.asarray(policy, dtype=int)
        if policy.shape != (self.n_states,) or policy.min() < 0 or policy.max() >= self.n_actions:
            raise ValueError("policy must give one valid action index per state")
        idx = np.arange(self.n_states)
        return self.transitions[policy, idx, :], self.costs[idx, policy]


def build_mdp(params: ShuttleParams) -> TabularMDP:
    """The shuttle dispatching problem as a TabularMDP (see the module docstring for the model)."""
    params.validate()
    M, K = params.max_customers, params.capacity
    arrivals, probs = params.arrival_distribution()
    n_states, n_actions = M + 1, params.max_dispatch + 1
    s = np.arange(n_states)
    P = np.zeros((n_actions, n_states, n_states))
    C = np.zeros((n_states, n_actions))
    for a in range(n_actions):
        served = np.minimum(s, a * K)
        left = s - served
        if params.holding_charged == "after_dispatch":
            waiting = left.astype(float)
        elif params.holding_charged == "before_dispatch":
            waiting = s.astype(float)
        else:  # end_of_period: everyone still there after this period's arrivals (in expectation)
            waiting = left + params.mean_arrivals
        C[:, a] = a * params.dispatch_cost + params.holding_cost * waiting
        for arrival, p in zip(arrivals, probs):
            np.add.at(P[a], (s, np.minimum(left + arrival, M)), p)
    if n_actions == 2:
        labels: tuple = ("Do not dispatch", "Dispatch")
    else:
        labels = tuple("0 shuttles" if a == 0 else ("1 shuttle" if a == 1 else f"{a} shuttles") for a in range(n_actions))
    return TabularMDP(P, C, params.gamma, states=s, action_labels=labels, name="shuttle")
