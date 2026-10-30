"""Solvers for finite MDPs with costs (Section 2 (a)-(c)) and the utilities the validation tests use.

Every solver takes a `TabularMDP` (rl/dp/shuttle.py) and returns a `SolverResult`. rl/dp/run.py saves it to
results/dp/<method>.json and draws the plot the handout asks for from `states` and `values` (or `policy`).

* `enumeration(mdp, horizon)`: finite-horizon backward induction (dynamic programming over the T periods).
  V_T = 0 (or `terminal_values`), V_t(s) = min_a [c(s, a) + gamma * E V_{t+1}(s')]; returns V_0.
* `value_iteration(mdp, tol)`: V_{n+1} = T V_n until ||V_{n+1} - V_n||_inf < tol. Because T is a
  gamma-contraction (proved in lean/RLVerify/Shuttle.lean, `bellman_contraction`), the result is within
  gamma * tol / (1 - gamma) of the optimal values (`vi_error_bound`); that bound is saved in `info`.
* `policy_iteration(mdp)`: policy evaluation by solving (I - gamma P_pi) V = c_pi exactly, then greedy
  improvement that keeps the current action unless another is strictly better (so it terminates).

Utilities: `q_values`, `greedy_policy`, `bellman_residual`, `evaluate_policy`, `simulate_policy`,
`policy_threshold`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rl.dp.shuttle import TabularMDP


@dataclass
class SolverResult:
    states: np.ndarray  # x-axis of the plots: number of customers waiting
    values: np.ndarray | None = None  # optimal value per state (enumeration: at time 0)
    policy: np.ndarray | None = None  # optimal action per state
    iterations: int | None = None
    history: dict[str, list[float]] = field(default_factory=dict)  # per-iteration diagnostics, plotted if set
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


# --- utilities -------------------------------------------------------------------------------------------


def q_values(mdp: TabularMDP, values: np.ndarray) -> np.ndarray:
    """Q(s, a) = c(s, a) + gamma * sum_s' P(s'|s, a) V(s'), shape (states, actions)."""
    return mdp.q_values(values)


def greedy_policy(mdp: TabularMDP, values: np.ndarray, current: np.ndarray | None = None, tie_tol: float = 0.0) -> np.ndarray:
    """The action minimising Q(s, .) in every state.

    With `current`, a state keeps its current action unless another action is better by more than
    `tie_tol` (relative to the magnitude of the best Q), which is what policy iteration needs to terminate.
    """
    Q = q_values(mdp, values)
    best = Q.argmin(axis=1)
    if current is None:
        return best
    current = np.asarray(current, dtype=int)
    idx = np.arange(mdp.n_states)
    slack = tie_tol * np.maximum(1.0, np.abs(Q[idx, best]))
    keep = Q[idx, current] <= Q[idx, best] + slack
    return np.where(keep, current, best)


def bellman_residual(mdp: TabularMDP, values: np.ndarray) -> float:
    """||T V - V||_inf: zero exactly at the optimal values."""
    return float(np.max(np.abs(mdp.bellman(values) - np.asarray(values, dtype=float))))


def evaluate_policy(mdp: TabularMDP, policy: np.ndarray, method: str = "solve", tol: float = 1e-12, max_iter: int = 1_000_000) -> np.ndarray:
    """V^pi, the expected discounted cost of following `policy` from every state.

    method="solve" solves the linear system (I - gamma P_pi) V = c_pi; method="iterative" applies
    V <- c_pi + gamma P_pi V until the change is below `tol`.
    """
    P_pi, c_pi = mdp.policy_transitions(policy)
    if method == "solve":
        if mdp.gamma >= 1.0:
            raise ValueError("policy evaluation by solving the linear system needs gamma < 1")
        return np.linalg.solve(np.eye(mdp.n_states) - mdp.gamma * P_pi, c_pi)
    if method == "iterative":
        V = np.zeros(mdp.n_states)
        for _ in range(max_iter):
            new = c_pi + mdp.gamma * P_pi @ V
            done = np.max(np.abs(new - V)) < tol
            V = new
            if done:
                return V
        warnings.warn(f"iterative policy evaluation did not reach tol={tol:g} in {max_iter} sweeps", stacklevel=2)
        return V
    raise ValueError(f"method must be 'solve' or 'iterative', got {method!r}")


def simulate_policy(
    mdp: TabularMDP, policy: np.ndarray, start: int, horizon: int, episodes: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Monte Carlo estimate of V^pi(start): roll `episodes` trajectories of `horizon` periods.

    Returns the mean discounted cost, its standard error, the per-episode costs and the first trajectory
    (states, actions, costs), which the validation tests print as a step-by-step trace.
    """
    rng = np.random.default_rng(seed)
    policy = np.asarray(policy, dtype=int)
    cum = np.cumsum(mdp.transitions, axis=2)
    s = np.full(episodes, int(start))
    total = np.zeros(episodes)
    trace = []
    discount = 1.0
    for t in range(horizon):
        a = policy[s]
        cost = mdp.costs[s, a]
        total += discount * cost
        u = rng.random(episodes)
        nxt = (cum[a, s, :] <= u[:, None]).sum(axis=1)
        nxt = np.minimum(nxt, mdp.n_states - 1)
        if t < 10:
            trace.append({"t": t, "state": int(s[0]), "action": int(a[0]), "cost": float(cost[0]), "next_state": int(nxt[0])})
        s = nxt
        discount *= mdp.gamma
    return {
        "mean": float(total.mean()),
        "stderr": float(total.std(ddof=1) / np.sqrt(episodes)) if episodes > 1 else float("nan"),
        "costs": total,
        "trace": trace,
        "truncation_bound": float(discount * mdp.costs.max() / (1 - mdp.gamma)) if mdp.gamma < 1 else float("inf"),
    }


def policy_threshold(policy: np.ndarray) -> int | None:
    """For a two-action policy: the smallest state where the action is 1, or None if it never is."""
    hits = np.flatnonzero(np.asarray(policy) >= 1)
    return int(hits[0]) if len(hits) else None


def is_monotone(policy: np.ndarray) -> bool:
    """True when the action index never decreases with the state (a threshold / control-limit policy)."""
    p = np.asarray(policy)
    return bool(np.all(np.diff(p) >= 0))


# --- (a) enumeration: finite-horizon backward induction ---------------------------------------------------


def enumeration(mdp: TabularMDP, horizon: int, terminal_values: np.ndarray | None = None, record: int = 0) -> SolverResult:
    """(a) Optimal values over a finite horizon of `horizon` periods, computed by backward induction.

    `values` is the optimal value function at time 0 (with `horizon` decisions left), `policy` the optimal
    first decision. `history` holds max_s |V_t - V_{t+1}| for each backward step. With `record` = k, `info`
    also keeps the value functions with 0, 1, ..., k periods to go (for step-by-step traces).
    """
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    V = np.zeros(mdp.n_states) if terminal_values is None else np.asarray(terminal_values, dtype=float).copy()
    if V.shape != (mdp.n_states,):
        raise ValueError("terminal_values must have one entry per state")
    changes: list[float] = []
    trace = [V.copy()]
    policies = []
    policy = None
    for steps_to_go in range(1, horizon + 1):
        Q = q_values(mdp, V)
        new = Q.min(axis=1)
        policy = Q.argmin(axis=1)
        changes.append(float(np.max(np.abs(new - V))))
        V = new
        if steps_to_go <= record:
            trace.append(V.copy())
            policies.append(policy.copy())
    return SolverResult(
        states=mdp.states,
        values=V,
        policy=policy,
        iterations=horizon,
        history={"max |V_t − V_{t+1}|": changes},
        info={
            "horizon": horizon,
            "values_by_steps_to_go": trace if record else None,
            "policies_by_steps_to_go": policies if record else None,
            "final_change": changes[-1] if changes else 0.0,
        },
    )


# --- (b) value iteration ---------------------------------------------------------------------------------


def value_iteration(mdp: TabularMDP, tol: float = 1e-6, max_iter: int = 100_000, initial_values: np.ndarray | None = None) -> SolverResult:
    """(b) Infinite-horizon optimal values by value iteration.

    Stops when ||V_{n+1} - V_n||_inf < tol. The optimal values are then within
    gamma * ||V_{n+1} - V_n|| / (1 - gamma) of `values` (saved as info["error_bound"]), and `policy` is the
    greedy policy for `values`.
    """
    if mdp.gamma >= 1.0:
        raise ValueError("value iteration needs gamma < 1 to converge (gamma = 1 gives an infinite-horizon total cost)")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    V = np.zeros(mdp.n_states) if initial_values is None else np.asarray(initial_values, dtype=float).copy()
    residuals: list[float] = []
    converged = False
    delta = float("inf")
    for _ in range(max_iter):
        new = mdp.bellman(V)
        delta = float(np.max(np.abs(new - V)))
        residuals.append(delta)
        V = new
        if delta < tol:
            converged = True
            break
    if not converged:
        warnings.warn(f"value iteration did not reach tol={tol:g} in {max_iter} iterations (last change {delta:.3g})", stacklevel=2)
    # Contraction ratios ||V_{n+2} - V_{n+1}|| / ||V_{n+1} - V_n||, which the theory bounds by gamma. Ratios
    # of residuals close to floating-point noise (relative to the size of the values) are meaningless and skipped.
    noise_floor = 1e8 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(V))))
    ratios = [b / a for a, b in zip(residuals[:-1], residuals[1:]) if a > noise_floor]
    return SolverResult(
        states=mdp.states,
        values=V,
        policy=greedy_policy(mdp, V),
        iterations=len(residuals),
        history={"‖V_{n+1} − V_n‖∞": residuals},
        info={
            "tol": tol,
            "converged": converged,
            "final_change": delta,
            "error_bound": mdp.gamma * delta / (1.0 - mdp.gamma),
            "bellman_residual": bellman_residual(mdp, V),
            "max_contraction_ratio": max(ratios) if ratios else None,
            "contraction_noise_floor": noise_floor,
        },
    )


# --- (c) policy iteration --------------------------------------------------------------------------------


def policy_iteration(
    mdp: TabularMDP,
    max_iter: int = 1_000,
    initial_policy: np.ndarray | None = None,
    evaluation: str = "solve",
    tie_tol: float = 1e-9,
) -> SolverResult:
    """(c) Optimal policy by policy iteration (evaluate, improve, repeat until the policy is stable).

    `values` is V^pi of the final policy. `history` counts the states whose action changed at each
    improvement step; info["policies"] keeps every policy visited.
    """
    policy = np.zeros(mdp.n_states, dtype=int) if initial_policy is None else np.asarray(initial_policy, dtype=int).copy()
    if policy.shape != (mdp.n_states,) or policy.min() < 0 or policy.max() >= mdp.n_actions:
        raise ValueError("initial_policy must give one valid action index per state")
    changes: list[float] = []
    value_changes: list[float] = []
    policies = [policy.copy()]
    converged = False
    V = evaluate_policy(mdp, policy, evaluation)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        new_policy = greedy_policy(mdp, V, current=policy, tie_tol=tie_tol)
        changed = int(np.count_nonzero(new_policy != policy))
        changes.append(float(changed))
        if changed == 0:
            converged = True
            break
        policy = new_policy
        policies.append(policy.copy())
        new_V = evaluate_policy(mdp, policy, evaluation)
        value_changes.append(float(np.max(np.abs(new_V - V))))
        V = new_V
    if not converged:
        warnings.warn(f"policy iteration did not stabilise in {max_iter} iterations", stacklevel=2)
    return SolverResult(
        states=mdp.states,
        values=V,
        policy=policy,
        iterations=iterations,
        history={"states whose action changed": changes, "‖V^π_{k+1} − V^π_k‖∞": value_changes},
        info={
            "converged": converged,
            "evaluation": evaluation,
            "policies": [p.tolist() for p in policies],
            "bellman_residual": bellman_residual(mdp, V),
            "threshold": policy_threshold(policy) if mdp.n_actions == 2 else None,
        },
    )
