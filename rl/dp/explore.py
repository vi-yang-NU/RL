"""The interactive explorer for Section 2 ("Explore the problem" in the dashboard's tab 2).

`explore(params, tol, steps)` runs the real solvers (rl/dp/solvers.py) on `build_mdp(params)` and collects
what the page shows, so nothing in the browser re-implements the model or the algorithms:

* the MDP itself: the cost table and, for every (state, action), the sparse list of successor states with
  their probabilities. The browser replays the first periods of any policy from these tables with a seeded
  random draw per period;
* the optimal values, Q-values and policy (policy iteration's exact solution; enumeration's when gamma = 1);
* the first `steps` steps of each algorithm: enumeration's value functions with 0, 1, ..., steps periods to
  go and the optimal decision at each; value iteration's first sweeps and their greedy policies; every policy
  that policy iteration visits with its exact value;
* for the finite-horizon problem, the optimal policy of each of the first `steps` periods (with T, T-1, ...
  periods to go), so the walkthrough can follow the time-dependent policy exactly, whatever T is.

`render_html(data)` puts that data into rl/dp/explore.html (the page; rl/dp/explore_engine.js is the
replay logic it shares with the unit tests). The dashboard embeds the result (ui/explore.py) and
`python run.py dp --explorer FILE` writes it to disk as a standalone page.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rl.dp import solvers
from rl.dp.shuttle import ShuttleParams, TabularMDP, build_mdp

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "explore.html"
ENGINE = HERE / "explore_engine.js"

METHOD_LABELS = {
    "enumeration": "Enumeration (T = {horizon})",
    "value_iteration": "Value iteration",
    "policy_iteration": "Policy iteration",
}


def sparse_successors(mdp: TabularMDP, eps: float = 0.0) -> list[list[list[list[float]]]]:
    """succ[s][a] = [[s', P(s' | s, a)], ...] for the successors with positive probability."""
    out = []
    for s in range(mdp.n_states):
        per_action = []
        for a in range(mdp.n_actions):
            row = mdp.transitions[a, s]
            idx = np.flatnonzero(row > eps)
            per_action.append([[int(j), float(row[j])] for j in idx])
        out.append(per_action)
    return out


def thin_history(values: list[float], keep: int = 10, cap: int = 600) -> list[list[float]]:
    """[[iteration, value], ...]: every one of the first `keep`, then at most `cap` evenly spaced points."""
    n = len(values)
    if n <= cap:
        idx = range(n)
    else:
        rest = np.linspace(keep, n - 1, cap - keep).round().astype(int)
        idx = sorted(set(range(min(keep, n))) | set(int(i) for i in rest))
    return [[int(i), float(values[i])] for i in idx]


def _policy_entry(mdp: TabularMDP, policy: np.ndarray | None) -> tuple[list[int] | None, int | None]:
    if policy is None:
        return None, None
    policy = np.asarray(policy, dtype=int)
    threshold = solvers.policy_threshold(policy) if mdp.n_actions == 2 else None
    return policy.tolist(), threshold


def _enumeration_block(mdp: TabularMDP, params: ShuttleParams, steps: int) -> dict[str, Any]:
    T = params.horizon
    full = solvers.enumeration(mdp, T, record=min(steps, T))
    # The first `steps` backward steps: V_k and the optimal decision with k periods to go (greedy w.r.t. V_{k-1}).
    recorded = full.info["values_by_steps_to_go"] or [np.zeros(mdp.n_states)]
    decisions = full.info["policies_by_steps_to_go"] or []
    out_steps = []
    for k, V in enumerate(recorded):
        policy, threshold = _policy_entry(mdp, decisions[k - 1] if k >= 1 else None)
        out_steps.append({
            "k": k, "values": V.tolist(), "policy": policy, "threshold": threshold,
            "change": float(full.history["max |V_t − V_{t+1}|"][k - 1]) if k >= 1 else None,
        })
    # The policies of the first min(steps, T) periods of the horizon: with T, T-1, ... periods to go. They are
    # the last backward steps, so continue the recursion from V_{T-m} for m more steps and record those.
    m = min(steps, T)
    first_periods = []
    if m > 0:
        head = solvers.enumeration(mdp, T - m)
        tail = solvers.enumeration(mdp, m, terminal_values=head.values, record=m)
        tail_policies = tail.info["policies_by_steps_to_go"]
        tail_values = tail.info["values_by_steps_to_go"]
        for t in range(m):
            to_go = T - t
            policy, threshold = _policy_entry(mdp, tail_policies[m - t - 1])
            first_periods.append({"t": t, "periods_to_go": to_go, "policy": policy, "threshold": threshold,
                                  "values": tail_values[m - t].tolist()})
    policy, threshold = _policy_entry(mdp, full.policy)
    return {
        "label": METHOD_LABELS["enumeration"].format(horizon=T),
        "iterations": T,
        "values": full.values.tolist(),
        "policy": policy,
        "threshold": threshold,
        "history": thin_history(full.history["max |V_t − V_{t+1}|"]),
        "history_label": "max |V_t − V_{t+1}|",
        "steps": out_steps,
        "first_periods": first_periods,
        "summary": f"T = {T} backward steps; final change {full.info['final_change']:.3g}",
    }


def _value_iteration_block(mdp: TabularMDP, tol: float, steps: int, max_iter: int) -> dict[str, Any]:
    if mdp.gamma >= 1.0:
        return {"label": METHOD_LABELS["value_iteration"], "error": "value iteration needs γ < 1 to converge"}
    result = solvers.value_iteration(mdp, tol=tol, max_iter=max_iter)
    # The first sweeps from V_0 = 0 are exactly backward induction with k periods to go.
    first = solvers.enumeration(mdp, min(steps, result.iterations), record=min(steps, result.iterations))
    recorded = first.info["values_by_steps_to_go"] or [np.zeros(mdp.n_states)]
    out_steps = []
    for k, V in enumerate(recorded):
        policy, threshold = _policy_entry(mdp, solvers.greedy_policy(mdp, V))
        out_steps.append({
            "k": k, "values": V.tolist(), "policy": policy, "threshold": threshold,
            "change": float(result.history["‖V_{n+1} − V_n‖∞"][k - 1]) if k >= 1 else None,
        })
    policy, threshold = _policy_entry(mdp, result.policy)
    return {
        "label": METHOD_LABELS["value_iteration"],
        "iterations": result.iterations,
        "values": result.values.tolist(),
        "policy": policy,
        "threshold": threshold,
        "history": thin_history(result.history["‖V_{n+1} − V_n‖∞"]),
        "history_label": "‖V_{n+1} − V_n‖∞",
        "tol": tol,
        "error_bound": float(result.info["error_bound"]),
        "converged": bool(result.info["converged"]),
        "steps": out_steps,
        "summary": f"{result.iterations} sweeps to ‖V_{{n+1}} − V_n‖ < {tol:g}; error bound {result.info['error_bound']:.2g}",
    }


def _policy_iteration_block(mdp: TabularMDP, max_iter: int) -> dict[str, Any]:
    if mdp.gamma >= 1.0:
        return {"label": METHOD_LABELS["policy_iteration"], "error": "policy evaluation by a linear solve needs γ < 1"}
    result = solvers.policy_iteration(mdp, max_iter=max_iter)
    out_steps = []
    previous = None
    for k, policy in enumerate(result.info["policies"]):
        policy = np.asarray(policy, dtype=int)
        values = solvers.evaluate_policy(mdp, policy)
        entry, threshold = _policy_entry(mdp, policy)
        out_steps.append({
            "k": k, "values": values.tolist(), "policy": entry, "threshold": threshold,
            "changed": int(np.count_nonzero(policy != previous)) if previous is not None else None,
        })
        previous = policy
    policy, threshold = _policy_entry(mdp, result.policy)
    return {
        "label": METHOD_LABELS["policy_iteration"],
        "iterations": result.iterations,
        "values": result.values.tolist(),
        "policy": policy,
        "threshold": threshold,
        "history": thin_history(result.history["states whose action changed"]),
        "history_label": "states whose action changed",
        "converged": bool(result.info["converged"]),
        "steps": out_steps,
        "summary": f"{len(out_steps) - 1} improvement step(s) from “never dispatch”; "
                   f"Bellman residual {result.info['bellman_residual']:.2g}",
    }


def explore(params: ShuttleParams, tol: float = 1e-8, steps: int = 10, max_iter: int = 100_000) -> dict[str, Any]:
    """Everything the explorer page shows for `params` (see the module docstring); JSON-serialisable."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    params.validate()
    mdp = build_mdp(params)
    methods = {
        "enumeration": _enumeration_block(mdp, params, steps),
        "value_iteration": _value_iteration_block(mdp, tol, steps, max_iter),
        "policy_iteration": _policy_iteration_block(mdp, max_iter),
    }
    if "error" in methods["policy_iteration"]:
        source, values = "enumeration", np.asarray(methods["enumeration"]["values"])
    else:
        source, values = "policy_iteration", np.asarray(methods["policy_iteration"]["values"])
    q = solvers.q_values(mdp, values)
    policy, threshold = _policy_entry(mdp, q.argmin(axis=1))
    arrivals, probs = params.arrival_distribution()
    return {
        "problem": {
            "name": "shuttle",
            **asdict(params),
            "arrivals": arrivals.tolist(),
            "arrival_probs": probs.tolist(),
            "mean_arrivals": params.mean_arrivals,
            "describe": params.describe(),
        },
        "mdp": {
            "n_states": mdp.n_states,
            "n_actions": mdp.n_actions,
            "gamma": float(mdp.gamma),
            "action_labels": list(mdp.action_labels),
            "cost": mdp.costs.tolist(),
            "succ": sparse_successors(mdp),
        },
        "optimal": {
            "source": source,
            "values": values.tolist(),
            "q": q.tolist(),
            "policy": policy,
            "threshold": threshold,
        },
        "methods": methods,
        "steps": steps,
        "tol": tol,
    }


def render_html(data: dict[str, Any], theme: str | None = None) -> str:
    """The explorer page with `data` embedded. `theme` = "light" | "dark" pins the colour scheme (the dashboard
    passes its own); None follows the viewer's system setting."""
    payload = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    engine = ENGINE.read_text(encoding="utf-8")
    attr = f' data-theme="{theme}"' if theme in ("light", "dark") else ""
    return html.replace("__THEME_ATTR__", attr).replace("__ENGINE__", engine).replace("__DATA__", payload)


def override_arguments(params: ShuttleParams, tol: float | None = None) -> list[str]:
    """The `python run.py dp` arguments that reproduce `params`: `--set KEY=VALUE` for every field that differs
    from the handout's defaults (in the form `rl.params.parse_overrides` reads), plus `--tol` when it is given
    and not the default 1e-8."""
    defaults = asdict(ShuttleParams())
    args: list[str] = []
    for key, value in asdict(params).items():
        if value != defaults[key]:
            text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
            args += ["--set", f"{key}={text}"]
    if tol is not None and tol != 1e-8:
        args += ["--tol", f"{tol:g}"]
    return args


def write_explorer(path: Path, params: ShuttleParams, tol: float = 1e-8, steps: int = 10, theme: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(explore(params, tol=tol, steps=steps), theme=theme), encoding="utf-8")
    return path
