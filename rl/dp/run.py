"""`python run.py dp`: solve the shuttle MDP and save results and plots to results/dp/.

    python run.py dp                                   all three methods with the handout's parameters
    python run.py dp --method value_iteration          one method
    python run.py dp --set capacity=10 --set gamma=0.9 any ShuttleParams field (see rl/dp/shuttle.py)
    python run.py dp --tol 1e-8 --max-iter 5000        value iteration / policy iteration controls
    python run.py dp --explorer results/dp/explore.html  also write the interactive explorer page
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rl.dp import solvers
from rl.dp.shuttle import ShuttleParams, build_mdp
from rl.outputs import FIGURE_BY_KEY, ROOT, dp_dir, write_json
from rl.params import apply_overrides, parse_overrides
from rl.plotting import plot_policy, plot_series, plot_value_function

# method -> (figure key, plot title)
METHODS = {
    "enumeration": ("dp_enumeration", "Enumeration (T = {horizon}): optimal value function at t = 0"),
    "value_iteration": ("dp_value_iteration", "Value iteration (T = ∞): optimal value function"),
    "policy_iteration": ("dp_policy_iteration", "Policy iteration: optimal policy"),
}
VALUE_LABEL = "Optimal expected discounted cost V(s)"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all", help="which solver to run (default: all)")
    parser.add_argument("--tol", type=float, default=1e-8, help="value iteration stops when the values change less than this")
    parser.add_argument("--max-iter", type=int, default=100_000, help="iteration cap for value and policy iteration")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a ShuttleParams field, e.g. --set capacity=10 --set arrivals=[1,2,3] (repeatable)")
    parser.add_argument("--backend", choices=["numpy", "cuda"], default="numpy",
                        help="numpy (default) or the CUDA kernels in rl/cuda (Numba; the CPU simulator without a GPU)")
    parser.add_argument("--explorer", metavar="FILE", default=None,
                        help="also write the interactive explorer page (dashboard tab 2) for these parameters to FILE")


def solve(name: str, mdp, params: ShuttleParams, tol: float = 1e-8, max_iter: int = 100_000, backend: str = "numpy") -> solvers.SolverResult:
    if backend == "cuda":
        from rl.cuda import dp as cuda_dp

        if name == "enumeration":
            return cuda_dp.enumeration_gpu(mdp, params.horizon)
        if name == "value_iteration":
            return cuda_dp.value_iteration_gpu(mdp, tol=tol, max_iter=max_iter)
        if name == "policy_iteration":
            return cuda_dp.policy_iteration_gpu(mdp, max_iter=max_iter)
    elif backend == "numpy":
        if name == "enumeration":
            return solvers.enumeration(mdp, params.horizon)
        if name == "value_iteration":
            return solvers.value_iteration(mdp, tol=tol, max_iter=max_iter)
        if name == "policy_iteration":
            return solvers.policy_iteration(mdp, max_iter=max_iter)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    raise ValueError(f"unknown method {name!r}")


def summary(name: str, result: solvers.SolverResult, mdp) -> str:
    parts = []
    if result.values is not None:
        parts.append(f"V(0) = {result.values[0]:.4f}, V({int(mdp.states[-1])}) = {result.values[-1]:.4f}")
    if result.policy is not None and mdp.n_actions == 2:
        threshold = solvers.policy_threshold(result.policy)
        parts.append("never dispatches" if threshold is None else f"dispatch when s ≥ {threshold}")
    if name == "value_iteration":
        parts.append(f"{result.iterations} iterations, error bound {result.info['error_bound']:.2e}")
    elif name == "policy_iteration":
        parts.append(f"{result.iterations} improvement steps")
    if "backend" in result.info:
        parts.append(f"{result.info['backend']} on {result.info['device']}, {result.info['kernel']}, {result.info['time_s']:.3f} s")
    return "; ".join(parts)


def main(args: argparse.Namespace) -> int:
    try:
        params = apply_overrides(ShuttleParams(), parse_overrides(args.set))
    except ValueError as e:
        print(f"error: {e}")
        return 2
    out = dp_dir()
    write_json(out / "params.json", asdict(params))
    if args.backend == "cuda":
        from rl.cuda import dp as cuda_dp
        from rl.cuda.backend import backend_name, device_summary

        print(f"backend: {backend_name()} on {device_summary()} (the MDP tables are built on the device by init_shuttle_tables)")
        cost, succ_idx, succ_p = cuda_dp.shuttle_tables_gpu(params)
        labels = build_mdp(params).action_labels
        mdp = cuda_dp.mdp_from_tables(cost, succ_idx, succ_p, params.gamma, labels)
    else:
        mdp = build_mdp(params)
    print(f"shuttle MDP: {mdp.n_states} states, {mdp.n_actions} actions; {params.describe()}")

    results = {}
    for name in METHODS if args.method == "all" else [args.method]:
        figure_key, title = METHODS[name]
        title = title.format(horizon=params.horizon)
        print(f"== {name}")
        result = solve(name, mdp, params, tol=args.tol, max_iter=args.max_iter, backend=args.backend)
        results[name] = result
        write_json(out / f"{name}.json", result.to_json())
        figure = FIGURE_BY_KEY[figure_key]
        if name == "policy_iteration":
            plot_policy(result.states, result.policy, figure.file, title, subtitle=params.describe(), action_labels=mdp.action_labels)
        else:
            plot_value_function(result.states, result.values, figure.file, title, subtitle=params.describe(), ylabel=VALUE_LABEL)
        print(f"   {summary(name, result, mdp)}")
        print(f"   saved {figure.file.relative_to(ROOT)}")
        history = {k: v for k, v in result.history.items() if v}
        if history:
            path = plot_series(
                history, out / f"{name}_convergence.png", f"{name.replace('_', ' ').capitalize()}: convergence",
                xlabel="Iteration", ylabel="Value", logy=name != "policy_iteration",
            )
            print(f"   saved {path.relative_to(ROOT)}")

    if args.explorer:
        from rl.dp.explore import write_explorer

        print(f"   saved {write_explorer(Path(args.explorer), params, tol=args.tol)}")
    if len(results) > 1:
        values = {k: r.values for k, r in results.items() if r.values is not None}
        names = list(values)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = float(np.max(np.abs(values[names[i]] - values[names[j]])))
                print(f"max |V_{names[i]} − V_{names[j]}| = {diff:.3g}")
    return 0
