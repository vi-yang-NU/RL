"""`python run.py dp`: solve the shuttle MDP and save results and plots to results/dp/."""
from __future__ import annotations

import argparse
from dataclasses import asdict

from rl.dp import solvers
from rl.dp.shuttle import ShuttleParams, build_mdp
from rl.outputs import FIGURE_BY_KEY, ROOT, dp_dir, write_json
from rl.plotting import plot_policy, plot_series, plot_value_function

# method -> (solver, figure key, plot title)
METHODS = {
    "enumeration": (solvers.enumeration, "dp_enumeration", "Enumeration (T = 500): optimal value function at t = 0"),
    "value_iteration": (solvers.value_iteration, "dp_value_iteration", "Value iteration (T = ∞): optimal value function"),
    "policy_iteration": (solvers.policy_iteration, "dp_policy_iteration", "Policy iteration: optimal policy"),
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all", help="which solver to run (default: all)")


def main(args: argparse.Namespace) -> int:
    params = ShuttleParams()
    out = dp_dir()
    write_json(out / "params.json", asdict(params))
    try:
        mdp = build_mdp(params)
    except NotImplementedError as e:
        print(f"TODO: {e}")
        return 1

    status = 0
    for name in METHODS if args.method == "all" else [args.method]:
        solve, figure_key, title = METHODS[name]
        print(f"== {name}")
        try:
            result = solve(mdp, params.horizon) if name == "enumeration" else solve(mdp)
        except NotImplementedError as e:
            print(f"   TODO: {e}")
            status = 1
            continue

        write_json(out / f"{name}.json", result.to_json())
        figure = FIGURE_BY_KEY[figure_key]
        if name == "policy_iteration":
            plot_policy(result.states, result.policy, figure.file, title, subtitle=params.describe())
        else:
            plot_value_function(result.states, result.values, figure.file, title, subtitle=params.describe())
        print(f"   saved {figure.file.relative_to(ROOT)}")
        if result.history:
            path = plot_series(
                result.history, out / f"{name}_convergence.png", f"{name.replace('_', ' ').capitalize()}: convergence",
                xlabel="Iteration", ylabel="Value",
            )
            print(f"   saved {path.relative_to(ROOT)}")
    return status
