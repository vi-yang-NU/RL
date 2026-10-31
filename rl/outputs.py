"""Where every part of the assignment writes its results.

The run scripts, the test harness, the dashboard and the report builder all go through
this module, so a figure saved by `python run.py dp` is the one the UI and report show.
FIGURES lists the plots the handout requires for each section.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

SECTIONS = {
    "dp": "Value and Policy Iteration",
    "pg1": "Policy Gradient: Part 1",
    "pg2": "Policy Gradient: Part 2",
}
ENV_LABELS = {"cartpole": "CartPole-v1", "pong": "Pong-v5"}


def dp_dir(tag: str | None = None) -> Path:
    """results/dp/ (the handout run the report shows) or results/dp_<tag>/ for a run kept apart from it."""
    return RESULTS / (f"dp_{tag}" if tag else "dp")


def pg_run_name(part: int, env: str, tag: str | None = None) -> str:
    name = f"part{part}_{env}"
    return f"{name}_{tag}" if tag else name


def pg_dir(part: int, env: str, tag: str | None = None) -> Path:
    return RESULTS / "pg" / pg_run_name(part, env, tag)


def tests_dir() -> Path:
    return RESULTS / "tests"


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_to_builtin))
    return path


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _to_builtin(obj: Any) -> Any:
    """json.dumps fallback for numpy scalars/arrays, tensors and paths."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


@dataclass(frozen=True)
class Figure:
    key: str
    path: str  # relative to results/
    caption: str
    command: str  # the command that produces it

    @property
    def file(self) -> Path:
        return RESULTS / self.path


def _pg_figures(part: int) -> list[Figure]:
    figures = []
    for letter, env in (("a", "cartpole"), ("b", "pong")):
        run = pg_run_name(part, env)
        label = f"Part {part} ({letter}), {ENV_LABELS[env]}"
        command = f"python run.py pg --part {part} --env {env}"
        figures += [
            Figure(
                f"{run}_train",
                f"pg/{run}/training_curve.png",
                f"{label}: episode reward versus training episode, with the 100-episode moving average.",
                command,
            ),
            Figure(
                f"{run}_eval",
                f"pg/{run}/eval_histogram.png",
                f"{label}: histogram of the rewards of 500 episodes rolled out with the trained policy.",
                command,
            ),
        ]
    return figures


FIGURES: dict[str, list[Figure]] = {
    "dp": [
        Figure(
            "dp_enumeration",
            "dp/enumeration_value.png",
            "Section 2 (a), enumeration with T = 500: optimal value function at time 0 versus number of customers waiting.",
            "python run.py dp --method enumeration",
        ),
        Figure(
            "dp_value_iteration",
            "dp/value_iteration_value.png",
            "Section 2 (b), value iteration (T = ∞): optimal value function versus number of customers waiting.",
            "python run.py dp --method value_iteration",
        ),
        Figure(
            "dp_policy_iteration",
            "dp/policy_iteration_policy.png",
            "Section 2 (c), policy iteration: optimal policy versus number of customers waiting.",
            "python run.py dp --method policy_iteration",
        ),
    ],
    "pg1": _pg_figures(1),
    "pg2": _pg_figures(2),
}
FIGURE_BY_KEY = {f.key: f for figures in FIGURES.values() for f in figures}


class ProgressLogger:
    """Per-episode training log.

    Appends one JSON record per episode to <run_dir>/progress.jsonl (the dashboard reads it
    to draw a live training curve) and prints a line every `print_every` episodes.
    """

    def __init__(self, run_dir: Path, print_every: int = 10, resume: bool = False):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "progress.jsonl"
        if not resume:
            self.path.write_text("")
        self.print_every = print_every
        self.start = time.time()

    def log(self, episode: int, reward: float, **extra: Any) -> None:
        record = {"episode": episode, "reward": float(reward), "time_s": round(time.time() - self.start, 2), **extra}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=_to_builtin) + "\n")
        if episode % self.print_every == 0:
            fields = "".join(f"  {k} {v:.4g}" if isinstance(v, float) else f"  {k} {v}" for k, v in extra.items())
            print(f"episode {episode:>6}  reward {reward:9.2f}{fields}  [{record['time_s']:.0f}s]", flush=True)
