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
    alt: str = ""  # Markdown alt text; "full-page": the PDF fits it into the room left under its text, up to 80 % of a page

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
                f"{label}: the episode reward against the number of training episodes, overlaid with the moving average "
                f"of the rewards over 100 episodes. The lower panel shows the policy itself per episode: its entropy and the "
                f"fraction of {'RIGHT' if env == 'pong' else 'push-right'} actions"
                + (", where a collapse onto one action would show. Trained with Adam at learning rate 10⁻⁴ instead of the "
                   "handout's 0.01, plus an entropy bonus of 0.01 per step: with 0.01 and no bonus the policy collapsed onto "
                   "a single action within about 150 episodes and lost every game 0 : 21 (results/pg/part1_pong_lr0.01)."
                   if env == "pong" else "."),
                command,
            ),
            Figure(
                f"{run}_eval",
                f"pg/{run}/eval_histogram.png",
                f"{label}: histogram of the 500 episode rewards (500 episodes rolled out with the trained policy), with the "
                f"mean and the standard deviation (also the minimum, maximum and median), over the whole range of rewards "
                f"the game allows; below it, the same 500 rewards in the order the episodes were played.",
                command,
            ),
            Figure(
                f"{run}_steps",
                f"pg/{run}/first_steps.png",
                f"{label}: the first 10 steps of one episode played by the trained policy, with the action taken, "
                f"the policy's probabilities, the reward and the reward-to-go at each step"
                + (": the frame the environment gives, what the handout's preprocess function makes of it, and the "
                   "frame difference the network receives (the dashboard's \"Watch the game\" panel replays the whole episode)."
                   if env == "pong" else " (the dashboard's \"Watch the game\" panel replays the whole episode)."),
                command,
                alt="full-page" if env == "pong" else "",
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
        Figure(
            "dp_tests_table",
            "dp/tests_table.png",
            "Section 2, validation: summary of the validation tests T1–T13, with what each one checks, how it validates "
            "the solver, its pass criterion and the result observed (the table in report/dp_tests_table.md). The "
            "harness's detailed results for each experiment follow.",
            "python scripts/tests_table.py",
            alt="full-page",
        ),
        Figure(
            "dp_explore_steps",
            "dp/explore_steps.png",
            "Section 2, validation: the dashboard's \"Explore the problem\" panel at the start of each of the first 10 "
            "periods under the optimal policy (handout parameters, empty station at t = 0, the explorer's default "
            "seed): customers waiting, the policy's decision, the previous period's cost, the discounted cost so far "
            "and the cost-to-go. The label above each screenshot gives that period's state, value, Q-values, action, "
            "cost, arrivals and next state.",
            "python scripts/explore_collage.py",
            alt="full-page",
        ),
    ],
    "pg1": [
        Figure(
            "pg1_tests_table",
            "pg/pg1_tests_table.png",
            "Policy Gradient Part 1, validation: summary of the validation experiments G1–G7, with what each one checks, how "
            "it validates the algorithm, its pass criterion and the result observed (the table in report/pg1_tests_table.md). "
            "The harness's detailed results for each experiment follow.",
            "python scripts/tests_table.py --source report/pg1_tests_table.md --out results/pg/pg1_tests_table.png",
            alt="full-page",
        ),
        *_pg_figures(1),
    ],
    "pg2": [
        Figure(
            "pg2_tests_table",
            "pg/pg2_tests_table.png",
            "Policy Gradient Part 2, validation: summary of the validation experiments B1–B5, with what each one checks, how "
            "it validates the baseline, its pass criterion and the result observed (the table in report/pg2_tests_table.md). "
            "The harness's detailed results for each experiment follow.",
            "python scripts/tests_table.py --source report/pg2_tests_table.md --out results/pg/pg2_tests_table.png",
            alt="full-page",
        ),
        *_pg_figures(2),
        *(
            Figure(
                f"compare_{env}",
                f"pg/compare_{env}.png",
                f"Part 2 ({letter}), {ENV_LABELS[env]}: the Part 1 run (no baseline) against the Part 2 run (value baseline "
                f"V(s)), with the same seed, learning rate and discount factor. Top: the episode rewards and their 100-episode "
                f"moving averages. Bottom: the root mean square of the weights that multiply ∇ log π(a_t|s_t) in each update, "
                f"G_t without the baseline and G_t − V(s_t) with it (log scale); the lower the line, the smaller the "
                f"variance of the gradient estimate, which is what the baseline is for.",
                f"python run.py pg --part 2 --env {env} --compare",
            )
            for letter, env in (("a", "cartpole"), ("b", "pong"))
        ),
    ],
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
