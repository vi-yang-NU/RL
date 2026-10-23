"""Plot helpers so every figure in the report has the same labeled, formatted look.

All helpers save a PNG and close the figure. They only draw what they are given; the
numbers come from the solvers and training code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless, so plotting works over ssh on deepdish

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Chart ink and a colorblind-validated categorical order (blue, orange, aqua, ...).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_LIGHT = "#86b6ef"  # raw / noisy series
BLUE_DARK = "#1c5cab"  # smoothed series drawn over it


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 4.2),
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.6,
            "axes.prop_cycle": plt.cycler(color=SERIES),
        }
    )


def new_figure(title: str, xlabel: str, ylabel: str, subtitle: str | None = None):
    """A figure with one axes, a left-aligned title, optional subtitle and axis labels."""
    apply_style()
    fig, ax = plt.subplots()
    ax.set_title(title, pad=20 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9, va="bottom")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def moving_average(values: Sequence[float], window: int = 100) -> np.ndarray:
    """Trailing mean over the last `window` values (over fewer at the start of the series)."""
    x = np.asarray(values, dtype=float)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    end = np.arange(1, len(x) + 1)
    start = np.maximum(0, end - window)
    return (csum[end] - csum[start]) / (end - start)


def plot_value_function(
    states: Sequence[float],
    values: Sequence[float],
    path: Path,
    title: str,
    subtitle: str | None = None,
    xlabel: str = "Customers waiting at the station",
    ylabel: str = "Optimal value V(s)",
) -> Path:
    fig, ax = new_figure(title, xlabel, ylabel, subtitle)
    ax.plot(states, values, color=SERIES[0], linewidth=2)
    ax.set_xlim(min(states), max(states))
    return save(fig, path)


def plot_policy(
    states: Sequence[float],
    policy: Sequence[int],
    path: Path,
    title: str,
    subtitle: str | None = None,
    action_labels: Sequence[str] = ("Do not dispatch", "Dispatch"),
    xlabel: str = "Customers waiting at the station",
    ylabel: str = "Optimal action",
) -> Path:
    fig, ax = new_figure(title, xlabel, ylabel, subtitle)
    ax.step(states, policy, where="mid", color=SERIES[0], linewidth=2)
    ax.set_yticks(range(len(action_labels)), list(action_labels))
    ax.set_ylim(-0.25, len(action_labels) - 0.75)
    ax.set_xlim(min(states), max(states))
    ax.grid(axis="y", visible=False)
    return save(fig, path)


def plot_training_curve(
    rewards: Sequence[float],
    path: Path,
    title: str,
    subtitle: str | None = None,
    window: int = 100,
    xlabel: str = "Training episode",
    ylabel: str = "Episode reward",
) -> Path:
    episodes = np.arange(1, len(rewards) + 1)
    fig, ax = new_figure(title, xlabel, ylabel, subtitle)
    ax.plot(episodes, rewards, color=BLUE_LIGHT, linewidth=0.8, label="Episode reward")
    ax.plot(episodes, moving_average(rewards, window), color=BLUE_DARK, linewidth=2, label=f"{window}-episode moving average")
    ax.set_xlim(1, max(len(rewards), 2))
    ax.legend(loc="best")
    return save(fig, path)


def plot_reward_histogram(
    rewards: Sequence[float],
    path: Path,
    title: str,
    subtitle: str | None = None,
    bins: int | str = "auto",
    xlabel: str = "Episode reward",
    ylabel: str = "Number of episodes",
) -> tuple[Path, float, float]:
    """Histogram with the mean and a ±1 standard deviation band. Returns (path, mean, std)."""
    r = np.asarray(rewards, dtype=float)
    mean, std = float(r.mean()), float(r.std())
    fig, ax = new_figure(title, xlabel, ylabel, subtitle or f"{len(r)} evaluation episodes")
    ax.axvspan(mean - std, mean + std, color=MUTED, alpha=0.15, linewidth=0, label=f"±1 std ({std:.2f})")
    ax.hist(r, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(mean, color=INK, linewidth=1.5, label=f"Mean ({mean:.2f})")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="best")
    return save(fig, path), mean, std


def plot_series(
    series: Mapping[str, Sequence[float]],
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    subtitle: str | None = None,
    x: Sequence[float] | None = None,
    logy: bool = False,
) -> Path:
    """Line chart of up to eight named series sharing one y-axis (e.g. convergence diagnostics)."""
    if len(series) > len(SERIES):
        raise ValueError(f"plot_series takes at most {len(SERIES)} series; split them across figures")
    fig, ax = new_figure(title, xlabel, ylabel, subtitle)
    for color, (label, ys) in zip(SERIES, series.items()):
        xs = x if x is not None else np.arange(1, len(ys) + 1)
        ax.plot(xs, ys, color=color, linewidth=2, label=label)
    if logy:
        ax.set_yscale("log")
    if len(series) > 1:
        ax.legend(loc="best")
    return save(fig, path)
