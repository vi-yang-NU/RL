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
    """A figure with one axes, a left-aligned title, optional subtitle and axis labels. Long titles and subtitles
    are wrapped so they stay within the figure's width instead of widening the saved image."""
    import textwrap

    apply_style()
    fig, ax = plt.subplots()
    title = textwrap.fill(title, 62)
    if subtitle:
        subtitle = "\n".join(textwrap.fill(line, 92) for line in subtitle.splitlines())
    ax.set_title(title, pad=(20 + 13 * subtitle.count("\n")) if subtitle else 8, linespacing=1.2)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9, va="bottom", linespacing=1.4)
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


def _decorate(ax, title: str, xlabel: str, ylabel: str, subtitle: str | None = None) -> None:
    """Left-aligned title, optional subtitle and axis labels, as new_figure draws them."""
    import textwrap

    title = textwrap.fill(title, 62)
    if subtitle:
        subtitle = "\n".join(textwrap.fill(line, 92) for line in subtitle.splitlines())
    ax.set_title(title, pad=(20 + 13 * subtitle.count("\n")) if subtitle else 8, linespacing=1.2)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9, va="bottom", linespacing=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _headroom(ax, lo: float, hi: float, top: float = 0.24, ticks_up_to: float | None = None) -> None:
    """Y-limits from lo to hi with an empty band of `top` × the range above the data for a legend; tick labels
    stop at `ticks_up_to` (default hi), so the band does not suggest values the data cannot take."""
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - 0.04 * span, hi + top * span)
    limit = hi if ticks_up_to is None else ticks_up_to
    ax.set_yticks([t for t in ax.get_yticks() if lo - 0.04 * span - 1e-9 <= t <= limit + 1e-9])


def plot_training_curve(
    rewards: Sequence[float],
    path: Path,
    title: str,
    subtitle: str | None = None,
    window: int = 100,
    xlabel: str = "Training episode",
    ylabel: str = "Episode reward",
    entropy: Sequence[float] | None = None,
    frac_right: Sequence[float] | None = None,
    right_label: str = "right",
    n_actions: int = 2,
) -> Path:
    """The episode reward per training episode with its moving average and, when `entropy` and `frac_right` are
    given (one value per episode), a second panel with the policy's entropy as a fraction of its maximum ln(n_actions)
    and the fraction of "right" actions, where a collapse onto one action shows as the entropy falling to 0 while
    the fraction goes to 0 or 1. Legends sit in a band left free above the data, so they never cover it."""
    import textwrap

    r = np.asarray(rewards, dtype=float)
    episodes = np.arange(1, len(r) + 1)
    extra = entropy is not None and frac_right is not None and len(entropy) == len(r) and len(r) > 0
    apply_style()
    if extra:
        fig = plt.figure(figsize=(7.0, 6.8))
        grid = fig.add_gridspec(2, 1, height_ratios=[3, 1.7], hspace=0.45)
        ax = fig.add_subplot(grid[0])
    else:
        fig, ax = plt.subplots()
    _decorate(ax, title, xlabel, ylabel, subtitle)
    ax.plot(episodes, r, color=BLUE_LIGHT, linewidth=0.8, label="Episode reward")
    ax.plot(episodes, moving_average(r, window), color=BLUE_DARK, linewidth=2, label=f"{window}-episode moving average")
    ax.set_xlim(0, max(len(r), 2))
    if len(r):
        _headroom(ax, float(np.nanmin(r)), float(np.nanmax(r)))
    ax.legend(loc="upper left", ncol=2)
    if extra:
        ax2 = fig.add_subplot(grid[1])
        h_max = float(np.log(max(2, int(n_actions))))
        h = np.asarray([np.nan if v is None else v for v in entropy], dtype=float) / h_max
        f = np.asarray([np.nan if v is None else v for v in frac_right], dtype=float)
        ax2.set_title(textwrap.fill("The policy during training: its entropy as a fraction of the maximum "
                                    f"ln\u00a0{int(n_actions)} (1 = uniform) and the fraction of '{right_label}' actions", 72),
                      fontsize=9, pad=6, linespacing=1.25)
        ax2.plot(episodes, h, color=BLUE_LIGHT, linewidth=0.7)
        ax2.plot(episodes, moving_average(np.nan_to_num(h, nan=0.0), window), color=BLUE_DARK, linewidth=2,
                 label=f"entropy / ln {int(n_actions)}")
        ax2.plot(episodes, f, color="#f5b08a", linewidth=0.7)
        ax2.plot(episodes, moving_average(np.nan_to_num(f, nan=0.0), window), color=SERIES[1], linewidth=2,
                 label=f"fraction of '{right_label}'")
        ax2.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
        ax2.set_xlim(0, max(len(r), 2))
        ax2.set_ylim(-0.04, 1.36)
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("fraction")
        ax2.legend(loc="upper left", ncol=2, fontsize=8.5)
        ax2.text(0, -0.36, textwrap.fill(f"Thin lines: each episode; thick lines: the {window}-episode moving average; dotted: "
                                         "0.5. A collapse onto one action shows as the entropy falling to 0 while the "
                                         "fraction goes to 0 or 1.", 98),
                 transform=ax2.transAxes, fontsize=8, color=INK_SECONDARY, va="top", linespacing=1.3)
    return save(fig, path)


def reward_stats(rewards: Sequence[float]) -> dict:
    """Summary of evaluation rewards: episodes, mean, std, min, max, median, distinct values, the most common value."""
    r = np.asarray(rewards, dtype=float)
    values, counts = np.unique(r, return_counts=True)
    return {
        "episodes": int(len(r)), "mean": float(r.mean()), "std": float(r.std()), "min": float(r.min()), "max": float(r.max()),
        "median": float(np.median(r)), "distinct": int(len(values)),
        "most_common": [float(values[counts.argmax()]), int(counts.max())],
    }


def _integer_edges(lo: float, hi: float, target: int = 25) -> np.ndarray:
    """Bin edges for integer-valued rewards over [lo, hi]: one bin per integer when the range is short, otherwise a
    round bin width (2, 5, 10, 20, 25, 50, ...) giving about `target` bins; every bin edge falls between integers."""
    span = hi - lo
    if span <= 2.4 * target:
        return np.arange(lo - 0.5, hi + 1.5, 1.0)
    width = next((w for w in (2, 5, 10, 20, 25, 50, 100, 200, 500, 1000) if span / w <= target), 1000)
    edges = lo - 0.5 + width * np.arange(0, int(np.ceil(span / width)) + 1)
    edges[-1] = max(edges[-1], hi + 0.5)
    return edges


def _percent(count: int, total: int) -> str:
    """A share as a percentage: whole numbers, one decimal below 1 % so a single episode does not read as 0 %."""
    p = 100 * count / total
    return f"{p:.0f}%" if p >= 1 or p == 0 else f"{p:.1f}%"


def plot_reward_histogram(
    rewards: Sequence[float],
    path: Path,
    title: str,
    subtitle: str | None = None,
    bins: int | str = "auto",
    xlabel: str = "Episode reward",
    ylabel: str = "Number of episodes",
    value_range: tuple[float, float] | None = None,
    integer: bool = False,
    note: str | None = None,
    length_text: str | None = None,
) -> tuple[Path, float, float]:
    """Histogram of evaluation rewards with the statistics written on it, and the same rewards in the order they
    were played underneath. Returns (path, mean, std).

    `value_range` is the range of rewards the environment makes possible (e.g. (0, 500) for CartPole-v1, (−21, 21)
    for Pong-v5): the axis shows all of it with both ends marked, so a single bar at one end reads as "the best
    possible" or "the worst possible" instead of a bar in empty space. `integer` picks bins whose edges fall
    between integers (one per integer when the range is short). A single value gets its count written on its bar;
    two to eight values are counted one by one in the statistics box (adjacent one-wide bars leave no room for
    labels); `note` is printed under the figure (e.g. why every episode scored the same).
    """
    import textwrap

    r = np.asarray(rewards, dtype=float)
    stats = reward_stats(r)
    mean, std = stats["mean"], stats["std"]
    lo, hi = (float(value_range[0]), float(value_range[1])) if value_range else (float(r.min()), float(r.max()))
    if integer:
        edges = _integer_edges(lo, hi)
    elif value_range and lo < hi:
        edges = np.linspace(lo, hi, 31)
    else:
        edges = np.histogram_bin_edges(r, bins=bins) if r.min() < r.max() else np.array([r.min() - 0.5, r.max() + 0.5])
    counts, edges = np.histogram(r, bins=edges)
    span = max(hi - lo, 1.0)

    apply_style()
    fig = plt.figure(figsize=(7.0, 5.4))
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 1.15], hspace=0.75)
    ax = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1])
    subtitle = subtitle or (f"{stats['episodes']} episodes rolled out with the trained policy · mean {mean:.2f} · "
                            f"standard deviation {std:.2f}\nmin {stats['min']:g} · max {stats['max']:g} · median {stats['median']:g}"
                            + (f"\n{length_text}" if length_text else ""))
    subtitle = "\n".join(textwrap.fill(line, 92) for line in subtitle.splitlines())
    ax.set_title(textwrap.fill(title, 62), pad=20 + 13 * subtitle.count("\n"), linespacing=1.2)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9, va="bottom", linespacing=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if std > 0:
        ax.axvspan(mean - std, mean + std, color=MUTED, alpha=0.15, linewidth=0, label=f"±1 std ({std:.2f})")
    ax.bar((edges[:-1] + edges[1:]) / 2, counts, width=np.diff(edges), color=SERIES[0], edgecolor=SURFACE, linewidth=1.0,
           label="Episodes per bin")
    ax.axvline(mean, color=INK, linewidth=1.5, label=f"Mean ({mean:.2f})")
    if value_range:
        ax.axvline(lo, color=INK_SECONDARY, linewidth=1, linestyle=(0, (2, 3)), label=f"Lowest and highest possible ({lo:g}, {hi:g})")
        ax.axvline(hi, color=INK_SECONDARY, linewidth=1, linestyle=(0, (2, 3)))
    if stats["distinct"] == 1:  # one value: say how many episodes the bar holds (more values: counts in the box)
        for x, c in zip((edges[:-1] + edges[1:]) / 2, counts):
            if c:  # a bar at either end of the range gets its label pulled inside the axes
                ha = "left" if x < lo + 0.06 * span else "right" if x > hi - 0.06 * span else "center"
                x = lo - 0.5 if ha == "left" else hi + 0.5 if ha == "right" else x
                ax.text(x, c * 1.02, f"{c} episode{'s' if c != 1 else ''} ({100 * c / len(r):.0f}%)", ha=ha, va="bottom",
                        fontsize=8.5, color=INK, fontweight="semibold", zorder=6,  # on a patch, so the lines don't cross it
                        bbox={"boxstyle": "square,pad=0.2", "facecolor": SURFACE, "edgecolor": "none"})
    ax.set_xlim(lo - 0.03 * span - (0.5 if integer else 0), hi + 0.03 * span + (0.5 if integer else 0))
    ax.set_ylim(0, counts.max() * 1.22)
    ax.grid(axis="x", visible=False)
    box = (f"n = {stats['episodes']}\nmean {mean:.2f}\nstd {std:.2f}\nmin {stats['min']:g} · max {stats['max']:g}\n"
           f"median {stats['median']:g}\n{stats['distinct']} distinct value" + ("s" if stats["distinct"] != 1 else ""))
    if 2 <= stats["distinct"] <= 8:  # a few values, often in adjacent bars too narrow to label: count each one here
        values, value_counts = np.unique(r, return_counts=True)
        width = max(len(f"{v:g}") for v in values)
        box += "\nepisodes per reward:" + "".join(f"\n{v:>{width}g}: {c:>3d} ({_percent(c, len(r))})"
                                                   for v, c in zip(values, value_counts))
    at_left = mean > (lo + hi) / 2  # put the box where the bars are not, clear of the lowest/highest-possible lines
    ax.text(0.065 if at_left else 0.935, 0.95, box, transform=ax.transAxes, ha="left" if at_left else "right", va="top",
            fontsize=8.5, color=INK, linespacing=1.35, family="monospace", multialignment="left",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": SURFACE, "edgecolor": GRID})
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8.5)  # between the two panels

    ax2.set_title("The same rewards in the order the episodes were played", fontsize=9, pad=6)
    ax2.plot(np.arange(1, len(r) + 1), r, linestyle="none", marker=".", markersize=3, color=SERIES[0], alpha=0.85)
    ax2.axhline(mean, color=INK, linewidth=1)
    ax2.set_xlim(0, len(r) + 1)
    ax2.set_ylim(lo - 0.08 * span, hi + 0.08 * span)
    ax2.set_xlabel("Evaluation episode")
    ax2.set_ylabel("Reward")
    if note:  # under the figure, aligned with the plots' left edge
        ax2.text(0, -0.62, "\n".join(textwrap.wrap(note, 100)), transform=ax2.transAxes, fontsize=8.5, color=INK_SECONDARY,
                 va="top", linespacing=1.35)
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


def plot_baseline_comparison(
    runs: Sequence[Mapping],
    path: Path,
    title: str,
    subtitle: str | None = None,
    window: int = 100,
    note: str | None = None,
) -> Path:
    """Runs of the same problem side by side, e.g. Part 1 (no baseline) against Part 2 (with a baseline). Each run is
    a mapping with "label", "rewards" (one per training episode) and optionally "weights": the root mean square of
    the per-step weights on ∇ log π in each episode (G_t without a baseline, G_t − b_t with one), drawn in a second
    panel on a log scale, where a baseline that reduces the variance of the gradient estimate sits lower."""
    import textwrap

    with_weights = [run for run in runs if run.get("weights") is not None and len(run["weights"])]
    apply_style()
    if with_weights:
        fig = plt.figure(figsize=(7.0, 6.8))
        grid = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.45)
        ax = fig.add_subplot(grid[0])
    else:
        fig, ax = plt.subplots()
    _decorate(ax, title, "Training episode", "Episode reward", subtitle)
    lo, hi, longest = np.inf, -np.inf, 2
    for color, run in zip(SERIES, runs):
        r = np.asarray(run["rewards"], dtype=float)
        if not len(r):
            continue
        x = np.arange(1, len(r) + 1)
        ma = moving_average(r, window)
        ax.plot(x, r, color=color, linewidth=0.6, alpha=0.2)
        ax.plot(x, ma, color=color, linewidth=2.2,
                label=f"{run['label']}: last {min(window, len(r))} episodes average {ma[-1]:.1f}")
        lo, hi, longest = min(lo, float(np.nanmin(r))), max(hi, float(np.nanmax(r))), max(longest, len(r))
    ax.set_xlim(0, longest)
    if np.isfinite(lo):
        _headroom(ax, lo, hi, top=0.3)
    ax.legend(loc="upper left", fontsize=8.5)
    if with_weights:
        ax2 = fig.add_subplot(grid[1])
        ax2.set_title(textwrap.fill("Size of the weights on ∇ log π(a_t|s_t) in each update: the root mean square of "
                                    "G_t (no baseline) or of G_t − b(s_t) (baseline), log scale", 72),
                      fontsize=9, pad=6, linespacing=1.25)
        top = 0.0
        for color, run in zip(SERIES, runs):
            w = run.get("weights")
            if w is None or not len(w):
                continue
            w = np.asarray(w, dtype=float)
            x = np.arange(1, len(w) + 1)
            ax2.plot(x, w, color=color, linewidth=0.6, alpha=0.25)
            ax2.plot(x, moving_average(w, window), color=color, linewidth=2.2,
                     label=f"{run['label']}: mean {np.nanmean(w):.2f}")
            top = max(top, float(np.nanmax(w)))
        ax2.set_yscale("log")
        bottom = ax2.get_ylim()[0]
        ax2.set_ylim(bottom, top * 12)  # a band above the lines for the legend
        ax2.set_xlim(0, longest)
        ax2.set_xlabel("Training episode")
        ax2.set_ylabel("rms of the weights")
        ax2.legend(loc="upper left", fontsize=8.5, ncol=1)
    if note:
        anchor = ax2 if with_weights else ax
        anchor.text(0, -0.36 if with_weights else -0.2, "\n".join(textwrap.wrap(note, 98)), transform=anchor.transAxes,
                    fontsize=8, color=INK_SECONDARY, va="top", linespacing=1.3)
    return save(fig, path)
