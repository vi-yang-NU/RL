"""Watching an episode: record one game of CartPole-v1 or Pong-v5 frame by frame, save it, show it.

`record_episode` plays one episode exactly as `rl.pg.reinforce.run_episode` does (same adapter, same sampling
from the policy's softmax) with the environment created in `render_mode="rgb_array"`, and keeps everything the
dashboard's "Watch the game" panel and the report's first-steps figure show:

* the frame the environment renders at every step (for Pong the frame *is* the observation);
* for Pong, what the handout's `preprocess` makes of each frame (the 80x80 binary image) and the network's input
  (that image minus the previous one, `PongAdapter`); for CartPole the four state variables;
* the action taken, the policy's probabilities for the allowed actions, the reward and the reward-to-go G_t.

`save_recording` / `load_recording` keep a recording as a compressed .npz (`python run.py pg` writes one with the
trained policy at the end of every run, so the dashboard can replay the trained agent without the checkpoint);
`viewer_data` + `render_html` turn one into the viewer page rl/pg/watch.html; `first_steps_figure` draws the
first ten steps for the report. `python run.py watch` records an episode from the command line.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rl.outputs import ENV_LABELS, RESULTS, ROOT, read_json
from rl.pg.envs import ENV_IDS, GAMMA, PONG_ACTIONS, PONG_FRAME, SECONDS_PER_STEP, make_adapter, make_env

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "watch.html"
UNTRAINED = "Untrained policy (random initial weights)"
CROP = (35, 195)  # the rows of the frame that `preprocess` keeps
CARTPOLE_LIMITS = (2.4, None, 12 * 2 * np.pi / 360, None)  # |x| and |θ| at which CartPole-v1 ends the episode
ACTIONS = {
    # per environment: (label per policy action, short label) in the order of the policy's outputs
    "cartpole": (("0: push the cart to the left", "1: push the cart to the right"), ("Left", "Right")),
    "pong": (("RIGHT (2): paddle up", "LEFT (3): paddle down"), ("RIGHT ↑", "LEFT ↓")),
}
FPS = {"cartpole": 50, "pong": 15}  # agent steps per second in real time (Pong-v5: 60 frames/s, frameskip 4)
NOTES = {
    "cartpole": "CartPole-v1: the state is (x, ẋ, θ, θ̇); the episode ends when |x| > 2.4 or |θ| > 12°, or after "
                "500 steps; every step gives reward 1.",
    "pong": "ALE/Pong-v5 (gymnasium id of Pong-v5): frameskip 4, sticky actions with probability 0.25, 6 actions; "
            "the policy chooses between RIGHT (2) and LEFT (3) only. Reward +1 when the agent scores, −1 when the "
            "opponent does; a game ends at 21 points.",
}


@dataclass
class Recording:
    """One episode. Index t = 0 … T−1 is a step (state s_t, action a_t, reward r_t); the state arrays have T+1
    entries, the last being the final state, in which no action is taken."""

    env: str
    policy_label: str
    seed: int
    gamma: float
    frames: np.ndarray  # (T+1, H, W, 3) uint8: the environment's rendering of s_t
    actions: np.ndarray  # (T,) policy action index
    env_actions: np.ndarray  # (T,) the action sent to the environment
    probs: np.ndarray  # (T, n_actions) π(· | s_t)
    rewards: np.ndarray  # (T,)
    observations: np.ndarray | None = None  # CartPole: (T+1, 4) state vectors
    pre: np.ndarray | None = None  # Pong: (T+1, 80, 80) uint8, preprocess(frame_t)
    inputs: np.ndarray | None = None  # Pong: (T+1, 80, 80) int8, the network input x_t (frame difference)
    greedy: bool = False
    truncated: bool = False  # stopped by the step cap, not by the environment
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return int(len(self.rewards))

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum()) if self.steps else 0.0

    @property
    def env_id(self) -> str:
        return ENV_IDS[self.env]

    def returns(self) -> np.ndarray:
        """Reward-to-go G_t = Σ_{k≥t} γ^(k−t) r_k, the weight of log π(a_t|s_t) in the gradient."""
        from rl.pg.reinforce import discounted_returns

        return discounted_returns(self.rewards, self.gamma)

    def score(self) -> np.ndarray:
        """Pong: (T+1, 2) points of the agent and the opponent before each step."""
        agent = np.concatenate([[0], np.cumsum(self.rewards > 0)])
        opponent = np.concatenate([[0], np.cumsum(self.rewards < 0)])
        return np.stack([agent, opponent], axis=1).astype(int)


# --- recording ----------------------------------------------------------------------------------

def headless_pygame() -> None:
    """Make pygame (CartPole's renderer) draw off-screen: SDL's dummy video and audio drivers, no window.

    Frames are the same; without this, initialising pygame opens a connection to the window system, which fails on
    a server without a display and, on macOS, aborts the whole process with an NSInternalInconsistencyException
    when it happens outside the main thread, as it does in the dashboard (Streamlit runs scripts in worker threads).
    Set SDL_VIDEODRIVER yourself to override.
    """
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def render_env(name: str):
    """The environment with frame rendering on (`env.render()` returns the RGB frame; CartPole draws with pygame)."""
    headless_pygame()
    return make_env(name, render_mode="rgb_array")


def record_episode(cfg: Any, env: Any, policy: Any, *, seed: int | None = None, max_steps: int = 0,
                   greedy: bool = False, policy_label: str = UNTRAINED) -> Recording:
    """Play one episode of `env` (created with render_mode="rgb_array") with `policy` and record every step.

    Actions are sampled from the policy's softmax as in training and evaluation (or taken greedily); `max_steps`
    > 0 cuts the episode off after that many steps (the recording says so in `truncated`).
    """
    import torch
    from torch.distributions import Categorical

    from rl.pg.options import options_for

    opts = options_for(cfg)
    adapter = make_adapter(cfg, env, opts)
    device = next(policy.parameters()).device
    seed = cfg.seed if seed is None else int(seed)
    torch.manual_seed(seed)
    obs = adapter.reset(env, seed)
    frames, actions, env_actions, probs, rewards = [env.render()], [], [], [], []
    observations, pre, inputs = [], [], []
    is_pong = cfg.env == "pong"

    def keep_state(x: np.ndarray) -> None:
        if is_pong:
            inputs.append(np.asarray(x, dtype=np.float32).reshape(PONG_FRAME).astype(np.int8))
            pre.append(adapter.previous.astype(np.uint8) if adapter.previous is not None else np.zeros(PONG_FRAME, np.uint8))
        else:
            observations.append(np.asarray(x, dtype=np.float32).ravel())

    keep_state(obs)
    truncated = False
    with torch.no_grad():
        while True:
            logits = policy(torch.as_tensor(obs, device=device).unsqueeze(0))
            p = torch.softmax(logits, dim=-1)[0]
            action = int(logits.argmax(dim=-1).item() if greedy else Categorical(logits=logits).sample().item())
            next_obs, reward, done, _ = adapter.step(env, action)
            actions.append(action)
            env_actions.append(int(adapter.action(action)))
            probs.append(p.cpu().numpy().astype(np.float32))
            rewards.append(float(reward))
            frames.append(env.render())
            keep_state(next_obs)
            obs = next_obs
            if done:
                break
            if max_steps and len(rewards) >= max_steps:
                truncated = True
                break
    return Recording(
        env=cfg.env, policy_label=policy_label, seed=seed, gamma=float(cfg.gamma),
        frames=np.stack(frames).astype(np.uint8), actions=np.asarray(actions, dtype=np.int16),
        env_actions=np.asarray(env_actions, dtype=np.int16), probs=np.stack(probs), rewards=np.asarray(rewards, dtype=np.float32),
        observations=np.stack(observations) if observations else None,
        pre=np.stack(pre) if pre else None, inputs=np.stack(inputs) if inputs else None,
        greedy=greedy, truncated=truncated, extra={"options": opts.describe()},
    )


def load_run(run_dir: Path, device: str = "cpu") -> tuple[Any, Any]:
    """(cfg, policy) of a finished run: results/pg/<run>/config.json and policy.pt."""
    import torch

    from rl.pg.policy import build_policy
    from rl.pg.reinforce import TrainConfig

    run_dir = Path(run_dir)
    config = read_json(run_dir / "config.json")
    if not config:
        raise FileNotFoundError(f"{run_dir / 'config.json'} not found")
    names = {f.name for f in fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in config.items() if k in names})
    cfg.device = device
    env = make_env(cfg.env)
    try:
        policy = build_policy(cfg, env).to(device)
    finally:
        env.close()
    policy.load_state_dict(torch.load(run_dir / "policy.pt", map_location=device))
    policy.eval()
    return cfg, policy


def untrained(env_name: str, seed: int = 0, **extra: Any) -> tuple[Any, Any]:
    """(cfg, policy) with freshly initialised weights (what training starts from)."""
    import torch

    from rl.pg.policy import build_policy
    from rl.pg.reinforce import TrainConfig

    cfg = TrainConfig(env=env_name, part=1, gamma=GAMMA[env_name], seed=int(seed), device="cpu", extra=dict(extra))
    torch.manual_seed(cfg.seed)
    env = make_env(env_name)
    try:
        return cfg, build_policy(cfg, env)
    finally:
        env.close()


def record(env_name: str, run: str | Path | None = None, *, seed: int = 0, max_steps: int = 0, greedy: bool = False) -> Recording:
    """Record one episode with the policy of run `run` (a results/pg folder or its name) or, without one, with an
    untrained policy."""
    if run:
        run_dir = Path(run) if Path(run).is_absolute() or Path(run).exists() else RESULTS / "pg" / str(run)
        cfg, policy = load_run(run_dir)
        if cfg.env != env_name:
            raise ValueError(f"{run_dir.name} was trained on {cfg.env}, not {env_name}")
        label = f"Trained policy: {run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir}/policy.pt"
    else:
        cfg, policy = untrained(env_name, seed)
        label = UNTRAINED
    env = render_env(env_name)
    try:
        return record_episode(cfg, env, policy, seed=seed, max_steps=max_steps, greedy=greedy, policy_label=label)
    finally:
        env.close()


# --- files --------------------------------------------------------------------------------------

_ARRAYS = ("frames", "actions", "env_actions", "probs", "rewards", "observations", "pre", "inputs")


def save_recording(rec: Recording, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: getattr(rec, name) for name in _ARRAYS if getattr(rec, name) is not None}
    meta = {"env": rec.env, "policy_label": rec.policy_label, "seed": rec.seed, "gamma": rec.gamma, "greedy": rec.greedy,
            "truncated": rec.truncated, "extra": rec.extra}
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
    return path


def load_recording(path: Path) -> Recording:
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        arrays = {name: data[name] for name in _ARRAYS if name in data.files}
    return Recording(**meta, **arrays)


def policy_sources(env_name: str, part: int | None = None) -> list[dict[str, Any]]:
    """What the dashboard can replay for `env_name`: saved episodes and checkpoints of finished runs (of `part`,
    or any part), then the untrained policy. Each entry: kind ("episode", "checkpoint", "untrained"), label, path."""
    sources = []
    pattern = f"part{part}_{env_name}*" if part else f"part*_{env_name}*"
    for run_dir in sorted((RESULTS / "pg").glob(pattern)):
        rel = run_dir.relative_to(ROOT)
        if (run_dir / "episode.npz").exists():
            sources.append({"kind": "episode", "label": f"Trained policy of {rel} (the episode recorded at the end of the run)",
                            "path": str(run_dir / "episode.npz")})
        if (run_dir / "policy.pt").exists() and (run_dir / "config.json").exists():
            sources.append({"kind": "checkpoint", "label": f"Trained policy of {rel}: play a new episode (policy.pt)",
                            "path": str(run_dir)})
    sources.append({"kind": "untrained", "label": UNTRAINED, "path": ""})
    return sources


# --- the viewer page ------------------------------------------------------------------------------

def _png_uri(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _frame_images(frames: np.ndarray) -> list:
    """PIL images of the frames (RGB PNGs: a Pong frame is about 0.7 KB, a CartPole frame about 2 KB)."""
    from PIL import Image

    return [Image.fromarray(np.ascontiguousarray(frame), "RGB") for frame in frames]


def _overlay_images(pre: np.ndarray, inputs: np.ndarray) -> tuple[list, list]:
    """Transparent PNGs of the preprocessed frames (white where 1) and of the inputs (blue +1, orange −1)."""
    from PIL import Image

    pre_images, input_images = [], []
    for p, x in zip(pre, inputs):
        rgba = np.zeros((*p.shape, 4), dtype=np.uint8)
        rgba[p > 0] = (255, 255, 255, 255)
        pre_images.append(Image.fromarray(rgba, "RGBA"))
        rgba = np.zeros((*x.shape, 4), dtype=np.uint8)
        rgba[x > 0] = (57, 135, 229, 255)
        rgba[x < 0] = (235, 104, 52, 255)
        input_images.append(Image.fromarray(rgba, "RGBA"))
    return pre_images, input_images


def points(rewards: Sequence[float]) -> list[dict[str, Any]]:
    """Pong: one entry per point scored, from the rewards: who scored, the step at which the point ended (1-based,
    the state after it is frame `step`), how many steps the rally lasted and the score afterwards."""
    out, last_end, agent, opponent = [], 0, 0, 0
    for t, r in enumerate(rewards):
        if r == 0:
            continue
        agent, opponent = agent + (r > 0), opponent + (r < 0)
        out.append({"n": len(out) + 1, "scorer": "agent" if r > 0 else "opponent", "step": t + 1, "rally": t + 1 - last_end,
                    "score": [int(agent), int(opponent)]})
        last_end = t + 1
    return out


def viewer_data(rec: Recording, max_steps: int = 0) -> dict[str, Any]:
    """Everything rl/pg/watch.html shows, JSON-serialisable (frames as PNG data URIs). `max_steps` > 0 shows only
    the first steps of a longer recording."""
    T = rec.steps if not max_steps else min(rec.steps, int(max_steps))
    labels, short = ACTIONS[rec.env]
    returns = rec.returns()
    data: dict[str, Any] = {
        "env": rec.env, "env_id": rec.env_id, "env_label": ENV_LABELS[rec.env], "note": NOTES[rec.env],
        "policy": rec.policy_label, "seed": rec.seed, "gamma": rec.gamma, "greedy": rec.greedy, "fps": FPS[rec.env],
        "seconds_per_step": SECONDS_PER_STEP[rec.env],
        "steps": T, "episode_steps": rec.steps, "total_reward": rec.total_reward,
        "cut": bool(T < rec.steps or rec.truncated),
        "action_labels": list(labels), "action_short": list(short),
        "actions": rec.actions[:T].astype(int).tolist(), "env_actions": rec.env_actions[:T].astype(int).tolist(),
        "probs": np.round(rec.probs[:T].astype(float), 4).tolist(), "rewards": rec.rewards[:T].astype(float).tolist(),
        "returns": np.round(returns[:T], 4).tolist(),
        "frames": [_png_uri(im) for im in _frame_images(rec.frames[:T + 1])],
        "frame_size": [int(rec.frames.shape[2]), int(rec.frames.shape[1])],
    }
    if rec.env == "pong":
        pre_images, input_images = _overlay_images(rec.pre[:T + 1], rec.inputs[:T + 1])
        data.update({
            "pre": [_png_uri(im) for im in pre_images], "inputs": [_png_uri(im) for im in input_images],
            "input_size": list(PONG_FRAME), "crop": list(CROP),
            "input_counts": [[int((x > 0).sum()), int((x < 0).sum())] for x in rec.inputs[:T + 1]],
            "score": rec.score()[:T + 1].tolist(), "pong_actions": list(PONG_ACTIONS),
            "points": points(rec.rewards[:T].tolist()),
        })
    else:
        data.update({
            "observations": np.round(rec.observations[:T + 1].astype(float), 4).tolist(),
            "obs_labels": ["cart position x", "cart velocity ẋ", "pole angle θ", "pole angular velocity θ̇"],
            "obs_limits": [None if lim is None else float(lim) for lim in CARTPOLE_LIMITS],
        })
    return data


def render_html(data: dict[str, Any], theme: str | None = None) -> str:
    """The viewer page with `data` embedded; `theme` = "light" | "dark" pins the colour scheme (the dashboard passes
    its own), None follows the viewer's system setting."""
    payload = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    attr = f' data-theme="{theme}"' if theme in ("light", "dark") else ""
    return TEMPLATE.read_text(encoding="utf-8").replace("__THEME_ATTR__", attr).replace("__DATA__", payload)


def write_viewer(path: Path, rec: Recording, max_steps: int = 0, theme: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(viewer_data(rec, max_steps), theme=theme), encoding="utf-8")
    return path


# --- the report's figure ----------------------------------------------------------------------------

def _step_label(rec: Recording, t: int, returns: np.ndarray) -> str:
    short = ACTIONS[rec.env][1]
    p = rec.probs[t]
    probs = f"π = ({p[0]:.2f}, {p[1]:.2f})" if len(p) == 2 else "π = (" + ", ".join(f"{x:.2f}" for x in p) + ")"
    return f"t = {t} · a = {short[rec.actions[t]]}\n{probs}\nr = {rec.rewards[t]:g} · G = {returns[t]:.2f}"


def _frame_axes(ax, plotting) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(plotting.AXIS)
        spine.set_linewidth(0.6)


def first_steps_figure(rec: Recording, path: Path, title: str, steps: int = 10) -> Path:
    """The first `steps` steps as a grid of two rows: for Pong each step shows the frame (crop marked), the
    preprocessed frame and the network input; for CartPole the frame with the state under it."""
    import textwrap

    import matplotlib.pyplot as plt

    from rl import plotting

    plotting.apply_style()
    n = min(steps, rec.steps)
    returns = rec.returns()
    cols = max(1, (n + 1) // 2)
    is_pong = rec.env == "pong"
    sub = (f"{rec.env_id} · {rec.policy_label} · seed {rec.seed} · the episode lasted {rec.steps} steps, total reward "
           f"{rec.total_reward:g}. a: the action taken, π: the policy's probabilities of ({ACTIONS[rec.env][1][0]}, "
           f"{ACTIONS[rec.env][1][1]}), r: the reward, G: the reward-to-go Σ γ^k r (γ = {rec.gamma:g}).")
    if is_pong:
        sub += (" Rows per step: the frame (dashed: the rows preprocess keeps), preprocess(frame) (white = 1), and the "
                "network input preprocess(frame_t) − preprocess(frame_t−1) (blue +1, orange −1).")
    lines = textwrap.wrap(sub, 150)
    if is_pong:
        fig = plt.figure(figsize=(7.5, 10.6))
        head = 0.02 + 0.012 * len(lines)  # the title and the subtitle lines, as a fraction of the height
        outer = fig.add_gridspec(2, cols, hspace=0.30, wspace=0.12, top=1 - head - 0.035, bottom=0.025, left=0.02, right=0.98)
    else:
        fig = plt.figure(figsize=(7.5, 4.4))
        head = 0.045 + 0.035 * len(lines)
        outer = fig.add_gridspec(2, cols, hspace=0.78, wspace=0.08, top=1 - head - 0.085, bottom=0.12, left=0.02, right=0.98)
    for i in range(n):
        row, col = divmod(i, cols)
        label = _step_label(rec, i, returns)
        if is_pong:
            inner = outer[row, col].subgridspec(3, 1, height_ratios=[210 / 160, 1, 1], hspace=0.08)
            ax = fig.add_subplot(inner[0])
            ax.imshow(rec.frames[i], interpolation="nearest")
            ax.add_patch(plt.Rectangle((-0.5, CROP[0] - 0.5), rec.frames.shape[2], CROP[1] - CROP[0], fill=False,
                                       linestyle="--", linewidth=0.8, edgecolor=plotting.SERIES[3]))
            ax.set_title(label, fontsize=6.5, pad=3, linespacing=1.15)
            ax_pre = fig.add_subplot(inner[1])
            ax_pre.imshow(rec.pre[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax_in = fig.add_subplot(inner[2])
            rgb = np.full((*PONG_FRAME, 3), 26, dtype=np.uint8)
            rgb[rec.inputs[i] > 0] = (57, 135, 229)
            rgb[rec.inputs[i] < 0] = (235, 104, 52)
            ax_in.imshow(rgb, interpolation="nearest")
            n_pos, n_neg = int((rec.inputs[i] > 0).sum()), int((rec.inputs[i] < 0).sum())
            ax_in.set_xlabel(f"+1: {n_pos} px · −1: {n_neg} px", fontsize=6, labelpad=2)
            for a in (ax, ax_pre, ax_in):
                _frame_axes(a, plotting)
        else:
            ax = fig.add_subplot(outer[row, col])
            ax.imshow(rec.frames[i], interpolation="bilinear")
            ax.set_title(label, fontsize=6.5, pad=3, linespacing=1.15)
            x, xd, th, thd = rec.observations[i]
            ax.set_xlabel(f"x = {x:.3f}  ẋ = {xd:.2f}\nθ = {np.degrees(th):.1f}°  θ̇ = {thd:.2f}", fontsize=6.5, labelpad=2)
            _frame_axes(ax, plotting)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, fontweight="semibold", color=plotting.INK, y=0.99 if is_pong else 0.975)
    fig.text(0.02, 0.972 if is_pong else 0.925, "\n".join(lines), fontsize=7, color=plotting.INK_SECONDARY, va="top", linespacing=1.3)
    return plotting.save(fig, path)


# --- `python run.py watch` --------------------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", choices=list(ENV_IDS), required=True)
    parser.add_argument("--run", help="results/pg folder (or its name, e.g. part1_pong) whose policy.pt plays; default: an untrained policy")
    parser.add_argument("--episode", help="show this saved recording (episode.npz) instead of playing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0, help="stop after this many steps (0 = the whole episode)")
    parser.add_argument("--greedy", action="store_true", help="take the most probable action instead of sampling")
    parser.add_argument("--out", help="the viewer page to write (default: results/pg/watch_<env>.html)")
    parser.add_argument("--save", help="also save the recording as this .npz file")
    parser.add_argument("--figure", help="also draw the first-ten-steps figure to this PNG")
    parser.add_argument("--theme", choices=["light", "dark"], help="pin the page's colour scheme")


def main(args: argparse.Namespace) -> int:
    if args.episode:
        rec = load_recording(Path(args.episode))
        if rec.env != args.env:
            raise SystemExit(f"{args.episode} is a {rec.env} episode, not {args.env}")
    else:
        rec = record(args.env, args.run, seed=args.seed, max_steps=args.steps, greedy=args.greedy)
    print(f"{rec.env_id} · {rec.policy_label} · seed {rec.seed}: {rec.steps} steps, total reward {rec.total_reward:g}"
          + (" (cut off by --steps)" if rec.truncated else ""))
    if args.save:
        print(f"   saved {save_recording(rec, Path(args.save))}")
    if args.figure:
        print(f"   saved {first_steps_figure(rec, Path(args.figure), f'{ENV_LABELS[rec.env]}: the first 10 steps of one episode')}")
    out = Path(args.out) if args.out else RESULTS / "pg" / f"watch_{args.env}.html"
    print(f"   wrote {write_viewer(out, rec, theme=args.theme)}  (open it in a browser)")
    return 0
