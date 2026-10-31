"""`python run.py pg --part {1,2} --env {cartpole,pong}`: train, evaluate and plot one run.

Outputs go to results/pg/part<P>_<env>[_<tag>]/: config.json, progress.jsonl, train.json,
policy.pt, eval.json, training_curve.png and eval_histogram.png.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
from dataclasses import asdict

from rl.outputs import ENV_LABELS, ROOT, ProgressLogger, pg_dir, write_json
from rl.pg.envs import ENV_IDS, GAMMA, SECONDS_PER_STEP

DEFAULT_EPISODES = {"cartpole": 1000, "pong": 2000}
# The episode rewards each environment makes possible: CartPole-v1 pays 1 per step up to its 500-step cap; a Pong-v5
# game ends when one side reaches 21 points, so the reward is (agent's points − opponent's points) in [−21, 21].
REWARD_RANGE = {"cartpole": (0, 500), "pong": (-21, 21)}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--part", type=int, choices=(1, 2), required=True, help="1 = no baseline, 2 = with baseline")
    parser.add_argument("--env", choices=list(ENV_IDS), required=True)
    parser.add_argument("--episodes", type=int, help="training episodes (default: cartpole 1000, pong 2000)")
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate (handout: 0.01)")
    parser.add_argument("--gamma", type=float, help="discount factor (handout: cartpole 0.95, pong 0.99)")
    parser.add_argument("--baseline", help="Part 2 baseline name (default: DEFAULT_BASELINE in rl/pg/baselines.py)")
    parser.add_argument("--eval-episodes", type=int, default=500, help="episodes rolled out after training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto",
                        help="auto (cuda when available, else cpu), cpu, cuda, cuda:N or mps (Apple GPU); cuda/mps fail if unavailable")
    parser.add_argument("--tag", help="suffix for the results folder, to keep several runs side by side")
    parser.add_argument("--eval-only", action="store_true", help="skip training; evaluate the saved policy.pt")
    parser.add_argument("--replot", action="store_true",
                        help="redraw training_curve.png and eval_histogram.png from the saved train.json and eval.json, nothing else")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra setting for cfg.extra (repeatable)")


def _parse_extra(pairs: list[str]) -> dict:
    from rl.params import parse_overrides
    from rl.pg.options import OPTION_NAMES

    extra = parse_overrides(pairs)
    unknown = sorted(set(extra) - set(OPTION_NAMES))
    if unknown:
        raise SystemExit(f"--set: unknown option(s) {unknown}; choose from {list(OPTION_NAMES)} (rl/pg/options.py)")
    return extra


def main(args: argparse.Namespace) -> int:
    import torch

    from rl.pg.baselines import DEFAULT_BASELINE, make_baseline
    from rl.pg.envs import make_env
    from rl.pg.policy import build_policy
    from rl.pg.reinforce import TrainConfig, evaluate_episodes, train
    from rl.plotting import plot_training_curve

    baseline = "none" if args.part == 1 else (args.baseline or DEFAULT_BASELINE)
    if baseline is None:
        print("TODO: choose a baseline for Part 2: pass --baseline <name> or set DEFAULT_BASELINE in rl/pg/baselines.py")
        return 1
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available (pass --device cpu to run on the CPU, or run on deepdish)")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Apple GPU) is not available on this machine")
    if device.startswith("cuda"):
        index = torch.device(device).index or 0
        print(f"GPU: {torch.cuda.get_device_name(index)} ({device}); CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    elif args.device == "auto":
        print("note: no CUDA device visible, running on the CPU")
    if device.startswith("cuda") and torch.cuda.device_count() > 1:
        print(
            f"warning: {torch.cuda.device_count()} GPUs are visible, but deepdish jobs must use a single card. "
            "Launch through the dashboard / scripts/jobs.sh, or set CUDA_VISIBLE_DEVICES=<index>."
        )

    cfg = TrainConfig(
        env=args.env,
        part=args.part,
        gamma=args.gamma if args.gamma is not None else GAMMA[args.env],
        lr=args.lr,
        episodes=args.episodes or DEFAULT_EPISODES[args.env],
        baseline=baseline,
        seed=args.seed,
        device=device,
        eval_episodes=args.eval_episodes,
        extra=_parse_extra(args.set),
    )
    run_dir = pg_dir(args.part, args.env, args.tag)
    label = f"Part {args.part}, {ENV_LABELS[args.env]}" + (f" (baseline: {baseline})" if args.part == 2 else "")
    subtitle = f"γ = {cfg.gamma}, Adam lr = {cfg.lr:g} · overlaid: the moving average of the rewards over 100 episodes"
    if args.replot:
        return replot(cfg, run_dir, label, subtitle)
    print(f"== {label} -> {run_dir.relative_to(ROOT)} on {device}")
    write_json(
        run_dir / "config.json",
        {**asdict(cfg), "host": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
    )

    torch.manual_seed(cfg.seed)
    env = make_env(cfg.env)
    try:
        policy = build_policy(cfg, env).to(device)
        if args.eval_only:
            policy.load_state_dict(torch.load(run_dir / "policy.pt", map_location=device))
        else:
            rewards = train(cfg, env, policy, make_baseline(cfg.baseline, cfg, env), ProgressLogger(run_dir))
            write_json(run_dir / "train.json", {"episode_rewards": rewards})
            torch.save(policy.state_dict(), run_dir / "policy.pt")
            path = plot_training_curve(rewards, run_dir / "training_curve.png",
                                       f"{label}: episode reward against the number of training episodes", subtitle)
            print(f"   saved {path.relative_to(ROOT)}")

        stats = evaluate_episodes(cfg, env, policy, cfg.eval_episodes)
        eval_rewards = [s.reward for s in stats]
        path, mean, std = save_eval(cfg, run_dir, label, eval_rewards, [s.length for s in stats])
        print(f"   saved {path.relative_to(ROOT)}  (mean {mean:.2f}, std {std:.2f} over {len(eval_rewards)} episodes)")
        record_game(cfg, policy, run_dir, label)
    except NotImplementedError as e:
        print(f"TODO: {e}")
        return 1
    finally:
        env.close()
    return 0


def game_time(env: str, steps: float) -> str:
    """`steps` of `env` as game time, e.g. "51 s" or "2 min 10 s"."""
    seconds = float(steps) * SECONDS_PER_STEP[env]
    if seconds < 100:
        return f"{seconds:.0f} s" if seconds >= 10 else f"{seconds:.1f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes} min {rest} s"


def length_summary(env: str, lengths) -> tuple[dict, str]:
    """(eval.json fields, one line of text) about how long the evaluation episodes lasted."""
    import numpy as np

    n = np.asarray(lengths, dtype=float)
    fields = {"lengths": [int(x) for x in lengths], "mean_length": float(n.mean()), "std_length": float(n.std()),
              "min_length": int(n.min()), "max_length": int(n.max()), "seconds_per_step": SECONDS_PER_STEP[env]}
    same = n.min() == n.max()
    text = (f"every episode lasted {int(n[0])} steps ({game_time(env, n[0])} of game time)" if same else
            f"length: mean {n.mean():.0f} steps ({game_time(env, n.mean())} of game time), min {int(n.min())}, max {int(n.max())}")
    return fields, text


def eval_note(env: str, rewards, lengths=None) -> str | None:
    """A sentence for the histogram when the evaluation rewards need explaining (all the same, at a bound)."""
    import numpy as np

    r = np.asarray(rewards, dtype=float)
    if len(r) == 0 or r.min() < r.max():
        return None
    value, lo, hi = float(r[0]), *REWARD_RANGE[env]
    lasted = ""
    if lengths is not None and len(lengths):
        n = np.asarray(lengths, dtype=float)
        lasted = (f" Each game lasted {int(n[0])} steps, {game_time(env, n[0])} of game time." if n.min() == n.max() else
                  f" The games lasted {n.mean():.0f} steps on average ({game_time(env, n.mean())} of game time).")
    if env == "cartpole" and value == hi:
        return (f"Every one of the {len(r)} episodes reached the 500-step cap of CartPole-v1 ({game_time(env, hi)} of game "
                "time), the highest reward possible: the pole never fell and the cart never left the track, so the "
                "distribution is one bar at 500 and the standard deviation is 0.")
    if env == "pong" and value == lo:
        return (f"Every one of the {len(r)} games was lost 0 : 21, the lowest reward possible, and all games are identical. "
                "That happens when the policy takes the same action in every state: Pong-v5's sticky actions then change "
                "nothing, so the game is deterministic and the standard deviation is 0." + lasted)
    return f"All {len(r)} episodes scored {value:g}, so the standard deviation is 0." + lasted


def save_eval(cfg, run_dir, label: str, eval_rewards, lengths=None):
    """Draw eval_histogram.png and write eval.json (the statistics, the episode lengths, the note and every reward).
    Returns (path, mean, std)."""
    from rl.plotting import plot_reward_histogram, reward_stats

    note = eval_note(cfg.env, eval_rewards, lengths)
    length_fields, length_text = length_summary(cfg.env, lengths) if lengths else ({}, None)
    path, mean, std = plot_reward_histogram(
        eval_rewards, run_dir / "eval_histogram.png", f"{label}: histogram of the {len(eval_rewards)} episode rewards",
        value_range=REWARD_RANGE[cfg.env], integer=True, note=note, length_text=length_text,
    )
    write_json(run_dir / "eval.json", {**reward_stats(eval_rewards), **length_fields, "note": note,
                                       "rewards": list(map(float, eval_rewards))})
    return path, mean, std


def replot(cfg, run_dir, label: str, subtitle: str) -> int:
    """`--replot`: redraw the two figures of a finished run from its saved numbers."""
    from rl.outputs import read_json
    from rl.plotting import plot_training_curve

    train = read_json(run_dir / "train.json")
    evaluation = read_json(run_dir / "eval.json")
    if not train or not evaluation:
        print(f"nothing to replot: {run_dir.relative_to(ROOT)} has no train.json / eval.json yet")
        return 1
    path = plot_training_curve(train["episode_rewards"], run_dir / "training_curve.png",
                               f"{label}: episode reward against the number of training episodes", subtitle)
    print(f"   saved {path.relative_to(ROOT)}")
    path, mean, std = save_eval(cfg, run_dir, label, evaluation["rewards"], evaluation.get("lengths"))
    print(f"   saved {path.relative_to(ROOT)}  (mean {mean:.2f}, std {std:.2f} over {len(evaluation['rewards'])} episodes)")
    return 0


def record_game(cfg, policy, run_dir, label: str) -> None:
    """One more episode with the trained policy, frame by frame, for the dashboard's "Watch the game" panel
    (episode.npz) and the report's first-ten-steps figure (first_steps.png). Rendering needs pygame for CartPole;
    if it is missing, the run still counts and the message says what to install."""
    from rl.pg.options import options_for
    from rl.pg.watch import first_steps_figure, record_episode, render_env, save_recording

    try:
        game_env = render_env(cfg.env)
    except Exception as e:  # noqa: BLE001 - e.g. gymnasium's DependencyNotInstalled for pygame
        print(f"   skipped the recorded episode: {type(e).__name__}: {e}  (pip install -r requirements.txt)")
        return
    try:
        rec = record_episode(cfg, game_env, policy, seed=cfg.seed + 20_000, max_steps=options_for(cfg).max_steps,
                             greedy=options_for(cfg).eval_greedy,
                             policy_label=f"Trained policy of {run_dir.relative_to(ROOT)}")
    finally:
        game_env.close()
    path = save_recording(rec, run_dir / "episode.npz")
    print(f"   saved {path.relative_to(ROOT)}  ({rec.steps} steps, reward {rec.total_reward:g}; watch it in the dashboard)")
    path = first_steps_figure(rec, run_dir / "first_steps.png", f"{label}: the first 10 steps of one episode")
    print(f"   saved {path.relative_to(ROOT)}")
