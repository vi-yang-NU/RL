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
from rl.pg.envs import ENV_IDS, GAMMA

DEFAULT_EPISODES = {"cartpole": 1000, "pong": 2000}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--part", type=int, choices=(1, 2), required=True, help="1 = no baseline, 2 = with baseline")
    parser.add_argument("--env", choices=list(ENV_IDS), required=True)
    parser.add_argument("--episodes", type=int, help="training episodes (default: cartpole 1000, pong 2000)")
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate (handout: 0.01)")
    parser.add_argument("--gamma", type=float, help="discount factor (handout: cartpole 0.95, pong 0.99)")
    parser.add_argument("--baseline", help="Part 2 baseline name (default: DEFAULT_BASELINE in rl/pg/baselines.py)")
    parser.add_argument("--eval-episodes", type=int, default=500, help="episodes rolled out after training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda (auto picks cuda when available)")
    parser.add_argument("--tag", help="suffix for the results folder, to keep several runs side by side")
    parser.add_argument("--eval-only", action="store_true", help="skip training; evaluate the saved policy.pt")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra setting for cfg.extra (repeatable)")


def _parse_extra(pairs: list[str]) -> dict:
    extra = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        try:
            extra[key] = json.loads(value)
        except json.JSONDecodeError:
            extra[key] = value
    return extra


def main(args: argparse.Namespace) -> int:
    import torch

    from rl.pg.baselines import DEFAULT_BASELINE, make_baseline
    from rl.pg.envs import make_env
    from rl.pg.policy import build_policy
    from rl.pg.reinforce import TrainConfig, evaluate, train
    from rl.plotting import plot_reward_histogram, plot_training_curve

    baseline = "none" if args.part == 1 else (args.baseline or DEFAULT_BASELINE)
    if baseline is None:
        print("TODO: choose a baseline for Part 2: pass --baseline <name> or set DEFAULT_BASELINE in rl/pg/baselines.py")
        return 1
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
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
    subtitle = f"γ = {cfg.gamma}, Adam lr = {cfg.lr:g}"
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
            path = plot_training_curve(rewards, run_dir / "training_curve.png", f"{label}: training reward", subtitle)
            print(f"   saved {path.relative_to(ROOT)}")

        eval_rewards = evaluate(cfg, env, policy, cfg.eval_episodes)
        path, mean, std = plot_reward_histogram(
            eval_rewards, run_dir / "eval_histogram.png", f"{label}: rewards of the trained policy"
        )
        write_json(run_dir / "eval.json", {"episodes": len(eval_rewards), "mean": mean, "std": std, "rewards": eval_rewards})
        print(f"   saved {path.relative_to(ROOT)}  (mean {mean:.2f}, std {std:.2f} over {len(eval_rewards)} episodes)")
    except NotImplementedError as e:
        print(f"TODO: {e}")
        return 1
    finally:
        env.close()
    return 0
