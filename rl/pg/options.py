"""Tunables of the policy-gradient code, all in one place.

Every field can be changed from the command line with `--set KEY=VALUE` (it lands in `cfg.extra`), e.g.

    python run.py pg --part 1 --env cartpole --set hidden=[128,128] --set normalize=true
    python run.py pg --part 2 --env pong --baseline value --set batch_episodes=4 --set baseline_lr=0.001

`options_for(cfg)` applies the per-environment defaults, then `cfg.extra`; unknown keys raise ValueError.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from rl.params import apply_overrides


@dataclass(frozen=True)
class Options:
    # policy network (rl/pg/policy.py)
    hidden: tuple = (64,)  # hidden layer sizes of the policy network (Pong default: (200,))
    arch: str = "mlp"  # "mlp" (flattened input), or "cnn" for Pong (two convolutions on the 80x80 frame)
    frame_diff: bool = True  # Pong: feed current frame minus previous frame, so the ball's motion is visible
    # policy-gradient estimator (rl/pg/reinforce.py)
    reward_to_go: bool = True  # weight log pi(a_t|s_t) by G_t (True) or by the whole-episode return G_0 (False)
    discount_grad: bool = False  # also weight step t by gamma^t: the exact gradient of the discounted objective
    normalize: bool = False  # standardise the advantages within each update (mean 0, standard deviation 1)
    batch_episodes: int = 1  # episodes collected per Adam step
    entropy_coef: float = 0.0  # entropy bonus beta: the objective gains beta * H(pi(.|s_t)) at every step, so the policy
    #                            keeps exploring instead of collapsing onto one action (Pong default 0.01, see DEFAULTS_BY_ENV)
    max_grad_norm: float = 0.0  # clip the gradient norm to this when > 0
    max_steps: int = 1000  # an episode ends when the environment says so or after this many steps (0 = no cap);
    #                       CartPole-v1 truncates itself at 500. Pong's default is 10000 (DEFAULTS_BY_ENV): a game
    #                       lasts 1000-3000 steps, so a 1000-step cap would turn the score into a partial score.
    # baseline, Part 2 (rl/pg/baselines.py)
    baseline_lr: float = 0.01  # Adam learning rate of the value baseline's network
    baseline_hidden: tuple = ()  # hidden sizes of the value network; () = same as the policy's
    baseline_epochs: int = 1  # gradient steps on the value network per policy update
    baseline_decay: float = 0.9  # exponential-moving-average decay of the constant ("mean") baseline
    # custom CUDA kernels (rl/cuda/pg.py)
    kernels: bool = False  # compute discounted returns with the scan kernel and Pong frames with the preprocess kernel
    # CPU runs
    cpu_threads: int = 2  # torch intra-op threads when training on the CPU (tiny networks only lose time to more threads); 0 = torch's default
    # evaluation and logging
    eval_greedy: bool = False  # evaluate with argmax actions instead of sampling from the policy
    log_every: int = 10  # print a progress line every this many episodes

    def describe(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


DEFAULTS_BY_ENV: dict[str, dict[str, Any]] = {
    "cartpole": {},
    # Pong: with the handout's learning rate 0.01 and no bonus the policy collapsed onto one action within ~150 episodes
    # (every game lost 0 : 21); the bonus, with the smaller learning rate in rl/pg/run.py DEFAULT_LR, prevents that.
    "pong": {"hidden": (200,), "max_steps": 10_000, "entropy_coef": 0.01},
}


def options_for(cfg: Any) -> Options:
    """The Options for a TrainConfig: environment defaults, then anything given with --set."""
    opts = apply_overrides(Options(), DEFAULTS_BY_ENV.get(getattr(cfg, "env", ""), {}))
    extra = dict(getattr(cfg, "extra", None) or {})
    return apply_overrides(opts, extra)


OPTION_NAMES = tuple(f.name for f in fields(Options))
