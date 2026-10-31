"""Basic policy gradient (REINFORCE): training and evaluation loops.

Objective: the expected discounted reward of an episode, J = E[sum_t gamma^t r_t]. Its gradient is estimated
from sampled episodes with the score function:

    grad J ~ sum_t grad log pi(a_t | s_t) * (G_t - b_t),     G_t = sum_{k>=t} gamma^(k-t) r_k,

so one Adam step minimises the surrogate loss -sum_t log pi(a_t|s_t) * (G_t - b_t) (averaged over the episodes
of a batch). b_t is the baseline (zero in Part 1; a baseline from rl/pg/baselines.py in Part 2). Weighting
step t by the reward-to-go G_t instead of the whole-episode return keeps the estimator unbiased and lowers its
variance; `--set reward_to_go=false` uses G_0 for every step, and `--set discount_grad=true` adds the gamma^t
factor of the exact gradient of the discounted objective. With an entropy bonus beta (`entropy_coef`, Pong's
default 0.01) the objective gains beta * H(pi(.|s_t)) at every step, which keeps the policy from collapsing onto
one action early. See rl/pg/options.py for every option.

Every training episode is logged with its reward, length, the policy's mean entropy over its steps and the
fraction of "right" actions (rl/pg/envs.py RIGHT_INDEX), so a collapse onto one action is visible in the log,
the dashboard's live chart and the training-curve figure.

Formal counterpart: lean/RLVerify/Returns.lean proves that the backward recursion used by `discounted_returns`
equals the definition of G_t; lean/RLVerify/Baseline.lean proves that subtracting a baseline that does not
depend on the action leaves the expected gradient unchanged.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from rl.outputs import ProgressLogger
from rl.pg.envs import make_adapter, right_index
from rl.pg.options import Options, options_for


@dataclass
class TrainConfig:
    env: str  # "cartpole" or "pong"
    part: int  # 1 (no baseline) or 2 (with baseline)
    gamma: float
    lr: float = 0.01  # Adam learning rate from the handout; note any change in the report
    episodes: int = 1000
    baseline: str = "none"
    seed: int = 0
    device: str = "cpu"
    eval_episodes: int = 500
    extra: dict[str, Any] = field(default_factory=dict)  # anything passed with --set key=value


@dataclass
class Episode:
    """One trajectory: observations (T, *obs_shape), policy action indices (T,) and rewards (T,)."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    entropy: float = float("nan")  # mean entropy of pi(.|s_t) over the episode's steps (nats), when recorded

    @property
    def length(self) -> int:
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum())


def check_device(policy: torch.nn.Module, baseline: Any, device: torch.device) -> None:
    """Fail loudly if the policy (or the baseline's network) is not on the device the run is configured for.

    Every tensor the policy interacts with is created on `device` (observations, actions, returns, baselines), so
    a module left on another device would raise a device-mismatch error at the first forward pass; this check
    gives a clearer message first.
    """
    for name, module in (("policy", policy), ("baseline", getattr(baseline, "net", None))):
        if module is None:
            continue
        devices = {p.device.type for p in module.parameters()}
        if devices and devices != {device.type}:
            raise RuntimeError(f"the {name} network is on {sorted(devices)} but the run is configured for {device}")


def limit_cpu_threads(device: torch.device, opts: Options) -> None:
    """On the CPU, PyTorch's default of one thread per core makes the tiny per-step forward passes of this code
    slower, not faster (synchronisation dominates); `cpu_threads` caps them."""
    if device.type == "cpu" and opts.cpu_threads > 0:
        torch.set_num_threads(int(opts.cpu_threads))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discounted_returns(rewards: Sequence[float], gamma: float) -> np.ndarray:
    """G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ... for every t, by the backward recursion
    G_t = r_t + gamma G_{t+1} (lean/RLVerify/Returns.lean, `returnsBackward_get`)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    returns = np.zeros(len(rewards))
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def run_episode(
    env: Any, adapter: Any, policy: torch.nn.Module, device: torch.device, greedy: bool = False,
    max_steps: int = 0, seed: int | None = None,
) -> Episode:
    """Roll out one episode, sampling actions from the policy (or taking the argmax when `greedy`).

    The episode ends when the environment terminates or truncates it, or after `max_steps` steps when that is
    positive (the upper limit on the trajectory length). Observations are moved to `device` before the policy
    sees them, so the forward passes run there.
    """
    obs = adapter.reset(env, seed)
    observations, actions, rewards = [], [], []
    entropy_sum = 0.0
    with torch.no_grad():
        while True:
            dist = Categorical(logits=policy(torch.as_tensor(obs, device=device).unsqueeze(0)))
            action = int(dist.probs.argmax(dim=-1).item() if greedy else dist.sample().item())
            entropy_sum += float(dist.entropy().item())
            next_obs, reward, done, _ = adapter.step(env, action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
            if done or (max_steps and len(rewards) >= max_steps):
                break
    return Episode(np.stack(observations), np.asarray(actions, dtype=np.int64), np.asarray(rewards, dtype=np.float32),
                   entropy=entropy_sum / max(1, len(rewards)))


def returns_for(rewards: Sequence[float], gamma: float, opts: Options) -> np.ndarray:
    """discounted_returns, or the CUDA scan kernel (rl/cuda/pg.py) when `kernels` is on."""
    if opts.kernels:
        from rl.cuda.pg import discounted_returns_gpu

        return discounted_returns_gpu(rewards, gamma)
    return discounted_returns(rewards, gamma)


def step_weights(episode: Episode, gamma: float, opts: Options) -> np.ndarray:
    """The return each step's log-probability is weighted by before the baseline is subtracted."""
    G = returns_for(episode.rewards, gamma, opts)
    weights = G if opts.reward_to_go else np.full_like(G, G[0] if len(G) else 0.0)
    if opts.discount_grad:
        weights = weights * gamma ** np.arange(len(G))
    return weights


def policy_gradient_loss(
    policy: torch.nn.Module, episodes: Sequence[Episode], gamma: float, baseline: Any, opts: Options, device: torch.device
) -> tuple[torch.Tensor, dict[str, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """The surrogate loss for a batch of episodes.

    Returns (loss, statistics, (observations, returns, timesteps)); the last tuple is what the baseline is
    trained on afterwards (targets are the reward-to-go returns G_t).
    """
    log_probs, advantages, entropies, obs_all, returns_all, t_all = [], [], [], [], [], []
    for ep in episodes:
        if ep.length == 0:
            continue
        obs = torch.as_tensor(ep.observations, device=device)
        acts = torch.as_tensor(ep.actions, device=device)
        t = torch.arange(ep.length, device=device)
        G = torch.as_tensor(returns_for(ep.rewards, gamma, opts), dtype=torch.float32, device=device)
        w = torch.as_tensor(step_weights(ep, gamma, opts), dtype=torch.float32, device=device)
        b = baseline.predict(obs, t).detach().to(w) if baseline is not None else torch.zeros_like(w)
        dist = Categorical(logits=policy(obs))
        log_probs.append(dist.log_prob(acts))
        entropies.append(dist.entropy())
        advantages.append(w - b)
        obs_all.append(obs)
        returns_all.append(G)
        t_all.append(t)
    if not log_probs:
        raise ValueError("policy_gradient_loss needs at least one non-empty episode")
    logp, adv, ent = torch.cat(log_probs), torch.cat(advantages), torch.cat(entropies)
    raw_adv = adv
    if opts.normalize and adv.numel() > 1:
        std = adv.std(unbiased=False)
        adv = (adv - adv.mean()) / (std if std > 1e-8 else 1.0)
    loss = -(logp * adv).sum() / len(episodes)
    if opts.entropy_coef:  # the bonus is per step, like the policy-gradient term: -beta * sum_t H(pi(.|s_t)) / episodes
        loss = loss - opts.entropy_coef * ent.sum() / len(episodes)
    stats = {
        "loss": float(loss.item()),
        "advantage_mean": float(raw_adv.mean().item()),
        "advantage_std": float(raw_adv.std(unbiased=False).item()) if raw_adv.numel() > 1 else 0.0,
        "batch_entropy": float(ent.mean().item()),
    }
    return loss, stats, (torch.cat(obs_all), torch.cat(returns_all), torch.cat(t_all))


def update(
    policy: torch.nn.Module, optimizer: torch.optim.Optimizer, episodes: Sequence[Episode], gamma: float,
    baseline: Any, opts: Options, device: torch.device,
) -> dict[str, float]:
    """One Adam step on the policy from a batch of episodes, then one baseline update. Returns statistics."""
    optimizer.zero_grad()
    loss, stats, (obs, returns, t) = policy_gradient_loss(policy, episodes, gamma, baseline, opts, device)
    loss.backward()
    if opts.max_grad_norm and opts.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), opts.max_grad_norm)
    stats["grad_norm"] = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in policy.parameters() if p.grad is not None)).item())
    optimizer.step()
    if baseline is not None:
        stats.update(baseline.update(obs, returns, t))
    return stats


def train(cfg: TrainConfig, env: Any, policy: torch.nn.Module, baseline: Any, logger: ProgressLogger) -> list[float]:
    """Train `policy` with REINFORCE and return the reward of every training episode, in order.

    Calls logger.log(episode, reward, ...) once per episode so progress shows up in the job log and in the
    dashboard's live training curve.
    """
    opts = options_for(cfg)
    device = torch.device(cfg.device)
    policy.to(device)
    adapter = make_adapter(cfg, env, opts)
    check_device(policy, baseline, device)
    limit_cpu_threads(device, opts)
    seed_everything(cfg.seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    if "log_every" in (cfg.extra or {}):  # otherwise keep the logger's own setting
        logger.print_every = max(1, int(opts.log_every))
    rewards: list[float] = []
    batch: list[Episode] = []
    stats: dict[str, float] = {}
    right = right_index(cfg.env, adapter.n_actions)
    for episode in range(1, cfg.episodes + 1):
        ep = run_episode(env, adapter, policy, device, max_steps=opts.max_steps, seed=cfg.seed if episode == 1 else None)
        rewards.append(ep.total_reward)
        batch.append(ep)
        if len(batch) >= opts.batch_episodes or episode == cfg.episodes:
            stats = update(policy, optimizer, batch, cfg.gamma, baseline, opts, device)
            batch = []
        logger.log(episode, ep.total_reward, steps=ep.length, entropy=round(ep.entropy, 4),
                   frac_right=round(float(np.mean(ep.actions == right)), 4) if ep.length else 0.0, **stats)
    return rewards


class EpisodeStats(NamedTuple):
    """What an evaluation episode is summarised by: its total reward and its length in steps."""

    reward: float
    length: int


def evaluate_episodes(cfg: TrainConfig, env: Any, policy: torch.nn.Module, episodes: int) -> list[EpisodeStats]:
    """Roll out `episodes` episodes with the trained policy and return each one's total reward and length."""
    opts = options_for(cfg)
    device = torch.device(cfg.device)
    policy.to(device)
    adapter = make_adapter(cfg, env, opts)
    check_device(policy, None, device)
    limit_cpu_threads(device, opts)
    seed_everything(cfg.seed + 10_000)
    results = []
    for i in range(int(episodes)):
        ep = run_episode(env, adapter, policy, device, greedy=opts.eval_greedy, max_steps=opts.max_steps,
                         seed=cfg.seed + 10_000 if i == 0 else None)
        results.append(EpisodeStats(ep.total_reward, ep.length))
    return results


def evaluate(cfg: TrainConfig, env: Any, policy: torch.nn.Module, episodes: int) -> list[float]:
    """Roll out `episodes` episodes with the trained policy and return their total rewards."""
    return [stats.reward for stats in evaluate_episodes(cfg, env, policy, episodes)]
