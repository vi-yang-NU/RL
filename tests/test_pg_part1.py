"""Validation experiments for Policy Gradient Part 1 (REINFORCE without a baseline).

Run with:  python run.py tests --section pg1        (a few minutes on a laptop CPU)
           python run.py tests --section pg1 -k corridor

The long CartPole-v1 and Pong-v5 trainings are `python run.py pg`; the experiments here are small and
deterministic enough to run anywhere: they check the gradient against its closed form, make REINFORCE solve
problems whose optimum we can compute exactly with the Section 2 solvers, isolate the effect of the discount
factor and of reward-to-go, run a short CartPole training, and exercise the Pong preprocessing.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from rl.dp import solvers
from rl.pg.envs import PONG_ACTIONS, make_adapter, make_env
from rl.pg.options import Options
from rl.pg.policy import MLP, build_policy
from rl.pg.reinforce import Episode, discounted_returns, evaluate, policy_gradient_loss, run_episode, step_weights
from rl.pg.toy import BanditEnv, CorridorEnv, corridor_mdp
from rl.plotting import SERIES, moving_average, new_figure
from tests.harness import validation_test
from tests.support import CPU, flat_grad, gradient_samples, mean_and_stderr, pg_config, probabilities, quick_train


class RecordingEnv:
    """Forwards to a gym environment and records the actions it receives."""

    def __init__(self, env: Any):
        self.env = env
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.actions: list = []

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        self.actions.append(int(action))
        return self.env.step(action)

    def close(self):
        self.env.close()


@validation_test(
    section="pg1",
    title="The implemented gradient equals the analytic policy gradient",
    what="For a linear softmax policy, compares the gradient PyTorch computes from the surrogate loss with the "
         "closed form −Σ_t G_t (e_{a_t} − π(·|s_t)) ⊗ s_t, on single-step bandit episodes with random logits and on "
         "multi-step corridor episodes.",
    how="REINFORCE's update is Σ_t ∇log π(a_t|s_t) G_t and the score of a softmax is e_a − π, so the two must agree "
         "to floating-point precision for every action and return; a sign error, a missing return, or a wrong "
         "time alignment would show up immediately.",
)
def analytic_gradient(ctx):
    torch.manual_seed(1)
    rows, worst = [], 0.0
    policy = MLP(1, 3, hidden=())
    with torch.no_grad():
        policy.net[0].weight.copy_(torch.randn(3, 1))
        policy.net[0].bias.copy_(torch.randn(3))
    pi = probabilities(policy, np.ones((1, 1)))[0]
    for action in range(3):
        for G in (-2.0, 0.5, 3.0):
            episode = Episode(np.ones((1, 1), np.float32), np.array([action]), np.array([G], np.float32))
            policy.zero_grad()
            loss, _, _ = policy_gradient_loss(policy, [episode], 0.9, None, Options(), CPU)
            loss.backward()
            analytic = -G * (np.eye(3)[action] - pi)
            diff = float(np.max(np.abs(policy.net[0].bias.grad.numpy() - analytic)))
            worst = max(worst, diff)
            rows.append({"action": action, "return G": G, "autograd ∂L/∂b": np.round(policy.net[0].bias.grad.numpy(), 5).tolist(),
                         "analytic −G(e_a − π)": np.round(analytic, 5).tolist(), "max |diff|": f"{diff:.1e}"})
    ctx.table("bandit: autograd versus closed form", rows)
    ctx.check("single-step gradients match the closed form to 1e-6", worst < 1e-6, f"max difference {worst:.1e}")
    # multi-step: corridor with a linear policy on one-hot states
    L, gamma = 5, 0.8
    env = CorridorEnv(length=L, slip=0.3, max_steps=20)
    policy = MLP(L, 2, hidden=())
    adapter = make_adapter(pg_config("corridor", gamma=gamma), env)
    worst_multi = 0.0
    for seed in range(5):
        episode = run_episode(env, adapter, policy, CPU, seed=seed)
        G = discounted_returns(episode.rewards, gamma)
        probs = probabilities(policy, episode.observations)
        dlogits = -(G[:, None] * (np.eye(2)[episode.actions] - probs))  # (T, 2)
        dW = dlogits.T @ episode.observations  # (2, L)
        db = dlogits.sum(axis=0)
        policy.zero_grad()
        loss, _, _ = policy_gradient_loss(policy, [episode], gamma, None, Options(), CPU)
        loss.backward()
        worst_multi = max(worst_multi, float(np.max(np.abs(policy.net[0].weight.grad.numpy() - dW))), float(np.max(np.abs(policy.net[0].bias.grad.numpy() - db))))
    ctx.metric("max |autograd − analytic| (multi-step)", worst_multi)
    ctx.check("multi-step gradients (returns-to-go, several states) match the closed form to 1e-5", worst_multi < 1e-5, f"{worst_multi:.1e}")


@validation_test(
    section="pg1",
    title="Bandits: the policy concentrates on the better arm",
    what="Trains the policy on one-step problems: a deterministic two-armed bandit with rewards −1 and +1, a "
         "Bernoulli bandit with success probabilities 0.3 and 0.7, and a three-armed bandit with positive rewards "
         "0, 1 and 0.5 (10 seeds).",
    how="Without state or sequence effects, REINFORCE is plain stochastic gradient ascent on E[r], so the "
         "probability of the best arm must approach 1 in the first two problems. The third problem documents a "
         "known weakness that motivates Part 2: with positive-only rewards and no baseline, every sampled arm is "
         "reinforced, so some seeds lock onto the 0.5 arm sampled early; the fraction that still finds the best arm "
         "is recorded (and compared with baselines in the Part 2 experiments).",
)
def bandits(ctx):
    env = BanditEnv((-1.0, 1.0))
    run = quick_train(pg_config("bandit", lr=0.05, episodes=150), env)
    p_best = float(probabilities(run["policy"], np.ones((1, 1)))[0, 1])
    ctx.metric("P(best arm), rewards ±1", p_best)
    ctx.check("deterministic ±1 bandit: P(best arm) > 0.95 after 150 episodes", p_best > 0.95, f"{p_best:.4f}")
    env = BanditEnv((0.3, 0.7), bernoulli=True, seed=0)
    run2 = quick_train(pg_config("bandit", lr=0.02, episodes=400), env)
    p_best2 = float(probabilities(run2["policy"], np.ones((1, 1)))[0, 1])
    ctx.metric("P(best arm), Bernoulli 0.3/0.7", p_best2)
    ctx.check("Bernoulli bandit: P(best arm) > 0.8 after 400 episodes", p_best2 > 0.8, f"{p_best2:.4f}")
    found = []
    for seed in range(10):
        run3 = quick_train(pg_config("bandit", lr=0.03, episodes=300, seed=seed), BanditEnv((0.0, 1.0, 0.5)))
        found.append(float(probabilities(run3["policy"], np.ones((1, 1)))[0, 1]) > 0.5)
    ctx.metric("positive-only rewards: seeds finding the best arm (of 10)", int(sum(found)))
    ctx.note(f"Positive-only three-armed bandit without baseline: {sum(found)}/10 seeds found the best arm; the others locked onto the 0.5 arm.")
    fig, ax = new_figure("Bandit training rewards (moving average over 20 episodes)", "Episode", "Reward")
    ax.plot(moving_average(run["rewards"], 20), color=SERIES[0], label="rewards −1 / +1")
    ax.plot(moving_average(run2["rewards"], 20), color=SERIES[1], label="Bernoulli 0.3 / 0.7")
    ax.legend()
    ctx.figure(fig, "bandits")


@validation_test(
    section="pg1",
    title="REINFORCE reaches the optimum computed by value iteration (corridor)",
    what="A corridor of 6 cells with the reward at the end (deterministic, and with a 20% chance that a move is "
         "reversed). The exact optimal policy and value come from the Section 2 solvers on the same MDP; REINFORCE "
         "is trained for 400 episodes and its greedy policy is evaluated exactly on the MDP.",
    how="Two independent implementations must agree: the greedy policy extracted from the trained network must be "
         "'always right' (the DP optimum), its exact value must equal the DP optimal value, and in the deterministic "
         "case the discounted return of a greedy rollout must be γ^(L−2) exactly. The stochastic policy's own "
         "average return must be close to the optimum once the per-state probabilities of 'right' are near 1.",
)
def corridor_vs_dp(ctx):
    L, gamma = 6, 0.9
    rows = []
    for slip in (0.0, 0.2):
        env = CorridorEnv(length=L, slip=slip, max_steps=40)
        mdp = corridor_mdp(L, slip, gamma)
        optimal = solvers.value_iteration(mdp, tol=1e-12)
        run = quick_train(pg_config("corridor", gamma=gamma, lr=0.05, episodes=400), env)
        probs = probabilities(run["policy"], np.eye(L))
        greedy = probs.argmax(axis=1)
        greedy_value = -solvers.evaluate_policy(mdp, greedy)[0]
        opt_value = -optimal.values[0]
        rollouts = [discounted_returns(run_episode(env, make_adapter(pg_config("corridor", gamma=gamma), env), run["policy"], CPU).rewards, gamma)[0] for _ in range(300)]
        sampled_value, se = mean_and_stderr(rollouts)
        rows.append({"slip": slip, "DP optimal value from start": round(opt_value, 4), "greedy policy value (exact)": round(greedy_value, 4),
                     "stochastic policy value (300 rollouts)": f"{sampled_value:.4f} ± {se:.4f}",
                     "P(right) per state": np.round(probs[:-1, 1], 3).tolist()})
        ctx.check(f"slip {slip}: greedy policy from the network equals the DP optimal policy", bool(np.all(greedy[:-1] == optimal.policy[:-1])),
                  f"greedy {greedy[:-1].tolist()}, optimal {optimal.policy[:-1].tolist()}")
        ctx.check_close(f"slip {slip}: exact value of the greedy policy = DP optimum", greedy_value, opt_value, 1e-9)
        ctx.check(f"slip {slip}: stochastic policy within 10% of the optimum", sampled_value >= 0.9 * opt_value, f"{sampled_value:.4f} vs {opt_value:.4f}")
        if slip == 0.0:
            greedy_return = discounted_returns(run_episode(env, make_adapter(pg_config("corridor", gamma=gamma), env), run["policy"], CPU, greedy=True).rewards, gamma)[0]
            ctx.check_close("deterministic corridor: greedy rollout return = γ^(L−2)", greedy_return, gamma ** (L - 2), 1e-9)
        ctx.metric(f"optimal value (slip {slip})", opt_value)
        if slip == 0.2:
            fig, ax = new_figure("Corridor (slip 0.2): REINFORCE reward per episode", "Episode", "Success (reward 1) and 50-episode average")
            ax.plot(run["rewards"], color="#86b6ef", linewidth=0.6, label="episode reward")
            ax.plot(moving_average(run["rewards"], 50), color=SERIES[0], linewidth=2, label="50-episode moving average")
            ax.legend()
            ctx.figure(fig, "corridor_learning_curve")
    ctx.table("corridor results", rows)


@validation_test(
    section="pg1",
    title="The discount factor decides whether a delayed reward is worth pursuing",
    what="A 5-cell corridor where every move to the right costs 0.1 and the goal pays 1. For γ ∈ {0, 0.5, 0.9, 0.99} "
         "the Section 2 solvers give the exact optimal policy of the corresponding MDP (go right from the start or "
         "not), and REINFORCE is trained with the same γ (3 seeds, 300 episodes); the success rate at the end of "
         "training is compared with what the optimal policy does. Also checks the return weights of one episode.",
    how="With γ = 0 only the immediate reward counts, so moving right (−0.1) is never worth it and the optimal agent "
         "stays at the start; with γ = 0.9 the discounted goal reward outweighs the move costs. Value iteration "
         "says exactly which γ flip the decision, and REINFORCE trained with each γ must reproduce that decision: "
         "near-zero success where the optimum is to stay, near-certain success where it is to go. This isolates "
         "the discount factor from every other part of the algorithm.",
)
def discount_factor(ctx):
    episode = Episode(np.eye(7, dtype=np.float32)[:6], np.ones(6, dtype=np.int64), np.array([0, 0, 0, 0, 0, 1], np.float32))
    ctx.check_close("γ = 0: weights are (0, 0, 0, 0, 0, 1)", step_weights(episode, 0.0, Options()), [0, 0, 0, 0, 0, 1], 1e-12)
    ctx.check_close("γ = 0.9: weights are 0.9^(5−t)", step_weights(episode, 0.9, Options()), 0.9 ** np.arange(5, -1, -1), 1e-12)
    L, right_reward = 5, -0.1
    rows, success, should_go, margin = [], {}, {}, {}
    for gamma in (0.0, 0.5, 0.9, 0.99):
        mdp = corridor_mdp(L, 0.0, gamma, right_reward=right_reward)
        optimal = solvers.value_iteration(mdp, tol=1e-12)
        q = -solvers.q_values(mdp, optimal.values)[0]  # value of (stay, go right) at the start
        should_go[gamma] = bool(optimal.policy[0] == 1)
        margin[gamma] = float(abs(q[1] - q[0]))
        per_seed = []
        for seed in range(3):
            env = CorridorEnv(length=L, slip=0.0, max_steps=30, right_reward=right_reward)
            run = quick_train(pg_config("corridor", gamma=gamma, lr=0.03, episodes=300, seed=seed, hidden=[16]), env)
            per_seed.append(float(np.mean([r["steps"] < 30 for r in run["records"][-50:]])))
        success[gamma] = float(np.mean(per_seed))
        rows.append({"γ": gamma, "optimal first move (value iteration)": "go right" if should_go[gamma] else "stay",
                     "value of 'stay' at the start": round(float(q[0]), 4), "value of 'go right' at the start": round(float(q[1]), 4),
                     "REINFORCE success rate, last 50 episodes (3 seeds)": round(success[gamma], 3), "per seed": per_seed})
        ctx.metric(f"success rate (γ = {gamma})", success[gamma])
    ctx.table("optimal decision versus learned behaviour", rows)
    ctx.check("value iteration: staying is optimal for γ = 0 and going is optimal for γ = 0.9", (not should_go[0.0]) and should_go[0.9])
    for gamma in (0.0, 0.5, 0.9, 0.99):
        if margin[gamma] < 0.1:
            ctx.note(f"γ = {gamma} is a marginal case (the two first moves differ by only {margin[gamma]:.3f} in value), so the "
                     f"learned success rate of {success[gamma]:.2f} is recorded without a pass/fail check.")
        elif should_go[gamma]:
            ctx.check(f"γ = {gamma}: REINFORCE reaches the goal in most episodes (≥ 80%)", success[gamma] >= 0.8, f"{success[gamma]:.2f}")
        else:
            ctx.check(f"γ = {gamma}: REINFORCE learns to stay (≤ 20% success)", success[gamma] <= 0.2, f"{success[gamma]:.2f}")


@validation_test(
    section="pg1",
    title="Three ways to weight the log-probabilities: which agree in expectation",
    what="With a fixed random policy on a stochastic corridor that charges 0.05 per step (so rewards arrive at every "
         "step, not only at the goal), draws 600 single-episode gradient estimates for "
         "three weightings of ∇log π(a_t|s_t): the whole-episode return G_0 (classic REINFORCE), γ^t·G_t (the exact "
         "gradient of the discounted objective, `--set discount_grad=true`) and the reward-to-go G_t without the "
         "γ^t factor (the implementation's default, as in most practical code), and compares means and variances.",
    how="Rewards earned before step t are independent of a_t given the history, so dropping them does not change "
         "the expectation: the G_0 and γ^t·G_t estimates must have the same mean within sampling error, and here the "
         "γ^t·G_t weights are pointwise smaller, so its variance must be lower. The default G_t weighting drops the "
         "γ^t factor and therefore estimates a slightly different gradient (it treats every visited state as a "
         "start state); the experiment records that difference so the choice is a documented one.",
)
def reward_to_go(ctx):
    torch.manual_seed(3)
    env = CorridorEnv(length=5, slip=0.2, max_steps=25, step_reward=-0.05)
    cfg = pg_config("corridor", gamma=0.9)
    policy = MLP(5, 2, hidden=())
    full = gradient_samples(policy, env, cfg, 600, options=Options(reward_to_go=False), seed=0)
    exact = gradient_samples(policy, env, cfg, 600, options=Options(discount_grad=True), seed=0)
    rtg = gradient_samples(policy, env, cfg, 600, options=Options(reward_to_go=True), seed=0)

    def z_scores(a, b):
        se = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
        return np.abs(a.mean(0) - b.mean(0)) / np.where(se > 0, se, 1.0)

    z_exact, z_rtg = z_scores(full, exact), z_scores(full, rtg)
    var = {"G_0": float(full.var(0).sum()), "γ^t G_t": float(exact.var(0).sum()), "G_t": float(rtg.var(0).sum())}
    ctx.table("gradient estimates (600 episodes each)", [
        {"weighting": "whole-episode return G_0", "mean ‖g‖": round(float(np.linalg.norm(full.mean(0))), 4), "total variance": round(var["G_0"], 4), "max |z| vs G_0": 0.0},
        {"weighting": "γ^t · G_t (exact discounted gradient)", "mean ‖g‖": round(float(np.linalg.norm(exact.mean(0))), 4), "total variance": round(var["γ^t G_t"], 4), "max |z| vs G_0": round(float(z_exact.max()), 2)},
        {"weighting": "reward-to-go G_t (default)", "mean ‖g‖": round(float(np.linalg.norm(rtg.mean(0))), 4), "total variance": round(var["G_t"], 4), "max |z| vs G_0": round(float(z_rtg.max()), 2)},
    ])
    ctx.metric("max |z|: G_0 vs γ^t G_t", float(z_exact.max()))
    ctx.metric("max |z|: G_0 vs G_t", float(z_rtg.max()))
    ctx.metric("variance ratio γ^t G_t / G_0", var["γ^t G_t"] / var["G_0"])
    ctx.check("G_0 and γ^t·G_t have the same mean (within 4 s.e. in every coordinate)", bool(np.all(z_exact <= 4.0)), f"max z = {z_exact.max():.2f}")
    ctx.check("γ^t·G_t has lower total variance than G_0", var["γ^t G_t"] < var["G_0"], f"{var['γ^t G_t']:.4f} < {var['G_0']:.4f}")
    ctx.note(f"The default reward-to-go weighting differs from G_0 in expectation (max z = {z_rtg.max():.1f}): it is the usual "
             "practical estimator, equal to the exact discounted gradient only when γ = 1.")


@validation_test(
    section="pg1",
    title="CartPole-v1 learns within 200 episodes on the CPU",
    what="Trains with the handout's settings (Adam, learning rate 0.01, γ = 0.95) for 300 episodes, seed 0, "
         "then evaluates 50 episodes; records the mean reward of the first 20 and of the last 50 training episodes.",
    how="A correct REINFORCE implementation improves CartPole quickly: the last 50 episodes must average at least "
         "1.5 times the first 20 and at least 100 steps, and the evaluation mean must exceed 50 (random play lasts "
         "about 22). The full 1000-episode run and its plots are produced by `python run.py pg --part 1 --env cartpole`.",
)
def cartpole_short(ctx):
    env = make_env("cartpole")
    try:
        start = time.time()
        cfg = pg_config("cartpole", gamma=0.95, lr=0.01, episodes=300)
        run = quick_train(cfg, env)
        elapsed = time.time() - start
        ma = moving_average(run["rewards"], 50)
        eval_rewards = evaluate(cfg, env, run["policy"], 50)
    finally:
        env.close()
    first, last = float(np.mean(run["rewards"][:20])), float(np.mean(run["rewards"][-50:]))
    ctx.metric("mean reward, episodes 1-20", first)
    ctx.metric("mean reward, episodes 251-300", last)
    ctx.metric("evaluation mean (50 episodes)", float(np.mean(eval_rewards)))
    ctx.metric("evaluation std", float(np.std(eval_rewards)))
    ctx.metric("training time (s)", round(elapsed, 1))
    ctx.check("the last 50 episodes average ≥ 1.5 × the first 20 and ≥ 100 steps", last >= 1.5 * first and last >= 100, f"{first:.1f} → {last:.1f}")
    ctx.check("evaluation mean > 50 steps", np.mean(eval_rewards) > 50, f"{np.mean(eval_rewards):.1f}")
    fig, ax = new_figure("CartPole-v1, 300 training episodes (seed 0)", "Episode", "Episode reward")
    ax.plot(run["rewards"], color="#86b6ef", linewidth=0.8, label="episode reward")
    ax.plot(ma, color=SERIES[0], linewidth=2, label="50-episode moving average")
    ax.legend()
    ctx.figure(fig, "cartpole_300")


@validation_test(
    section="pg1",
    title="Pong-v5 pipeline: preprocessing and the two-action restriction",
    what="Plays one full game of ALE/Pong-v5 with an untrained policy, recording every action sent to the "
         "environment and statistics of the preprocessed frame differences the network sees.",
    how="The handout restricts the agent to RIGHT (2) and LEFT (3): every recorded action must be one of them. "
         "The preprocessed input must be an 80×80 image with values in {−1, 0, 1} (frame difference of binary "
         "frames) that is non-zero most of the time (the ball and paddles move), and a game must end with a score "
         "between −21 and 21 (an untrained agent loses almost every point).",
)
def pong_pipeline(ctx):
    env = RecordingEnv(make_env("pong"))
    try:
        cfg = pg_config("pong", gamma=0.99)
        policy = build_policy(cfg, env)
        adapter = make_adapter(cfg, env)
        start = time.time()
        episode = run_episode(env, adapter, policy, CPU, seed=0)
        elapsed = time.time() - start
    finally:
        env.close()
    values = set(np.unique(episode.observations).tolist())
    nonzero = float(np.mean(np.abs(episode.observations).sum(axis=1) > 0))
    ctx.metric("steps in the game", episode.length)
    ctx.metric("game score (reward sum)", episode.total_reward)
    ctx.metric("fraction of frame differences with any non-zero pixel", nonzero)
    ctx.metric("seconds per step (CPU, untrained MLP)", round(elapsed / max(episode.length, 1), 4))
    ctx.check("only actions 2 (RIGHT) and 3 (LEFT) were sent to the environment", set(env.actions) <= set(PONG_ACTIONS), f"actions used: {sorted(set(env.actions))}")
    ctx.check("both actions were used", set(env.actions) == set(PONG_ACTIONS))
    ctx.check("inputs are 6400-dimensional with values in {−1, 0, 1}", episode.observations.shape[1] == 6400 and values <= {-1.0, 0.0, 1.0}, f"values {sorted(values)}")
    ctx.check("most frame differences are non-zero", nonzero > 0.5, f"{nonzero:.2%}")
    ctx.check("the score is between −21 and 21 and the game ended", -21 <= episode.total_reward <= 21 and episode.length > 100)
