"""Validation experiments for Policy Gradient Part 2 (REINFORCE with a baseline).

Run with:  python run.py tests --section pg2        (a few minutes on a laptop CPU)

The baseline of `python run.py pg --part 2` is a learned state-value network (`--baseline value`); a constant
moving-average baseline (`mean`) and a per-time-step baseline (`time`) are also implemented. These experiments
check what the theory promises (unchanged expected gradient, lower variance: lean/RLVerify/Baseline.lean),
show the baseline fixing a failure mode of plain REINFORCE, check that the value network learns the policy's
true values, and compare learning curves with and without a baseline.
"""
from __future__ import annotations

import numpy as np
import torch

from rl.dp import solvers
from rl.pg.baselines import MeanBaseline, make_baseline
from rl.pg.envs import make_adapter, make_env
from rl.pg.options import Options
from rl.pg.policy import MLP
from rl.pg.reinforce import discounted_returns, evaluate, run_episode
from rl.pg.toy import BanditEnv, CorridorEnv, corridor_mdp
from rl.plotting import SERIES, moving_average, new_figure
from tests.harness import validation_test
from tests.support import CPU, gradient_samples, pg_config, probabilities, quick_train


@validation_test(
    section="pg2",
    title="A baseline leaves the expected gradient unchanged and changes its variance",
    what="On a Bernoulli bandit (success probabilities 0.3 and 0.7) with a fixed policy, draws 2000 single-episode "
         "gradient estimates with no baseline, with the constant baseline b = 0.5 (the mean reward), with a "
         "deliberately bad constant b = 5, and with an untrained value network, and compares their means and "
         "variances.",
    how="Any baseline that does not depend on the action keeps the estimator unbiased (Lean: baseline_unbiased), "
         "so all four means must agree within sampling error. The variance is minimised near the mean return: "
         "b = 0.5 must have lower variance than no baseline, and b = 5 a much higher one, which is why the baseline "
         "has to be fitted rather than chosen arbitrarily.",
)
def unbiased_lower_variance(ctx):
    torch.manual_seed(0)
    env = BanditEnv((0.3, 0.7), bernoulli=True, seed=0)
    cfg = pg_config("bandit", part=2)
    policy = MLP(1, 2, hidden=())
    good, bad = MeanBaseline(), MeanBaseline()
    good.value, bad.value = 0.5, 5.0
    value_net = make_baseline("value", pg_config("bandit", part=2, baseline_hidden=[8]), env)
    samples = {"none": gradient_samples(policy, env, cfg, 2000, None, seed=0),
               "constant b = 0.5": gradient_samples(policy, env, cfg, 2000, good, seed=0),
               "constant b = 5": gradient_samples(policy, env, cfg, 2000, bad, seed=0),
               "untrained value network": gradient_samples(policy, env, cfg, 2000, value_net, seed=0)}
    ref = samples["none"]
    rows = []
    for name, g in samples.items():
        se = np.sqrt(g.var(0, ddof=1) / len(g) + ref.var(0, ddof=1) / len(ref))
        z = np.abs(g.mean(0) - ref.mean(0)) / np.where(se > 0, se, 1.0)
        rows.append({"baseline": name, "mean gradient (bias of arm 2)": round(float(g.mean(0)[-1]), 4), "s.e.": round(float(g.std(0, ddof=1)[-1] / np.sqrt(len(g))), 4),
                     "total variance": round(float(g.var(0).sum()), 4), "max |z| vs none": round(float(z.max()), 2)})
        if name != "none":
            ctx.check(f"{name}: same mean gradient as no baseline (within 4 s.e.)", bool(np.all(z <= 4.0)), f"max z = {z.max():.2f}")
    ctx.table("gradient estimates by baseline", rows)
    v = {name: float(g.var(0).sum()) for name, g in samples.items()}
    ctx.check("b = 0.5 lowers the variance", v["constant b = 0.5"] < v["none"], f"{v['constant b = 0.5']:.3f} < {v['none']:.3f}")
    ctx.check("b = 5 raises the variance", v["constant b = 5"] > v["none"], f"{v['constant b = 5']:.3f} > {v['none']:.3f}")
    ctx.metric("variance ratio (b = 0.5) / none", v["constant b = 0.5"] / v["none"])
    ctx.metric("variance ratio (b = 5) / none", v["constant b = 5"] / v["none"])


@validation_test(
    section="pg2",
    title="Baselines prevent lock-in on the first rewarded arm",
    what="Three-armed bandit with positive rewards 0, 1 and 0.5, learning rate 0.03, 300 episodes, 10 seeds, "
         "trained without a baseline and with the mean, time and value baselines; counts the seeds whose final "
         "policy prefers the best arm.",
    how="Without a baseline every sampled arm is reinforced (all returns are positive), so Adam can lock onto the "
         "0.5 arm before the 1.0 arm is ever tried. A baseline near the average return makes the 0.5 arm's advantage "
         "zero or negative, so the policy keeps exploring until it finds the 1.0 arm: the baselines must find the best "
         "arm at least as often as plain REINFORCE.",
)
def lock_in(ctx):
    found = {}
    rows = []
    for baseline in ("none", "mean", "time", "value"):
        hits = []
        for seed in range(10):
            env = BanditEnv((0.0, 1.0, 0.5))
            run = quick_train(pg_config("bandit", part=2, lr=0.03, episodes=300, seed=seed, baseline=baseline), env)
            hits.append(float(probabilities(run["policy"], np.ones((1, 1)))[0, 1]) > 0.5)
        found[baseline] = int(sum(hits))
        rows.append({"baseline": baseline, "seeds finding the best arm (of 10)": found[baseline], "per seed": ["✓" if h else "✗" for h in hits]})
        ctx.metric(f"seeds finding the best arm: {baseline}", found[baseline])
    ctx.table("lock-in on the 0.5 arm", rows)
    ctx.check("the value baseline finds the best arm at least as often as no baseline", found["value"] >= found["none"], str(found))
    ctx.check("the time baseline finds the best arm at least as often as no baseline", found["time"] >= found["none"], str(found))
    ctx.check("the best baseline finds the best arm in strictly more seeds than no baseline", max(found[b] for b in ("mean", "time", "value")) > found["none"], str(found))


@validation_test(
    section="pg2",
    title="The value baseline learns the policy's state values",
    what="Trains REINFORCE with the value baseline on a 6-cell corridor (10% slip, γ = 0.9) for 500 episodes, "
         "then compares the value network's prediction in every state with the exact value of the learned "
         "stochastic policy (computed by policy evaluation on the corridor MDP) and with the optimal value.",
    how="The baseline network is fitted by least squares to the observed returns, whose expectation is the value "
         "of the current policy, so V_φ(s) must be close to V^π(s) in the states the policy visits (within 0.1 on a "
         "0-1 scale) and must increase towards the goal. Because the policy is near-optimal by the end, V_φ is also "
         "close to V*. This checks that the baseline is state-dependent and informative, not a constant.",
)
def value_baseline_learns(ctx):
    L, slip, gamma = 6, 0.1, 0.9
    env = CorridorEnv(length=L, slip=slip, max_steps=40)
    cfg = pg_config("corridor", part=2, gamma=gamma, lr=0.03, episodes=500, baseline="value", baseline_lr=0.02, hidden=[16])
    run = quick_train(cfg, env)
    mdp = corridor_mdp(L, slip, gamma)
    probs = probabilities(run["policy"], np.eye(L))
    P_pi = np.einsum("sa,asj->sj", probs, mdp.transitions)
    c_pi = np.einsum("sa,sa->s", probs, mdp.costs)
    v_policy = -np.linalg.solve(np.eye(L) - gamma * P_pi, c_pi)
    v_optimal = -solvers.value_iteration(mdp, tol=1e-12).values
    v_critic = run["baseline"].predict(torch.eye(L), torch.arange(L)).numpy()
    rows = [{"state": s, "P(right)": round(float(probs[s, 1]), 3), "V^π(s) exact": round(float(v_policy[s]), 4), "V*(s)": round(float(v_optimal[s]), 4),
             "value network": round(float(v_critic[s]), 4)} for s in range(L - 1)]
    ctx.table("values per state (goal state excluded)", rows)
    err = np.abs(v_critic[:-1] - v_policy[:-1])
    ctx.metric("max |V_φ − V^π| over non-terminal states", float(err.max()))
    ctx.metric("max |V_φ − V*|", float(np.abs(v_critic[:-1] - v_optimal[:-1]).max()))
    ctx.check("value network within 0.1 of the policy's exact values in every non-terminal state", bool(np.all(err <= 0.1)), f"max error {err.max():.3f}")
    ctx.check("value network increases towards the goal", bool(np.all(np.diff(v_critic[:-1]) > 0)), str(np.round(v_critic[:-1], 3).tolist()))
    ctx.check("the learned policy is near-optimal (P(right) ≥ 0.9 everywhere)", bool(np.all(probs[:-1, 1] >= 0.9)))


@validation_test(
    section="pg2",
    title="Learning curves with and without a baseline",
    what="A 10-cell corridor with 15% slip, learning rate 0.02, 300 episodes, 5 seeds each for no baseline, the "
         "mean baseline and the value baseline; records the average success rate over training (area under the "
         "learning curve), the final success rate, and the root-mean-square of the per-step weights (G_t − b_t) "
         "relative to that of G_t, measured on fresh episodes with the final policy and baseline.",
    how="The variance of the gradient estimate scales with E[(G_t − b_t)²], so a useful baseline must shrink the "
         "root-mean-square weight (a constant baseline cannot change the standard deviation of the weights, only "
         "their mean square). Whether lower-variance updates also learn faster depends on the problem, so the "
         "areas under the learning curves are recorded and compared rather than asserted: on this sparse-reward "
         "corridor the only reward is a success, so plain REINFORCE's habit of reinforcing every success is not a "
         "handicap, while the value baseline first has to learn the values it subtracts.",
)
def learning_curves(ctx):
    L, slip, seeds = 10, 0.15, 5
    curves, auc, spread, rows = {}, {}, {}, []
    for baseline in ("none", "mean", "value"):
        rewards_all, ratios = [], []
        for seed in range(seeds):
            env = CorridorEnv(length=L, slip=slip, max_steps=80)
            cfg = pg_config("corridor", part=2, gamma=0.9, lr=0.02, episodes=300, seed=seed, baseline=baseline, hidden=[16])
            run = quick_train(cfg, env)
            rewards_all.append(run["rewards"])
            # spread of the weights on fresh episodes with the final policy and baseline
            adapter = make_adapter(cfg, env)
            weights, returns = [], []
            for _ in range(30):
                ep = run_episode(env, adapter, run["policy"], CPU)
                G = discounted_returns(ep.rewards, 0.9)
                b = run["baseline"].predict(torch.as_tensor(ep.observations), torch.arange(ep.length)).numpy() if run["baseline"] is not None else np.zeros_like(G)
                weights.extend(G - b)
                returns.extend(G)
            rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
            ratios.append(rms(weights) / rms(returns) if rms(returns) > 0 else 1.0)
        curve = np.mean(rewards_all, axis=0)
        curves[baseline] = curve
        auc[baseline] = float(curve.mean())
        spread[baseline] = float(np.mean(ratios))
        rows.append({"baseline": baseline, "mean success over training": round(auc[baseline], 3), "success in the last 50 episodes": round(float(curve[-50:].mean()), 3),
                     "rms(G − b) / rms(G)": round(spread[baseline], 3)})
        ctx.metric(f"area under the learning curve: {baseline}", auc[baseline])
    ctx.table("5-seed averages", rows)
    ctx.check("the value baseline shrinks the root-mean-square update weight", spread["value"] < 0.9, f"ratio {spread['value']:.3f}")
    ctx.check("the mean baseline shrinks the root-mean-square update weight", spread["mean"] < 1.0, f"ratio {spread['mean']:.3f}")
    ctx.note(f"Area under the learning curve (mean success over 300 episodes, 5 seeds): none {auc['none']:.3f}, mean baseline "
             f"{auc['mean']:.3f}, value baseline {auc['value']:.3f}.")
    fig, ax = new_figure("Corridor (10 cells, 15% slip): success rate, mean of 5 seeds", "Episode", "Success rate (20-episode moving average)")
    for color, (name, curve) in zip(SERIES, curves.items()):
        ax.plot(moving_average(curve, 20), color=color, linewidth=2, label=f"baseline: {name}")
    ax.legend()
    ctx.figure(fig, "learning_curves")


@validation_test(
    section="pg2",
    title="CartPole-v1 with the value baseline (200 episodes on the CPU)",
    what="Trains CartPole-v1 for 200 episodes with the value baseline (handout settings otherwise), evaluates 50 "
         "episodes, and measures on 20 fresh episodes how much the baseline shrinks the update weights "
         "(root-mean-square of G_t − V(s_t) relative to G_t).",
    how="The baseline must not stop CartPole from learning (moving average at least doubles, evaluation mean > 50) "
         "and must shrink the root-mean-square weight, which governs the variance of the gradient estimate, by a "
         "clear margin. The full 1000-episode run is `python run.py pg --part 2 --env cartpole`.",
)
def cartpole_with_baseline(ctx):
    env = make_env("cartpole")
    try:
        cfg = pg_config("cartpole", part=2, gamma=0.95, lr=0.01, episodes=200, baseline="value")
        run = quick_train(cfg, env)
        ma = moving_average(run["rewards"], 50)
        eval_rewards = evaluate(cfg, env, run["policy"], 50)
        adapter = make_adapter(cfg, env)
        weights, returns = [], []
        for _ in range(20):
            ep = run_episode(env, adapter, run["policy"], CPU)
            G = discounted_returns(ep.rewards, 0.95)
            b = run["baseline"].predict(torch.as_tensor(ep.observations), torch.arange(ep.length)).numpy()
            weights.extend(G - b)
            returns.extend(G)
    finally:
        env.close()
    ratio = float(np.sqrt(np.mean(np.square(weights))) / np.sqrt(np.mean(np.square(returns))))
    ctx.metric("moving average, episodes 1-50", float(ma[49]))
    ctx.metric("moving average, episodes 151-200", float(ma[-1]))
    ctx.metric("evaluation mean (50 episodes)", float(np.mean(eval_rewards)))
    ctx.metric("rms(G − V(s)) / rms(G) on fresh episodes", ratio)
    ctx.metric("final value loss", float(run["records"][-1].get("value_loss", float("nan"))))
    ctx.check("the 50-episode moving average at least doubles", ma[-1] >= 2 * ma[49], f"{ma[49]:.1f} → {ma[-1]:.1f}")
    ctx.check("evaluation mean > 50 steps", np.mean(eval_rewards) > 50, f"{np.mean(eval_rewards):.1f}")
    ctx.check("the baseline shrinks the root-mean-square update weight by at least 30%", ratio < 0.7, f"ratio {ratio:.3f}")
    fig, ax = new_figure("CartPole-v1 with the value baseline, 200 episodes (seed 0)", "Episode", "Episode reward")
    ax.plot(run["rewards"], color="#86b6ef", linewidth=0.8, label="episode reward")
    ax.plot(ma, color=SERIES[0], linewidth=2, label="50-episode moving average")
    ax.legend()
    ctx.figure(fig, "cartpole_200_baseline")
