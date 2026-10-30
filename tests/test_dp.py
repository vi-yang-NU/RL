"""Validation experiments for Section 2: Value and Policy Iteration.

Run with:  python run.py tests --section dp          (about a minute on a laptop)
           python run.py tests --section dp -k trace  (one experiment)

Each experiment changes the parameters or the structure of the shuttle problem, or isolates a few steps, and
checks that the solvers behave as the theory predicts. Several checks mirror theorems proved in
lean/RLVerify/Shuttle.lean (named in the descriptions). Results go to results/tests/dp.json and are shown in
the dashboard and the report.
"""
from __future__ import annotations

import json
from fractions import Fraction

import numpy as np

from rl.dp import solvers
from rl.dp.shuttle import ShuttleParams, build_mdp
from rl.outputs import ROOT
from rl.plotting import SERIES, new_figure
from tests.harness import validation_test

HANDOUT = ShuttleParams()
LEAN_REFERENCE = ROOT / "lean" / "reference" / "shuttle_reference.json"


def solve_all(params: ShuttleParams, tol: float = 1e-9):
    mdp = build_mdp(params)
    return mdp, solvers.enumeration(mdp, params.horizon), solvers.value_iteration(mdp, tol=tol), solvers.policy_iteration(mdp)


def threshold_of(params: ShuttleParams) -> int | None:
    return solvers.policy_threshold(solvers.policy_iteration(build_mdp(params)).policy)


@validation_test(
    section="dp",
    title="Three solvers, one answer",
    what="Solves the handout's problem (K = 15, c_f = 100, c_h = 2, A ~ Unif{1..5}, at most 200 waiting, γ = 0.95) "
         "with enumeration over T = 500 periods, value iteration and policy iteration, and compares the three "
         "value functions and policies state by state.",
    how="The three methods are different algorithms for the same optimality equation, so they must agree: "
         "enumeration with T = 500 differs from the infinite-horizon solution by at most γ^500 · max V ≈ 3e-8, and "
         "value iteration's stopping rule guarantees ‖V − V*‖ ≤ γ·tol/(1 − γ) (Lean: vi_error_bound). The values "
         "of policy iteration must have a zero Bellman residual, all policies must coincide, and the optimal policy "
         "must be a threshold rule (dispatch once enough customers wait).",
)
def methods_agree(ctx):
    mdp, enum, vi, pi = solve_all(HANDOUT)
    d_enum_pi = float(np.max(np.abs(enum.values - pi.values)))
    d_vi_pi = float(np.max(np.abs(vi.values - pi.values)))
    d_enum_vi = float(np.max(np.abs(enum.values - vi.values)))
    for name, value in {"max |V_enum − V_PI|": d_enum_pi, "max |V_VI − V_PI|": d_vi_pi, "max |V_enum − V_VI|": d_enum_vi,
                        "VI iterations": vi.iterations, "VI error bound": vi.info["error_bound"],
                        "PI improvement steps": pi.iterations, "dispatch threshold": solvers.policy_threshold(pi.policy),
                        "V*(0)": float(pi.values[0]), "V*(200)": float(pi.values[200])}.items():
        ctx.metric(name, value)
    ctx.check("enumeration (T = 500) and policy iteration agree to 1e-6", d_enum_pi < 1e-6, f"max difference {d_enum_pi:.2e}")
    ctx.check("value iteration is within its proven error bound of policy iteration", d_vi_pi <= vi.info["error_bound"] + 1e-12,
              f"{d_vi_pi:.2e} ≤ {vi.info['error_bound']:.2e}")
    ctx.check("the three policies are identical", np.array_equal(enum.policy, pi.policy) and np.array_equal(vi.policy, pi.policy))
    ctx.check("Bellman residual of the policy-iteration values is ~0", pi.info["bellman_residual"] < 1e-9, f"{pi.info['bellman_residual']:.2e}")
    ctx.check("the optimal policy is a threshold policy", solvers.is_monotone(pi.policy), f"dispatch iff s ≥ {solvers.policy_threshold(pi.policy)}")
    ctx.check("values are non-decreasing in the number waiting (Lean: value_mono_state)", bool(np.all(np.diff(pi.values) >= -1e-9)))
    ctx.table("values and policy at selected states", [
        {"s": s, "V enumeration": round(float(enum.values[s]), 4), "V value iteration": round(float(vi.values[s]), 4),
         "V policy iteration": round(float(pi.values[s]), 4), "optimal action": mdp.action_labels[int(pi.policy[s])]}
        for s in (0, 5, 10, 12, 13, 14, 15, 20, 30, 50, 100, 200)
    ])
    fig, ax = new_figure("Optimal value function from the three methods", "Customers waiting", "Expected discounted cost")
    for color, (label, values) in zip(SERIES, {"Enumeration (T = 500)": enum.values, "Value iteration": vi.values, "Policy iteration": pi.values}.items()):
        ax.plot(mdp.states, values, color=color, label=label, linewidth=2, linestyle="-" if label.startswith("E") else "--")
    ax.legend()
    ctx.figure(fig, "value_functions")


@validation_test(
    section="dp",
    title="Value iteration contracts at rate γ",
    what="Runs value iteration for γ ∈ {0.5, 0.8, 0.95} and records the change ‖V_{n+1} − V_n‖∞ after every "
         "iteration, the ratio of successive changes, and the distance of the final iterate from the policy-iteration "
         "solution.",
    how="The Bellman operator is a γ-contraction (Lean: bellman_contraction), so successive changes must shrink by "
         "at least the factor γ every iteration (value_succ_sub_bound) and the number of iterations to a tolerance "
         "must scale like log(tol)/log(γ). The stopping rule's a-posteriori bound γ·Δ/(1 − γ) must dominate the true "
         "error (vi_error_bound). Ratios are only examined while the changes are far above floating-point noise.",
)
def contraction(ctx):
    rows = []
    curves = {}
    for gamma in (0.5, 0.8, 0.95):
        params = HANDOUT.with_(gamma=gamma)
        mdp = build_mdp(params)
        vi = solvers.value_iteration(mdp, tol=1e-8)  # 1e-8: far above floating-point noise in values ~ 4000
        pi = solvers.policy_iteration(mdp)
        residuals = np.array(vi.history["‖V_{n+1} − V_n‖∞"])
        floor = vi.info["contraction_noise_floor"]
        reliable = residuals[:-1] > floor
        ratios = residuals[1:][reliable] / residuals[:-1][reliable]
        geometric_ok = bool(np.all(residuals[1:][reliable] <= residuals[0] * gamma ** np.arange(1, len(residuals))[reliable] * (1 + 1e-6)))
        true_error = float(np.max(np.abs(vi.values - pi.values)))
        predicted_iterations = np.log(1e-8 / residuals[0]) / np.log(gamma)
        curves[f"γ = {gamma}"] = residuals
        rows.append({"γ": gamma, "iterations": vi.iterations, "predicted ≈ log(tol/Δ₁)/log γ": round(float(predicted_iterations), 1),
                     "max ratio Δ_{n+1}/Δ_n": round(float(ratios.max()), 6), "ratios checked": int(reliable.sum()),
                     "true error vs PI": f"{true_error:.2e}", "error bound": f"{vi.info['error_bound']:.2e}"})
        ctx.check(f"γ = {gamma}: every ratio of successive changes ≤ γ", bool(np.all(ratios <= gamma + 1e-6)), f"max ratio {ratios.max():.6f}")
        ctx.check(f"γ = {gamma}: Δ_n ≤ γ^(n−1) Δ_1 (geometric decay)", geometric_ok)
        ctx.check(f"γ = {gamma}: true error ≤ stopping-rule bound", true_error <= vi.info["error_bound"] + 1e-12, f"{true_error:.2e} ≤ {vi.info['error_bound']:.2e}")
        ctx.check(f"γ = {gamma}: iteration count within 15% of the contraction prediction", abs(vi.iterations - predicted_iterations) <= 0.15 * predicted_iterations + 2,
                  f"{vi.iterations} vs {predicted_iterations:.0f}")
        ctx.metric(f"iterations (γ = {gamma})", vi.iterations)
    ctx.table("convergence summary", rows)
    ctx.table("first 10 changes for γ = 0.95", [{"n": n + 1, "‖V_{n+1} − V_n‖∞": float(curves['γ = 0.95'][n]),
                                                "ratio": float(curves['γ = 0.95'][n] / curves['γ = 0.95'][n - 1]) if n else None} for n in range(10)])
    fig, ax = new_figure("Value iteration: change per iteration", "Iteration n", "‖V_{n+1} − V_n‖∞")
    for color, (label, res) in zip(SERIES, curves.items()):
        ax.semilogy(np.arange(1, len(res) + 1), res, color=color, label=label, linewidth=2)
        gamma = float(label.split("= ")[1])
        ax.semilogy(np.arange(1, len(res) + 1), res[0] * gamma ** np.arange(len(res)), color=color, linestyle=":", linewidth=1, label=f"{label}: Δ₁ γⁿ⁻¹")
    ax.legend(ncol=2)
    ctx.figure(fig, "contraction")


@validation_test(
    section="dp",
    title="The first 10 steps of backward induction, and 10 periods of the optimal policy",
    what="On a station capped at 30 customers (same costs, K and arrivals as the handout), records the value "
         "function and the optimal first action after each of the first 10 backward-induction steps, recomputes "
         "the first two steps by hand in exact rational arithmetic, and then follows the optimal policy for 10 "
         "periods from an empty station, printing state, action, cost and next state.",
    how="With one period left no shuttle is worth dispatching (100 + 2·(s − 15)⁺ > 2s for s ≤ 30), so V₁(s) = 2s "
         "exactly; V₂ must equal the hand computation; values must grow with the horizon (Lean: value_mono_horizon) "
         "and with the queue (value_mono_state); the dispatch threshold must appear and settle as more periods are "
         "added. The 10-period trajectory lets one verify each cost and transition by hand (dispatch exactly "
         "when the queue reaches the threshold, 15 customers leave, 1-5 arrive).",
)
def first_ten_steps(ctx):
    params = HANDOUT.with_(max_customers=30, horizon=10)
    mdp = build_mdp(params)
    result = solvers.enumeration(mdp, 10, record=10)
    trace = result.info["values_by_steps_to_go"]
    policies = result.info["policies_by_steps_to_go"]
    states = (0, 5, 10, 13, 15, 20, 30)
    rows = []
    for k, V in enumerate(trace):
        row = {"periods to go": k}
        row.update({f"V({s})": round(float(V[s]), 4) for s in states})
        row["dispatch threshold"] = None if k == 0 else solvers.policy_threshold(policies[k - 1])
        rows.append(row)
    ctx.table("value function after each backward step", rows)
    ctx.check("V₀ ≡ 0", bool(np.all(trace[0] == 0)))
    ctx.check_close("V₁(s) = 2s (no dispatch with one period left)", trace[1], 2.0 * np.arange(31), 1e-12)
    # exact hand computation of V_2 with fractions
    V1 = [Fraction(2 * s) for s in range(31)]
    V2 = []
    for s in range(31):
        qs = []
        for a in (0, 1):
            left = max(s - 15 * a, 0)
            expect = sum(Fraction(1, 5) * V1[min(left + arr, 30)] for arr in (1, 2, 3, 4, 5))
            qs.append(100 * a + 2 * left + Fraction(19, 20) * expect)
        V2.append(float(min(qs)))
    ctx.check_close("V₂ equals the hand computation in exact arithmetic", trace[2], V2, 1e-9)
    ctx.check("values never decrease as periods are added (Lean: value_mono_horizon)",
              all(np.all(trace[k + 1] >= trace[k] - 1e-12) for k in range(10)))
    ctx.check("every V_k is non-decreasing in the state (Lean: value_mono_state)",
              all(np.all(np.diff(V) >= -1e-12) for V in trace))
    thresholds = [solvers.policy_threshold(p) for p in policies]
    ctx.check("a dispatch threshold appears within the first 10 steps and then settles", thresholds[-1] is not None and thresholds[-1] == thresholds[-2],
              f"thresholds by horizon: {thresholds}")
    pi = solvers.policy_iteration(mdp)
    sim = solvers.simulate_policy(mdp, pi.policy, start=0, horizon=10, episodes=1, seed=7)
    ctx.table("10 periods of the optimal policy from an empty station (one random trajectory)", sim["trace"])
    ok = all((step["action"] == 1) == (step["state"] >= solvers.policy_threshold(pi.policy)) for step in sim["trace"])
    ok &= all(step["cost"] == (100 + 2 * max(step["state"] - 15, 0) if step["action"] else 2 * step["state"]) for step in sim["trace"])
    ok &= all(1 <= step["next_state"] - max(step["state"] - 15 * step["action"], 0) <= 5 for step in sim["trace"])
    ctx.check("each step of the trajectory follows the model (threshold, cost, 1-5 arrivals)", ok)


@validation_test(
    section="dp",
    title="Exact rational arithmetic agrees with the floating-point solvers",
    what="Re-solves four small instances (two arrival values; a non-uniform arrival distribution; deterministic "
         "arrivals; the handout's costs on a 20-customer station) by backward induction in exact rational "
         "arithmetic, twice: with Python fractions and with the Lean definition that the formal proofs are about "
         "(lean/RLVerify/Shuttle.lean, exported to lean/reference/shuttle_reference.json), and compares with "
         "enumeration.",
    how="The floating-point solver must reproduce the exact values to round-off (1e-12) at every horizon and "
         "state, and the optimal first actions must coincide. The Lean values are computed by the very definition "
         "whose properties are proved, so this ties the Python implementation to the verified model.",
)
def exact_reference(ctx):
    from tests.unit.test_dp_lean_reference import INSTANCES, exact_values, params_from_reference, params_of

    worst = 0.0
    for name, inst in INSTANCES.items():
        mdp = build_mdp(params_of(inst))
        for n, exact in enumerate(exact_values(inst, inst["horizons"])):
            worst = max(worst, float(np.max(np.abs(solvers.enumeration(mdp, n).values - [float(v) for v in exact]))))
    ctx.metric("max |float − exact fractions|", worst)
    ctx.check("enumeration matches exact fractions to 1e-12 on all instances and horizons", worst < 1e-12, f"{worst:.2e}")
    if not LEAN_REFERENCE.exists():
        ctx.check("Lean reference file present (cd lean && lake env lean --run ShuttleRef.lean > reference/shuttle_reference.json)", False)
        return
    data = json.loads(LEAN_REFERENCE.read_text())
    worst_lean, policy_mismatch, rows = 0.0, 0, []
    for inst in data["instances"]:
        p = inst["params"]
        mdp = build_mdp(params_from_reference(p))
        for n, row in enumerate(inst["values"]):
            res = solvers.enumeration(mdp, n)
            worst_lean = max(worst_lean, float(np.max(np.abs(res.values - [float(Fraction(x)) for x in row]))))
            if n >= 1:
                policy_mismatch += int(np.sum(np.asarray(res.policy) != np.asarray(inst["first_actions"][n - 1])))
        rows.append({"instance": inst["name"], "states": p["max_customers"] + 1, "horizons": len(inst["values"]) - 1,
                     "V_T(0) exact (Lean)": inst["values"][-1][0], "V_T(0) float": float(solvers.enumeration(mdp, len(inst['values']) - 1).values[0])})
    ctx.table("Lean reference instances", rows)
    ctx.metric("max |float − Lean exact|", worst_lean)
    ctx.check("enumeration matches the Lean-computed exact values to 1e-12", worst_lean < 1e-12, f"{worst_lean:.2e}")
    ctx.check("optimal first actions match Lean's", policy_mismatch == 0, f"{policy_mismatch} mismatches")


@validation_test(
    section="dp",
    title="γ → 0 makes the agent myopic; more foresight dispatches earlier",
    what="Solves the problem for γ from 0 to 0.99 and records the dispatch threshold and V(0).",
    how="With γ = 0 only the current period counts, and dispatching (≥ 100) is never cheaper than holding "
         "(≤ 2·200 but 100 + 2(s−15) > 2s), so the myopic agent never dispatches and V(s) = 2s exactly. As γ grows, "
         "future holding costs weigh more, so the threshold must not increase and V(0) must increase.",
)
def discount_sweep(ctx):
    gammas = (0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99)
    rows, thresholds, v0s, curves = [], [], [], {}
    for gamma in gammas:
        mdp = build_mdp(HANDOUT.with_(gamma=gamma))
        pi = solvers.policy_iteration(mdp)
        t = solvers.policy_threshold(pi.policy)
        thresholds.append(t if t is not None else 10 ** 9)
        v0s.append(float(pi.values[0]))
        curves[f"γ = {gamma}"] = pi.values
        rows.append({"γ": gamma, "dispatch threshold": "never" if t is None else t, "V(0)": round(v0s[-1], 3), "V(200)": round(float(pi.values[200]), 3)})
    ctx.table("threshold and values versus γ", rows)
    myopic = solvers.value_iteration(build_mdp(HANDOUT.with_(gamma=0.0)))
    ctx.check("γ = 0: never dispatch and V(s) = 2s", bool(np.all(myopic.policy == 0)) and np.allclose(myopic.values, 2.0 * np.arange(201)))
    ctx.check("the threshold never increases with γ", all(b <= a for a, b in zip(thresholds, thresholds[1:])), f"{rows}")
    ctx.check("V(0) increases with γ", all(b > a for a, b in zip(v0s, v0s[1:])))
    ctx.metric("threshold at γ = 0.95", thresholds[gammas.index(0.95)])
    fig, ax = new_figure("Optimal values for different discount factors", "Customers waiting", "Expected discounted cost")
    for color, (label, values) in zip(SERIES, curves.items()):
        ax.plot(np.arange(201), values, color=color, label=label, linewidth=1.8)
    ax.set_yscale("symlog")
    ax.legend(ncol=2)
    ctx.figure(fig, "discount_sweep")


@validation_test(
    section="dp",
    title="Cost parameters move the dispatch threshold the right way",
    what="Sweeps the dispatch cost c_f, the holding cost c_h and the capacity K one at a time and records the "
         "optimal dispatch threshold; also solves the degenerate cases c_f = 0, c_h = 0 and K = 0.",
    how="A dearer shuttle must not be dispatched earlier (threshold non-decreasing in c_f); dearer waiting must not "
         "delay dispatch (threshold non-increasing in c_h). Free shuttles are dispatched as soon as anyone waits, "
         "free waiting or a useless (K = 0) shuttle means never dispatching.",
)
def parameter_sweep(ctx):
    sweeps = {
        "dispatch cost c_f": [(v, threshold_of(HANDOUT.with_(dispatch_cost=v))) for v in (10, 25, 50, 100, 200, 400, 800)],
        "holding cost c_h": [(v, threshold_of(HANDOUT.with_(holding_cost=v))) for v in (0.25, 0.5, 1, 2, 4, 8, 16)],
        "capacity K": [(v, threshold_of(HANDOUT.with_(capacity=v))) for v in (1, 5, 10, 15, 20, 30, 60)],
    }
    for name, pairs in sweeps.items():
        ctx.table(f"threshold versus {name}", [{name: v, "dispatch threshold": "never" if t is None else t} for v, t in pairs])
    big = 10 ** 9
    cf = [t if t is not None else big for _, t in sweeps["dispatch cost c_f"]]
    ch = [t if t is not None else big for _, t in sweeps["holding cost c_h"]]
    ctx.check("threshold is non-decreasing in c_f", all(a <= b for a, b in zip(cf, cf[1:])), str(cf))
    ctx.check("threshold is non-increasing in c_h", all(a >= b for a, b in zip(ch, ch[1:])), str(ch))
    free = solvers.policy_iteration(build_mdp(HANDOUT.with_(dispatch_cost=0.0))).policy
    ctx.check("c_f = 0: dispatch whenever anyone is waiting", bool(np.all(free[1:] == 1)))
    ctx.check("c_h = 0: never dispatch", threshold_of(HANDOUT.with_(holding_cost=0.0)) is None)
    ctx.check("K = 0: never dispatch", threshold_of(HANDOUT.with_(capacity=0)) is None)
    ctx.metric("threshold range over the c_f sweep", f"{min(cf)}–{max(cf)}")
    fig, ax = new_figure("Dispatch threshold versus each cost parameter", "Parameter value (log scale)", "Dispatch threshold (customers)")
    for color, (name, pairs) in zip(SERIES, sweeps.items()):
        xs = [v for v, t in pairs if t is not None]
        ys = [t for _, t in pairs if t is not None]
        ax.plot(xs, ys, marker="o", color=color, label=name, linewidth=2)
    ax.set_xscale("log")
    ax.legend()
    ctx.figure(fig, "parameter_sweep")


@validation_test(
    section="dp",
    title="Changing the structure of the problem",
    what="Solves variants of the model: (a) up to two shuttles per period; (b) holding cost charged at the end of "
         "the period instead of right after the dispatch; (c) holding charged before the dispatch; (d) a station "
         "cap of 400 instead of 200; (e) deterministic arrivals (always 3).",
    how="(a) More actions can only lower the optimal cost, and the second shuttle is only used for long queues. "
         "(b) Charging c_h·(left + A) instead of c_h·left adds the constant c_h·E[A] per period, so the policy must "
         "be identical and every value must shift by exactly c_h·E[A]/(1 − γ) = 120. (c) Charging everyone present "
         "costs more, so values rise. (d) Under the optimal policy the queue never approaches 200, so the cap is "
         "irrelevant: values must agree to round-off. (e) With deterministic arrivals a single simulated "
         "trajectory has no noise, so its discounted cost must equal V(0) to within γ^horizon.",
)
def structure_changes(ctx):
    mdp, _, _, base = solve_all(HANDOUT)
    two = solvers.policy_iteration(build_mdp(HANDOUT.with_(max_dispatch=2)))
    first_double = int(np.flatnonzero(two.policy == 2)[0]) if np.any(two.policy == 2) else None
    ctx.check("(a) allowing two shuttles never increases the cost", bool(np.all(two.values <= base.values + 1e-9)))
    ctx.check("(a) two shuttles are only sent for long queues", first_double is not None and first_double > solvers.policy_threshold(base.policy),
              f"second shuttle from s = {first_double}")
    ctx.metric("(a) largest saving from a second shuttle", float(np.max(base.values - two.values)))
    end = solvers.policy_iteration(build_mdp(HANDOUT.with_(holding_charged="end_of_period")))
    shift = HANDOUT.holding_cost * HANDOUT.mean_arrivals / (1 - HANDOUT.gamma)
    ctx.check("(b) end-of-period holding: same policy", np.array_equal(end.policy, base.policy))
    ctx.check_close("(b) end-of-period holding: values shift by exactly c_h·E[A]/(1 − γ) = 120", end.values - base.values, shift, 1e-6)
    before = solvers.policy_iteration(build_mdp(HANDOUT.with_(holding_charged="before_dispatch")))
    ctx.check("(c) charging everyone present costs more", bool(np.all(before.values >= base.values - 1e-9)))
    ctx.metric("(c) threshold when holding is charged before dispatch", solvers.policy_threshold(before.policy))
    big = solvers.policy_iteration(build_mdp(HANDOUT.with_(max_customers=400)))
    ctx.check_close("(d) a cap of 400 gives the same values for s ≤ 200", big.values[:201], base.values, 1e-6)
    ctx.check("(d) same threshold with the larger cap", solvers.policy_threshold(big.policy) == solvers.policy_threshold(base.policy))
    det_params = HANDOUT.with_(arrivals=(3,))
    det_mdp = build_mdp(det_params)
    det = solvers.policy_iteration(det_mdp)
    sim = solvers.simulate_policy(det_mdp, det.policy, start=0, horizon=800, episodes=2, seed=0)
    ctx.check("(e) deterministic arrivals: the simulated trajectory has no variance", sim["costs"][0] == sim["costs"][1])
    ctx.check_close("(e) deterministic arrivals: simulated discounted cost = V(0)", sim["mean"], det.values[0], sim["truncation_bound"] + 1e-9)
    ctx.table("variants", [
        {"variant": "handout", "threshold": solvers.policy_threshold(base.policy), "V(0)": round(float(base.values[0]), 3), "V(200)": round(float(base.values[200]), 3)},
        {"variant": "(a) up to 2 shuttles", "threshold": solvers.policy_threshold(two.policy >= 1), "V(0)": round(float(two.values[0]), 3), "V(200)": round(float(two.values[200]), 3)},
        {"variant": "(b) holding at end of period", "threshold": solvers.policy_threshold(end.policy), "V(0)": round(float(end.values[0]), 3), "V(200)": round(float(end.values[200]), 3)},
        {"variant": "(c) holding before dispatch", "threshold": solvers.policy_threshold(before.policy), "V(0)": round(float(before.values[0]), 3), "V(200)": round(float(before.values[200]), 3)},
        {"variant": "(d) cap 400", "threshold": solvers.policy_threshold(big.policy), "V(0)": round(float(big.values[0]), 3), "V(200)": round(float(big.values[200]), 3)},
        {"variant": "(e) arrivals always 3", "threshold": solvers.policy_threshold(det.policy), "V(0)": round(float(det.values[0]), 3), "V(200)": round(float(det.values[200]), 3)},
    ])


@validation_test(
    section="dp",
    title="Monte Carlo rollouts reproduce the value function",
    what="Simulates the optimal policy from several starting queues (5000 trajectories of 600 periods each) and "
         "compares the average discounted cost with V*(s); also simulates two deliberately worse threshold "
         "policies (dispatch at 5 and at 30).",
    how="V*(s) is defined as the expected discounted cost of the optimal policy, so the sample mean must land "
         "within three standard errors (plus the negligible truncation error γ^600·max cost/(1 − γ)). Any other "
         "policy must cost at least as much in every state, both in simulation and when evaluated exactly.",
)
def simulation(ctx):
    mdp = build_mdp(HANDOUT)
    pi = solvers.policy_iteration(mdp)
    rows = []
    for start in (0, 13, 50, 200):
        sim = solvers.simulate_policy(mdp, pi.policy, start=start, horizon=600, episodes=5000, seed=start)
        gap = abs(sim["mean"] - pi.values[start])
        rows.append({"start s": start, "V*(s)": round(float(pi.values[start]), 3), "simulated mean": round(sim["mean"], 3),
                     "standard error": round(sim["stderr"], 3), "|difference| / s.e.": round(gap / sim["stderr"], 2)})
        ctx.check(f"start {start}: simulation within 3 s.e. of V*", gap <= 3 * sim["stderr"] + sim["truncation_bound"], f"gap {gap:.3f}, s.e. {sim['stderr']:.3f}")
    ctx.table("optimal policy: simulation versus value function", rows)
    worse_rows = []
    for threshold in (5, 30):
        policy = (np.arange(201) >= threshold).astype(int)
        values = solvers.evaluate_policy(mdp, policy)
        sim = solvers.simulate_policy(mdp, policy, start=0, horizon=600, episodes=5000, seed=threshold)
        worse_rows.append({"policy": f"dispatch at ≥ {threshold}", "V^π(0) exact": round(float(values[0]), 3), "simulated": round(sim["mean"], 3), "V*(0)": round(float(pi.values[0]), 3)})
        ctx.check(f"threshold-{threshold} policy is never better than the optimal one (exact evaluation)", bool(np.all(values >= pi.values - 1e-9)))
        ctx.check(f"threshold-{threshold} policy: simulation agrees with its exact value", abs(sim["mean"] - values[0]) <= 3 * sim["stderr"] + sim["truncation_bound"])
    ctx.table("suboptimal policies", worse_rows)


@validation_test(
    section="dp",
    title="Policy iteration improves monotonically and ends at the greedy fixed point",
    what="Records every policy visited by policy iteration (starting from 'never dispatch' and, separately, from "
         "'always dispatch'), evaluates each exactly and compares consecutive value functions.",
    how="The policy-improvement theorem (Lean: policy_improvement) says the greedy policy for V^π is at least as "
         "good as π in every state, so the sequence of value functions must be pointwise non-increasing, strictly "
         "better somewhere at each step, and both starting points must reach the same policy, which must be greedy "
         "with respect to its own values.",
)
def policy_iteration_improvement(ctx):
    mdp = build_mdp(HANDOUT)
    pi = solvers.policy_iteration(mdp)
    policies = [np.array(p) for p in pi.info["policies"]]
    values = [solvers.evaluate_policy(mdp, p) for p in policies]
    rows = []
    monotone = True
    for k, (p, v) in enumerate(zip(policies, values)):
        if k:
            monotone &= bool(np.all(v <= values[k - 1] + 1e-9)) and bool(np.any(v < values[k - 1] - 1e-9))
        rows.append({"k": k, "threshold": solvers.policy_threshold(p), "dispatch states": int(p.sum()),
                     "V^π_k(0)": round(float(v[0]), 3), "V^π_k(200)": round(float(v[200]), 3),
                     "max improvement": None if k == 0 else round(float(np.max(values[k - 1] - v)), 3)})
    ctx.table("policies visited (from 'never dispatch')", rows)
    ctx.check("each improvement step lowers the cost somewhere and raises it nowhere", monotone)
    ctx.check("the final policy is greedy for its own values", np.array_equal(solvers.greedy_policy(mdp, values[-1], current=policies[-1], tie_tol=1e-9), policies[-1]))
    other = solvers.policy_iteration(mdp, initial_policy=np.ones(201, dtype=int))
    ctx.check("starting from 'always dispatch' reaches the same policy", np.array_equal(other.policy, pi.policy), f"{other.iterations} steps")
    ctx.metric("improvement steps from 'never dispatch'", pi.iterations)
    ctx.metric("improvement steps from 'always dispatch'", other.iterations)


@validation_test(
    section="dp",
    title="The CUDA kernels reproduce the solvers",
    what="Runs the GPU versions of Section 2 (rl/cuda/dp.py: the state grid initialised on the device, one "
         "thread per state for the Bellman sweeps with the value function in shared memory, greedy improvement, "
         "and one thread per trajectory for Monte Carlo) and compares them with the numpy solvers. On a machine "
         "with an NVIDIA GPU this is the full 201-state problem and 100000 simulated trajectories; without one, "
         "the same kernels run in Numba's CUDA simulator on a 31-state instance (slow, so the sizes are small).",
    how="The kernels implement the same equations with a different execution order (per-thread sparse sums, "
         "block-wide max reductions), so values must agree to round-off, policies and iteration counts must "
         "coincide, and the per-trajectory simulation must land within three standard errors of V*. The "
         "recorded kernel times are real only on a GPU.",
)
def cuda_kernels(ctx):
    try:
        from rl.cuda import backend, dp as cuda_dp
    except ImportError as e:  # numba not installed
        ctx.check("numba is installed (pip install -r requirements.txt)", False, str(e))
        return
    on_gpu = not backend.SIMULATOR
    params = HANDOUT if on_gpu else HANDOUT.with_(max_customers=30, gamma=0.8)
    ctx.note(f"backend: {backend.backend_name()} on {backend.device_summary()}; instance: {params.describe()}")
    cost, succ_idx, succ_p = cuda_dp.shuttle_tables_gpu(params)
    reference = build_mdp(params)
    device_mdp = cuda_dp.mdp_from_tables(cost, succ_idx, succ_p, params.gamma, reference.action_labels)
    ctx.check("the state grid built on the device equals build_mdp (costs and transitions)",
              np.allclose(device_mdp.costs, reference.costs) and np.allclose(device_mdp.transitions, reference.transitions))
    cpu = {"enumeration": solvers.enumeration(reference, 20 if not on_gpu else params.horizon),
           "value iteration": solvers.value_iteration(reference, tol=1e-9), "policy iteration": solvers.policy_iteration(reference)}
    gpu = {"enumeration": cuda_dp.enumeration_gpu(device_mdp, 20 if not on_gpu else params.horizon),
           "value iteration": cuda_dp.value_iteration_gpu(device_mdp, tol=1e-9),
           "policy iteration": cuda_dp.policy_iteration_gpu(device_mdp, eval_tol=1e-13 if on_gpu else 1e-10)}
    rows = []
    for name in cpu:
        diff = float(np.max(np.abs(cpu[name].values - gpu[name].values)))
        rows.append({"method": name, "max |V_cpu − V_gpu|": f"{diff:.2e}", "iterations cpu / gpu": f"{cpu[name].iterations} / {gpu[name].iterations}",
                     "same policy": bool(np.array_equal(cpu[name].policy, gpu[name].policy)), "kernel": gpu[name].info["kernel"],
                     "kernel time (s)": gpu[name].info["time_s"]})
        ctx.check(f"{name}: values agree to 1e-7 and the policies coincide", diff < 1e-7 and np.array_equal(cpu[name].policy, gpu[name].policy), f"max diff {diff:.2e}")
        ctx.metric(f"{name} kernel time (s)", gpu[name].info["time_s"])
    ctx.check("value iteration: same number of sweeps (±1)", abs(cpu["value iteration"].iterations - gpu["value iteration"].iterations) <= 1)
    ctx.table("CPU versus device solvers", rows)
    episodes = 100_000 if on_gpu else 128
    sim = cuda_dp.simulate_policy_gpu(device_mdp, gpu["policy iteration"].policy, start=0, horizon=400 if on_gpu else 60, episodes=episodes, seed=0,
                                      threads=256 if on_gpu else 32)
    gap = abs(sim["mean"] - cpu["policy iteration"].values[0])
    ctx.check(f"{episodes} device trajectories reproduce V*(0) within 3 s.e.", gap <= 3 * sim["stderr"] + sim["truncation_bound"], f"gap {gap:.4f}, s.e. {sim['stderr']:.4f}")
    ctx.metric("simulation kernel time (s)", sim["time_s"])
    ctx.metric("simulation grid (blocks × threads)", f"{sim['grid']['blocks']} × {sim['grid']['threads']}")
