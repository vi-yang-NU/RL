"""Unit tests for the explorer data (rl/dp/explore.py) and its replay engine (rl/dp/explore_engine.js).

The page shows what the real solvers computed; these tests pin that down: the recorded algorithm steps are
the solvers' own iterates, the finite-horizon policies are exact, the embedded MDP tables reproduce
`build_mdp`, and (when Node is installed) the JavaScript replay only ever makes moves the model allows.
"""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl.dp import solvers
from rl.dp.explore import ENGINE, explore, override_arguments, render_html, sparse_successors, thin_history, write_explorer
from rl.params import apply_overrides, parse_overrides
from rl.dp.shuttle import ShuttleParams, build_mdp

SMALL = ShuttleParams(max_customers=30, capacity=5, horizon=12)


class ExploreDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = SMALL
        cls.mdp = build_mdp(cls.params)
        cls.data = explore(cls.params, tol=1e-8, steps=10)

    def test_optimal_block_is_policy_iterations_solution(self):
        pi = solvers.policy_iteration(self.mdp)
        np.testing.assert_allclose(self.data["optimal"]["values"], pi.values, atol=1e-10)
        np.testing.assert_array_equal(self.data["optimal"]["policy"], pi.policy)
        np.testing.assert_allclose(self.data["optimal"]["q"], solvers.q_values(self.mdp, pi.values), atol=1e-10)
        self.assertEqual(self.data["optimal"]["source"], "policy_iteration")
        self.assertEqual(self.data["optimal"]["threshold"], solvers.policy_threshold(pi.policy))

    def test_value_iteration_steps_are_the_solvers_iterates(self):
        block = self.data["methods"]["value_iteration"]
        steps = block["steps"]
        self.assertEqual(len(steps), 11)
        V = np.zeros(self.mdp.n_states)
        for k, step in enumerate(steps):
            np.testing.assert_allclose(step["values"], V, atol=1e-12, err_msg=f"sweep {k}")
            np.testing.assert_array_equal(step["policy"], solvers.greedy_policy(self.mdp, V))
            if k:
                self.assertAlmostEqual(step["change"], float(np.max(np.abs(V - previous))), places=12)
            previous, V = V, self.mdp.bellman(V)
        full = solvers.value_iteration(self.mdp, tol=1e-8)
        self.assertEqual(block["iterations"], full.iterations)
        np.testing.assert_allclose(block["values"], full.values, atol=1e-12)
        self.assertEqual(block["history"][0], [0, full.history["‖V_{n+1} − V_n‖∞"][0]])

    def test_enumeration_steps_and_first_periods_are_exact(self):
        block = self.data["methods"]["enumeration"]
        T = self.params.horizon
        self.assertEqual(len(block["steps"]), 11)
        self.assertIsNone(block["steps"][0]["policy"])
        for step in block["steps"][1:]:
            k = step["k"]
            ref = solvers.enumeration(self.mdp, k)
            np.testing.assert_allclose(step["values"], ref.values, atol=1e-12, err_msg=f"{k} periods to go")
            np.testing.assert_array_equal(step["policy"], ref.policy)
        self.assertEqual([fp["periods_to_go"] for fp in block["first_periods"]], list(range(T, T - 10, -1)))
        for fp in block["first_periods"]:
            ref = solvers.enumeration(self.mdp, fp["periods_to_go"])
            np.testing.assert_array_equal(fp["policy"], ref.policy, err_msg=f"{fp['periods_to_go']} periods to go")
            np.testing.assert_allclose(fp["values"], ref.values, atol=1e-10)

    def test_policy_iteration_steps_follow_the_visited_policies(self):
        block = self.data["methods"]["policy_iteration"]
        pi = solvers.policy_iteration(self.mdp)
        self.assertEqual(len(block["steps"]), len(pi.info["policies"]))
        self.assertEqual(block["steps"][0]["policy"], [0] * self.mdp.n_states)
        self.assertIsNone(block["steps"][0]["changed"])
        for step, policy in zip(block["steps"], pi.info["policies"]):
            self.assertEqual(step["policy"], policy)
            np.testing.assert_allclose(step["values"], solvers.evaluate_policy(self.mdp, np.array(policy)), atol=1e-10)
        changed = [int(c) for c in pi.history["states whose action changed"] if c > 0]
        self.assertEqual([s["changed"] for s in block["steps"][1:]], changed)
        self.assertEqual(block["history"], [[i, c] for i, c in enumerate(pi.history["states whose action changed"])])

    def test_embedded_mdp_reproduces_build_mdp(self):
        m = self.data["mdp"]
        self.assertEqual((m["n_states"], m["n_actions"]), (self.mdp.n_states, self.mdp.n_actions))
        np.testing.assert_allclose(m["cost"], self.mdp.costs)
        for s in range(self.mdp.n_states):
            for a in range(self.mdp.n_actions):
                dense = np.zeros(self.mdp.n_states)
                for j, p in m["succ"][s][a]:
                    dense[j] += p
                np.testing.assert_allclose(dense, self.mdp.transitions[a, s], atol=1e-15)
                self.assertAlmostEqual(sum(p for _, p in m["succ"][s][a]), 1.0, places=12)
                self.assertLessEqual(len(m["succ"][s][a]), len(self.params.arrivals))

    def test_json_serialisable_and_rendered(self):
        text = json.dumps(self.data, allow_nan=False)
        self.assertIn('"problem"', text)
        html = render_html(self.data, theme="dark")
        self.assertIn('data-theme="dark"', html)
        self.assertIn("ExploreEngine", html)
        self.assertIn('"n_states":31', html)
        self.assertNotIn("__DATA__", html)
        self.assertNotIn("__ENGINE__", html)
        self.assertNotIn("</script>", html.split("const DATA = ")[1].split("\n")[0])
        self.assertNotIn("data-theme", render_html(self.data).split("<head>")[0])


class ExploreEdgeCaseTests(unittest.TestCase):
    def test_gamma_one_falls_back_to_enumeration(self):
        data = explore(ShuttleParams(gamma=1.0, horizon=8, max_customers=20), tol=1e-6)
        self.assertEqual(data["optimal"]["source"], "enumeration")
        self.assertIn("error", data["methods"]["value_iteration"])
        self.assertIn("error", data["methods"]["policy_iteration"])
        self.assertNotIn("steps", data["methods"]["value_iteration"])
        ref = solvers.enumeration(build_mdp(ShuttleParams(gamma=1.0, horizon=8, max_customers=20)), 8)
        np.testing.assert_allclose(data["optimal"]["values"], ref.values)
        json.dumps(data, allow_nan=False)

    def test_short_and_zero_horizons(self):
        data = explore(ShuttleParams(horizon=3, max_customers=10), steps=10)
        block = data["methods"]["enumeration"]
        self.assertEqual([s["k"] for s in block["steps"]], [0, 1, 2, 3])
        self.assertEqual([fp["periods_to_go"] for fp in block["first_periods"]], [3, 2, 1])
        self.assertEqual(len(data["methods"]["value_iteration"]["steps"]), 11)
        data = explore(ShuttleParams(horizon=0, max_customers=10))
        block = data["methods"]["enumeration"]
        self.assertEqual(len(block["steps"]), 1)
        self.assertEqual(block["first_periods"], [])
        self.assertIsNone(block["policy"])
        self.assertEqual(block["values"], [0.0] * 11)

    def test_gamma_zero_converges_in_one_sweep(self):
        data = explore(ShuttleParams(gamma=0.0, max_customers=25), tol=1e-8)
        vi = data["methods"]["value_iteration"]
        self.assertLessEqual(vi["iterations"], 2)
        self.assertEqual(len(vi["steps"]), vi["iterations"] + 1)
        self.assertIsNone(data["optimal"]["threshold"])  # dispatching never pays with no future
        self.assertTrue(data["methods"]["policy_iteration"]["converged"])

    def test_more_actions_and_other_holding_rules(self):
        data = explore(ShuttleParams(max_dispatch=3, max_customers=40, capacity=10, holding_charged="end_of_period"))
        self.assertEqual(data["mdp"]["n_actions"], 4)
        self.assertIsNone(data["optimal"]["threshold"])  # only defined for two actions
        self.assertEqual(len(data["mdp"]["action_labels"]), 4)
        self.assertTrue(all(len(step["policy"]) == 41 for step in data["methods"]["policy_iteration"]["steps"]))

    def test_argument_validation(self):
        with self.assertRaises(ValueError):
            explore(ShuttleParams(max_customers=10), steps=0)

    def test_thin_history_keeps_the_first_points_and_the_last(self):
        values = list(np.linspace(1.0, 0.0, 5000))
        thinned = thin_history(values, keep=10, cap=100)
        self.assertEqual([i for i, _ in thinned[:10]], list(range(10)))
        self.assertEqual(thinned[-1][0], 4999)
        self.assertLessEqual(len(thinned), 100)
        self.assertEqual(thin_history([3.0, 2.0]), [[0, 3.0], [1, 2.0]])

    def test_sparse_successors_merge_capped_arrivals(self):
        mdp = build_mdp(ShuttleParams(max_customers=3, arrivals=(1, 2, 3)))
        succ = sparse_successors(mdp)
        self.assertEqual(succ[3][0], [[3, 1.0]])  # full station, no dispatch: everyone is turned away
        self.assertEqual(succ[2][0], [[3, 1.0]])
        self.assertEqual([j for j, _ in succ[0][0]], [1, 2, 3])

    def test_write_explorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_explorer(Path(tmp) / "sub" / "explore.html", ShuttleParams(max_customers=12), theme="light")
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("<!doctype html>"))
            self.assertIn('"max_customers":12', text)


NODE = shutil.which("node")

NODE_SCRIPT = r"""
const E = require(process.argv[2]).ExploreEngine;
const data = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const mdp = { gamma: data.mdp.gamma, cost: data.mdp.cost, succ: data.mdp.succ };
const out = { walks: [], draws: [] };
for (const seed of [0, 1, 7, 12345]) {
  const w = new E.Walk(mdp, { start: 0, seed: seed, maxSteps: 10, policy: (t, s) => data.optimal.policy[s] });
  for (let t = 0; t < 10; t++) w.step(t === 3 ? 1 : undefined);
  const again = new E.Walk(mdp, { start: 0, seed: seed, maxSteps: 10, policy: (t, s) => data.optimal.policy[s] });
  for (let t = 0; t < 10; t++) again.step(t === 3 ? 1 : undefined);
  out.walks.push({ seed: seed, rows: w.rows, same: JSON.stringify(w.rows) === JSON.stringify(again.rows), total: w.total, finished: w.finished });
}
// the draws for (seed, t) do not depend on earlier actions, and the sampler follows the probabilities
const counts = {};
for (let i = 0; i < 20000; i++) { const s = E.sampleNext(data.mdp.succ[5][0], E.drawFor(99, i)); counts[s] = (counts[s] || 0) + 1; }
out.counts = counts;
out.horizon = (() => { const w = new E.Walk(mdp, { start: 0, seed: 1, maxSteps: 10, horizon: 4, policy: () => 0 }); while (w.step()) {} return [w.t, w.finished, w.limit]; })();
out.back = (() => { const w = new E.Walk(mdp, { start: 2, seed: 1, maxSteps: 10, policy: () => 0 }); w.step(); w.step(); w.back(); return [w.t, w.state, w.total]; })();
console.log(JSON.stringify(out));
"""


@unittest.skipUnless(NODE, "node is not installed")
class EngineUnderNodeTests(unittest.TestCase):
    """Runs rl/dp/explore_engine.js on a small instance and checks every move against the Python model."""

    @classmethod
    def setUpClass(cls):
        cls.params = ShuttleParams(max_customers=25, capacity=6)
        cls.mdp = build_mdp(cls.params)
        cls.data = explore(cls.params)
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data.json"
            data_path.write_text(json.dumps(cls.data), encoding="utf-8")
            script = Path(tmp) / "run.js"
            script.write_text(NODE_SCRIPT, encoding="utf-8")
            proc = subprocess.run([NODE, str(script), str(ENGINE), str(data_path)], capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)
        cls.out = json.loads(proc.stdout)

    def test_every_replayed_move_is_allowed_by_the_model(self):
        P, C, gamma = self.mdp.transitions, self.mdp.costs, self.mdp.gamma
        for walk in self.out["walks"]:
            self.assertTrue(walk["same"], "the same seed must reproduce the same rows")
            self.assertTrue(walk["finished"])
            s, total = 0, 0.0
            for row in walk["rows"]:
                self.assertEqual(row["s"], s)
                self.assertEqual(row["suggested"], self.data["optimal"]["policy"][s])
                self.assertEqual(row["overridden"], row["t"] == 3 and row["a"] != row["suggested"])
                self.assertGreater(P[row["a"], s, row["next"]], 0.0, f"seed {walk['seed']}, t={row['t']}: impossible transition")
                self.assertAlmostEqual(row["cost"], C[s, row["a"]], places=12)
                self.assertAlmostEqual(row["discount"], gamma ** row["t"], places=12)
                total += gamma ** row["t"] * C[s, row["a"]]
                self.assertAlmostEqual(row["cumulative"], total, places=9)
                s = row["next"]
            self.assertAlmostEqual(walk["total"], total, places=9)

    def test_sampler_follows_the_transition_probabilities(self):
        counts = {int(k): v for k, v in self.out["counts"].items()}
        n = sum(counts.values())
        for j, p in enumerate(self.mdp.transitions[0, 5]):
            frequency = counts.get(j, 0) / n
            self.assertAlmostEqual(frequency, p, delta=0.015, msg=f"P(s'={j} | s=5, wait)")

    def test_horizon_and_back(self):
        self.assertEqual(self.out["horizon"], [4, True, 4])
        t, state, total = self.out["back"]
        self.assertEqual(t, 1)
        self.assertAlmostEqual(total, self.mdp.costs[2, 0])
        self.assertGreater(self.mdp.transitions[0, 2, state], 0.0)


class OverrideArgumentTests(unittest.TestCase):
    """`override_arguments` writes the `--set` flags the explorer's "Run this case" button puts into the Run panel."""

    def test_handout_needs_no_arguments(self):
        self.assertEqual(override_arguments(ShuttleParams()), [])
        self.assertEqual(override_arguments(ShuttleParams(), tol=1e-8), [])

    def test_arguments_round_trip_through_the_command_line_parser(self):
        cases = [SMALL, ShuttleParams(arrivals=(1, 5), arrival_probs=(0.8, 0.2)), ShuttleParams(holding_charged="before_dispatch"),
                 ShuttleParams(gamma=1.0, horizon=20), ShuttleParams(max_dispatch=2, dispatch_cost=10.0, holding_cost=0.0)]
        for params in cases:
            args = override_arguments(params)
            self.assertTrue(args and all(a == "--set" for a in args[::2]), args)
            self.assertEqual(shlex.split(shlex.join(args)), args, "survives the Run panel's text box")
            self.assertEqual(apply_overrides(ShuttleParams(), parse_overrides(args[1::2])), params)

    def test_strings_are_bare_and_tol_is_added(self):
        args = override_arguments(ShuttleParams(holding_charged="before_dispatch"), tol=1e-10)
        self.assertEqual(args, ["--set", "holding_charged=before_dispatch", "--tol", "1e-10"])
        self.assertEqual(override_arguments(ShuttleParams(arrivals=(3,))), ["--set", "arrivals=[3]"])


if __name__ == "__main__":
    unittest.main()
