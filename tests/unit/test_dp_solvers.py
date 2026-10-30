"""Unit tests for enumeration, value iteration and policy iteration (rl/dp/solvers.py), including edge cases.

Several checks mirror theorems in lean/RLVerify/Shuttle.lean (named in the test docstrings).
"""
from __future__ import annotations

import unittest
import warnings

import numpy as np

from rl.dp import solvers
from rl.dp.shuttle import ShuttleParams, TabularMDP, build_mdp


def two_state_mdp(gamma: float) -> TabularMDP:
    """State 0: stay (cost 1) or switch to state 1 (cost 3). State 1: stay or switch, both free.

    V*(1) = 0 and V*(0) = min(1 / (1 - gamma), 3), so the optimal action in state 0 depends on gamma.
    """
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = P[0, 1, 1] = 1.0
    P[1, 0, 1] = P[1, 1, 0] = 1.0
    C = np.array([[1.0, 3.0], [0.0, 0.0]])
    return TabularMDP(P, C, gamma, np.arange(2), ("stay", "switch"), name="two-state")


class TwoStateTests(unittest.TestCase):
    def test_enumeration_matches_hand_computation(self):
        mdp = two_state_mdp(0.9)
        expected = {0: [0, 0], 1: [1, 0], 2: [1.9, 0], 3: [2.71, 0], 4: [3.0, 0], 10: [3.0, 0]}
        for horizon, values in expected.items():
            result = solvers.enumeration(mdp, horizon)
            np.testing.assert_allclose(result.values, values, atol=1e-12, err_msg=f"horizon {horizon}")
            self.assertEqual(result.iterations, horizon)
        self.assertIsNone(solvers.enumeration(mdp, 0).policy)
        np.testing.assert_array_equal(solvers.enumeration(mdp, 4).policy, [1, 0])
        np.testing.assert_array_equal(solvers.enumeration(mdp, 3).policy, [0, 0])

    def test_value_and_policy_iteration_find_the_closed_form(self):
        for gamma, value, action in ((0.9, 3.0, 1), (0.5, 2.0, 0)):
            mdp = two_state_mdp(gamma)
            vi = solvers.value_iteration(mdp, tol=1e-12)
            pi = solvers.policy_iteration(mdp)
            np.testing.assert_allclose(vi.values, [value, 0.0], atol=1e-10)
            np.testing.assert_allclose(pi.values, [value, 0.0], atol=1e-10)
            self.assertEqual(vi.policy[0], action)
            self.assertEqual(pi.policy[0], action)
            self.assertTrue(vi.info["converged"])
            self.assertTrue(pi.info["converged"])
            self.assertLessEqual(pi.iterations, 3)

    def test_policy_evaluation_closed_form(self):
        mdp = two_state_mdp(0.9)
        always_stay = np.array([0, 0])
        np.testing.assert_allclose(solvers.evaluate_policy(mdp, always_stay), [10.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(solvers.evaluate_policy(mdp, always_stay, method="iterative", tol=1e-13), [10.0, 0.0], atol=1e-9)
        with self.assertRaises(ValueError):
            solvers.evaluate_policy(mdp, always_stay, method="magic")
        with self.assertRaises(ValueError):
            solvers.evaluate_policy(mdp, np.array([0, 5]))

    def test_enumeration_with_terminal_values_and_trace(self):
        mdp = two_state_mdp(0.9)
        result = solvers.enumeration(mdp, 2, terminal_values=[0.0, 100.0], record=2)
        # one step to go: state 0 min(1 + 0.9*0, 3 + 0.9*100) = 1; state 1 min(0 + 90, 0 + 0) = 0 (switch)
        np.testing.assert_allclose(result.info["values_by_steps_to_go"][0], [0.0, 100.0])
        np.testing.assert_allclose(result.info["values_by_steps_to_go"][1], [1.0, 0.0])
        np.testing.assert_allclose(result.values, [1.9, 0.0])
        self.assertEqual(len(result.info["policies_by_steps_to_go"]), 2)
        with self.assertRaises(ValueError):
            solvers.enumeration(mdp, 2, terminal_values=[1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            solvers.enumeration(mdp, -1)


class ShuttleSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = ShuttleParams()
        cls.mdp = build_mdp(cls.params)
        cls.enum = solvers.enumeration(cls.mdp, cls.params.horizon)
        cls.vi = solvers.value_iteration(cls.mdp, tol=1e-9)
        cls.pi = solvers.policy_iteration(cls.mdp)

    def test_three_methods_agree(self):
        self.assertLess(np.max(np.abs(self.enum.values - self.pi.values)), 1e-6)
        self.assertLess(np.max(np.abs(self.vi.values - self.pi.values)), 1e-6)
        np.testing.assert_array_equal(self.vi.policy, self.pi.policy)
        np.testing.assert_array_equal(self.enum.policy, self.pi.policy)

    def test_value_iteration_error_bound_holds(self):
        """Lean `vi_error_bound`: ||V_n - V*|| <= gamma ||V_n - V_{n-1}|| / (1 - gamma); V* here is V^pi of PI."""
        self.assertTrue(self.vi.info["converged"])
        self.assertLess(self.vi.info["final_change"], 1e-9)
        self.assertLessEqual(np.max(np.abs(self.vi.values - self.pi.values)), self.vi.info["error_bound"] + 1e-12)

    def test_bellman_residuals(self):
        self.assertLess(solvers.bellman_residual(self.mdp, self.pi.values), 1e-9)
        self.assertLess(self.vi.info["bellman_residual"], 1e-8)

    def test_contraction_ratio_never_exceeds_gamma(self):
        """Lean `bellman_contraction`: successive changes shrink by at least the factor gamma."""
        # Ratios are compared with a tolerance of 1e-6: below the noise floor (about 1e-4 here, residuals
        # that are 1e8 machine epsilons of the largest value) rounding in V dominates the residuals.
        self.assertLessEqual(self.vi.info["max_contraction_ratio"], self.params.gamma + 1e-6)
        residuals = self.vi.history["‖V_{n+1} − V_n‖∞"]
        self.assertEqual(len(residuals), self.vi.iterations)
        floor = self.vi.info["contraction_noise_floor"]
        self.assertTrue(all(b <= a * (self.params.gamma + 1e-6) for a, b in zip(residuals, residuals[1:]) if a > floor))
        self.assertGreater(sum(a > floor for a in residuals), 100)  # the check covers most of the run

    def test_values_are_monotone_in_the_state(self):
        """Lean `finiteValue_mono_state`: more customers waiting never costs less."""
        for result in (self.enum, self.vi, self.pi):
            self.assertTrue(np.all(np.diff(result.values) >= -1e-9))
        self.assertTrue(np.all(self.pi.values >= 0))

    def test_policy_is_a_threshold_policy(self):
        self.assertTrue(solvers.is_monotone(self.pi.policy))
        threshold = solvers.policy_threshold(self.pi.policy)
        self.assertIsNotNone(threshold)
        self.assertEqual(self.pi.info["threshold"], threshold)
        self.assertTrue(0 < threshold <= self.params.capacity)

    def test_policy_iteration_is_stable_and_greedy(self):
        self.assertTrue(self.pi.info["converged"])
        np.testing.assert_array_equal(solvers.greedy_policy(self.mdp, self.pi.values, current=self.pi.policy, tie_tol=1e-9), self.pi.policy)
        self.assertEqual(len(self.pi.info["policies"]), self.pi.iterations)
        self.assertEqual(self.pi.history["states whose action changed"][-1], 0.0)

    def test_policy_evaluation_methods_agree(self):
        iterative = solvers.evaluate_policy(self.mdp, self.pi.policy, method="iterative", tol=1e-13)
        np.testing.assert_allclose(iterative, self.pi.values, atol=1e-8)

    def test_simulation_agrees_with_the_value_function(self):
        sim = solvers.simulate_policy(self.mdp, self.pi.policy, start=10, horizon=600, episodes=4000, seed=1)
        self.assertLess(sim["truncation_bound"], 1e-6)
        self.assertLess(abs(sim["mean"] - self.pi.values[10]), 4 * sim["stderr"] + sim["truncation_bound"])
        self.assertEqual(len(sim["trace"]), 10)
        self.assertEqual(sim["trace"][0]["state"], 10)

    def test_warm_start_and_initial_values(self):
        again = solvers.value_iteration(self.mdp, tol=1e-9, initial_values=self.pi.values)
        self.assertLessEqual(again.iterations, 2)
        pi2 = solvers.policy_iteration(self.mdp, initial_policy=self.pi.policy)
        self.assertEqual(pi2.iterations, 1)
        np.testing.assert_array_equal(pi2.policy, self.pi.policy)


class EdgeCaseTests(unittest.TestCase):
    def test_myopic_agent_never_dispatches(self):
        """gamma = 0: V(s) = min_a c(s, a) = 2s, because 100 + 2 max(s - 15, 0) > 2s for every s."""
        mdp = build_mdp(ShuttleParams(gamma=0.0))
        vi = solvers.value_iteration(mdp, tol=1e-12)
        np.testing.assert_allclose(vi.values, 2.0 * np.arange(201))
        self.assertTrue(np.all(vi.policy == 0))
        self.assertLessEqual(vi.iterations, 2)
        self.assertEqual(vi.info["error_bound"], 0.0)
        pi = solvers.policy_iteration(mdp)
        np.testing.assert_allclose(pi.values, vi.values)

    def test_free_waiting_means_no_dispatch(self):
        mdp = build_mdp(ShuttleParams(holding_cost=0.0))
        pi = solvers.policy_iteration(mdp)
        self.assertTrue(np.all(pi.policy == 0))
        np.testing.assert_allclose(pi.values, 0.0, atol=1e-12)

    def test_free_shuttles_are_always_dispatched(self):
        mdp = build_mdp(ShuttleParams(dispatch_cost=0.0))
        pi = solvers.policy_iteration(mdp)
        self.assertTrue(np.all(pi.policy[1:] == 1))
        vi = solvers.value_iteration(mdp, tol=1e-10)
        self.assertTrue(np.all(vi.policy[1:] == 1))

    def test_useless_shuttle_is_never_dispatched(self):
        pi = solvers.policy_iteration(build_mdp(ShuttleParams(capacity=0)))
        self.assertTrue(np.all(pi.policy == 0))

    def test_horizon_zero_and_one(self):
        mdp = build_mdp(ShuttleParams())
        zero = solvers.enumeration(mdp, 0)
        np.testing.assert_array_equal(zero.values, 0.0)
        self.assertIsNone(zero.policy)
        self.assertEqual(zero.history["max |V_t − V_{t+1}|"], [])
        one = solvers.enumeration(mdp, 1)
        np.testing.assert_allclose(one.values, 2.0 * np.arange(201))
        self.assertTrue(np.all(one.policy == 0))

    def test_undiscounted_problem(self):
        mdp = build_mdp(ShuttleParams(gamma=1.0))
        with self.assertRaises(ValueError):
            solvers.value_iteration(mdp)
        with self.assertRaises(ValueError):
            solvers.policy_iteration(mdp)  # exact evaluation needs gamma < 1
        self.assertEqual(solvers.enumeration(mdp, 3).values.shape, (201,))  # finite horizon is fine

    def test_single_state_problem(self):
        mdp = build_mdp(ShuttleParams(max_customers=0))
        np.testing.assert_allclose(solvers.value_iteration(mdp).values, [0.0])
        np.testing.assert_array_equal(solvers.policy_iteration(mdp).policy, [0])

    def test_iteration_caps_warn_instead_of_failing(self):
        mdp = build_mdp(ShuttleParams())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vi = solvers.value_iteration(mdp, tol=1e-12, max_iter=5)
            pi = solvers.policy_iteration(mdp, max_iter=1)
        self.assertFalse(vi.info["converged"])
        self.assertEqual(vi.iterations, 5)
        self.assertFalse(pi.info["converged"])
        self.assertEqual(len(caught), 2)
        with self.assertRaises(ValueError):
            solvers.value_iteration(mdp, tol=-1.0)

    def test_deterministic_arrivals_match_a_hand_rolled_recursion(self):
        params = ShuttleParams(arrivals=(3,), max_customers=30, capacity=5, dispatch_cost=20, holding_cost=1, gamma=0.8)
        mdp = build_mdp(params)
        V = np.zeros(31)
        for _ in range(12):
            new = np.empty(31)
            for s in range(31):
                stay = 1.0 * s + 0.8 * V[min(s + 3, 30)]
                go = 20 + 1.0 * max(s - 5, 0) + 0.8 * V[min(max(s - 5, 0) + 3, 30)]
                new[s] = min(stay, go)
            V = new
        np.testing.assert_allclose(solvers.enumeration(mdp, 12).values, V, atol=1e-12)

    def test_more_shuttles_allowed_never_hurts(self):
        one = solvers.policy_iteration(build_mdp(ShuttleParams()))
        two = solvers.policy_iteration(build_mdp(ShuttleParams(max_dispatch=2)))
        self.assertTrue(np.all(two.values <= one.values + 1e-9))
        self.assertEqual(two.policy.max(), 2)

    def test_threshold_helpers(self):
        self.assertIsNone(solvers.policy_threshold(np.zeros(5, dtype=int)))
        self.assertEqual(solvers.policy_threshold(np.array([0, 0, 1, 1])), 2)
        self.assertTrue(solvers.is_monotone(np.array([0, 0, 1, 1])))
        self.assertFalse(solvers.is_monotone(np.array([0, 1, 0, 1])))


if __name__ == "__main__":
    unittest.main()
