"""Unit tests for the shuttle MDP model (rl/dp/shuttle.py) and parameter overrides (rl/params.py)."""
from __future__ import annotations

import unittest

import numpy as np

from rl.dp.shuttle import HOLDING_CHARGED, ShuttleParams, TabularMDP, build_mdp
from rl.params import apply_overrides, parse_overrides


class ShuttleParamsTests(unittest.TestCase):
    def test_defaults_match_the_handout(self):
        p = ShuttleParams()
        self.assertEqual((p.capacity, p.dispatch_cost, p.holding_cost), (15, 100.0, 2.0))
        self.assertEqual(p.arrivals, (1, 2, 3, 4, 5))
        self.assertIsNone(p.arrival_probs)
        self.assertEqual((p.max_customers, p.gamma, p.horizon), (200, 0.95, 500))
        self.assertEqual(p.max_dispatch, 1)
        self.assertEqual(p.holding_charged, "after_dispatch")
        self.assertEqual(p.n_states, 201)
        self.assertTrue(p.is_uniform)
        self.assertAlmostEqual(p.mean_arrivals, 3.0)

    def test_with_changes_fields_without_touching_the_original(self):
        p = ShuttleParams()
        q = p.with_(capacity=5, gamma=0.5)
        self.assertEqual((q.capacity, q.gamma), (5, 0.5))
        self.assertEqual((p.capacity, p.gamma), (15, 0.95))

    def test_command_line_overrides_are_parsed_and_coerced(self):
        overrides = parse_overrides(["capacity=10", "arrivals=[1,2]", "gamma=0.5", "holding_charged=before_dispatch", "horizon=7.0"])
        p = apply_overrides(ShuttleParams(), overrides)
        self.assertEqual(p.capacity, 10)
        self.assertEqual(p.arrivals, (1, 2))
        self.assertIsInstance(p.arrivals, tuple)
        self.assertEqual(p.gamma, 0.5)
        self.assertEqual(p.holding_charged, "before_dispatch")
        self.assertEqual(p.horizon, 7)
        self.assertIsInstance(p.horizon, int)

    def test_unknown_override_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_overrides(ShuttleParams(), {"capacty": 10})
        with self.assertRaises(SystemExit):
            parse_overrides(["capacity"])

    def test_invalid_values_are_rejected(self):
        bad = [
            dict(capacity=-1), dict(capacity=1.5), dict(dispatch_cost=-5), dict(holding_cost=float("nan")),
            dict(arrivals=()), dict(arrivals=(1, -2)), dict(arrival_probs=(0.5, 0.5)),
            dict(arrivals=(1, 2), arrival_probs=(0.7, 0.7)), dict(arrivals=(1, 2), arrival_probs=(1.5, -0.5)),
            dict(gamma=1.2), dict(gamma=-0.1), dict(horizon=-1), dict(max_dispatch=0),
            dict(holding_charged="sometimes"), dict(max_customers=-3),
        ]
        for changes in bad:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ShuttleParams(**changes)

    def test_arrival_distribution_merges_duplicates_and_normalises(self):
        values, probs = ShuttleParams(arrivals=(2, 1, 1)).arrival_distribution()
        np.testing.assert_array_equal(values, [1, 2])
        np.testing.assert_allclose(probs, [2 / 3, 1 / 3])
        values, probs = ShuttleParams(arrivals=(1, 5), arrival_probs=(0.25, 0.75)).arrival_distribution()
        np.testing.assert_allclose(probs, [0.25, 0.75])
        self.assertFalse(ShuttleParams(arrivals=(1, 5), arrival_probs=(0.25, 0.75)).is_uniform)

    def test_describe_mentions_non_default_structure(self):
        self.assertIn("Unif{1,…,5}", ShuttleParams().describe())
        self.assertIn("2 shuttles", ShuttleParams(max_dispatch=2).describe())
        self.assertIn("before dispatch", ShuttleParams(holding_charged="before_dispatch").describe())
        self.assertIn("A_t = 3", ShuttleParams(arrivals=(3,)).describe())
        self.assertIn("non-uniform", ShuttleParams(arrivals=(1, 2), arrival_probs=(0.9, 0.1)).describe())


class BuildMdpTests(unittest.TestCase):
    def setUp(self):
        self.params = ShuttleParams()
        self.mdp = build_mdp(self.params)

    def test_shapes_labels_and_probabilities(self):
        self.assertEqual(self.mdp.transitions.shape, (2, 201, 201))
        self.assertEqual(self.mdp.costs.shape, (201, 2))
        self.assertEqual((self.mdp.n_states, self.mdp.n_actions), (201, 2))
        self.assertEqual(self.mdp.action_labels, ("Do not dispatch", "Dispatch"))
        np.testing.assert_array_equal(self.mdp.states, np.arange(201))
        np.testing.assert_allclose(self.mdp.transitions.sum(axis=2), 1.0)
        self.assertTrue(np.all(self.mdp.transitions >= 0))
        self.assertEqual(self.mdp.gamma, 0.95)

    def test_one_step_costs(self):
        C = self.mdp.costs
        self.assertEqual(C[20, 1], 100 + 2 * 5)  # 15 of 20 leave, 5 keep waiting
        self.assertEqual(C[10, 1], 100)  # everyone leaves, the shuttle is not full
        self.assertEqual(C[20, 0], 2 * 20)
        self.assertEqual(C[0, 0], 0)
        self.assertEqual(C[0, 1], 100)  # an empty shuttle still costs c_f
        self.assertEqual(C[200, 1], 100 + 2 * 185)

    def test_transitions_follow_the_arrivals(self):
        P = self.mdp.transitions
        np.testing.assert_allclose(P[1, 20, 6:11], 0.2)  # 20 - 15 = 5 left, plus 1..5 arrivals
        self.assertAlmostEqual(P[1, 20].sum(), 1.0)
        np.testing.assert_allclose(P[0, 20, 21:26], 0.2)
        self.assertEqual(P[0, 20, 20], 0.0)
        np.testing.assert_allclose(P[1, 3, 1:6], 0.2)  # fewer than K waiting: shuttle takes everyone

    def test_state_never_exceeds_the_cap(self):
        P = self.mdp.transitions
        self.assertEqual(P[0, 200, 200], 1.0)
        np.testing.assert_allclose(P[0, 198, [199, 200]], [0.2, 0.8])
        self.assertEqual(P[0, 198, :199].sum(), 0.0)
        huge = build_mdp(ShuttleParams(arrivals=(500,)))
        np.testing.assert_allclose(huge.transitions[0, :, 200], 1.0)

    def test_capacity_zero_means_a_dispatch_serves_nobody(self):
        mdp = build_mdp(ShuttleParams(capacity=0))
        np.testing.assert_allclose(mdp.costs[:, 1], 100 + 2 * np.arange(201))
        np.testing.assert_allclose(mdp.transitions[1], mdp.transitions[0])

    def test_several_shuttles_per_period(self):
        mdp = build_mdp(ShuttleParams(max_dispatch=2))
        self.assertEqual(mdp.n_actions, 3)
        self.assertEqual(mdp.action_labels, ("0 shuttles", "1 shuttle", "2 shuttles"))
        self.assertEqual(mdp.costs[40, 2], 200 + 2 * 10)
        self.assertEqual(mdp.costs[40, 1], 100 + 2 * 25)
        np.testing.assert_allclose(mdp.transitions[2, 40, 11:16], 0.2)

    def test_holding_cost_timing_variants(self):
        for variant in HOLDING_CHARGED:
            self.assertEqual(build_mdp(ShuttleParams(holding_charged=variant)).costs.shape, (201, 2))
        before = build_mdp(ShuttleParams(holding_charged="before_dispatch")).costs
        self.assertEqual(before[20, 1], 100 + 2 * 20)
        end = build_mdp(ShuttleParams(holding_charged="end_of_period")).costs
        self.assertAlmostEqual(end[20, 1], 100 + 2 * (5 + 3.0))  # 5 left over + 3 expected arrivals
        self.assertAlmostEqual(end[20, 0], 2 * (20 + 3.0))

    def test_non_uniform_and_deterministic_arrivals(self):
        mdp = build_mdp(ShuttleParams(arrivals=(1, 5), arrival_probs=(0.25, 0.75)))
        np.testing.assert_allclose(mdp.transitions[0, 10, [11, 15]], [0.25, 0.75])
        fixed = build_mdp(ShuttleParams(arrivals=(3,)))
        self.assertEqual(fixed.transitions[0, 10, 13], 1.0)
        self.assertEqual(fixed.transitions[1, 20, 8], 1.0)
        self.assertTrue(np.all((fixed.transitions == 0) | (fixed.transitions == 1)))

    def test_single_state_problem(self):
        mdp = build_mdp(ShuttleParams(max_customers=0))
        self.assertEqual(mdp.n_states, 1)
        self.assertEqual(mdp.transitions[0, 0, 0], 1.0)
        np.testing.assert_array_equal(mdp.costs[0], [0.0, 100.0])


class TabularMdpTests(unittest.TestCase):
    def test_rejects_malformed_inputs(self):
        P = np.zeros((1, 2, 2))
        P[0, 0, 1] = 1.0
        P[0, 1, 0] = 1.0
        TabularMDP(P, np.zeros((2, 1)), 0.9, np.arange(2), ("a",))
        with self.assertRaises(ValueError):
            TabularMDP(P * 0.5, np.zeros((2, 1)), 0.9, np.arange(2), ("a",))  # rows sum to 0.5
        with self.assertRaises(ValueError):
            TabularMDP(P, np.zeros((2, 2)), 0.9, np.arange(2), ("a",))  # costs shape
        with self.assertRaises(ValueError):
            TabularMDP(P, np.zeros((2, 1)), 1.5, np.arange(2), ("a",))  # gamma
        with self.assertRaises(ValueError):
            TabularMDP(P, np.array([[np.inf], [0.0]]), 0.9, np.arange(2), ("a",))  # infinite cost
        with self.assertRaises(ValueError):
            TabularMDP(P, np.zeros((2, 1)), 0.9, np.arange(3), ("a",))  # labels
        with self.assertRaises(ValueError):
            TabularMDP(-P, np.zeros((2, 1)), 0.9, np.arange(2), ("a",))  # negative probability

    def test_q_values_by_hand(self):
        P = np.zeros((2, 2, 2))
        P[0, 0, 0] = P[0, 1, 1] = 1.0  # action 0: stay
        P[1, 0, 1] = P[1, 1, 0] = 1.0  # action 1: switch
        C = np.array([[1.0, 3.0], [0.0, 0.0]])
        mdp = TabularMDP(P, C, 0.5, np.arange(2), ("stay", "switch"))
        V = np.array([10.0, 20.0])
        np.testing.assert_allclose(mdp.expected_next(V), [[10.0, 20.0], [20.0, 10.0]])
        np.testing.assert_allclose(mdp.q_values(V), [[1 + 5.0, 3 + 10.0], [0 + 10.0, 0 + 5.0]])
        np.testing.assert_allclose(mdp.bellman(V), [6.0, 5.0])
        P_pi, c_pi = mdp.policy_transitions(np.array([1, 0]))
        np.testing.assert_allclose(P_pi, [[0, 1], [0, 1]])
        np.testing.assert_allclose(c_pi, [3.0, 0.0])
        with self.assertRaises(ValueError):
            mdp.policy_transitions(np.array([2, 0]))


if __name__ == "__main__":
    unittest.main()
