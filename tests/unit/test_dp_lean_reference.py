"""Cross-checks of the floating-point solvers against exact arithmetic.

1. A re-implementation of backward induction with Python `fractions.Fraction` (exact rationals), independent
   of the numpy code, on small instances.
2. The values computed by the formally verified Lean definition (`RLVerify.Shuttle.Params.value`,
   lean/RLVerify/Shuttle.lean), exported by `cd lean && lake env lean --run ShuttleRef.lean > reference/shuttle_reference.json`.
   The test is skipped when that file has not been generated.
"""
from __future__ import annotations

import json
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from rl.dp import solvers
from rl.dp.shuttle import ShuttleParams, build_mdp
from rl.outputs import ROOT

REFERENCE = ROOT / "lean" / "reference" / "shuttle_reference.json"

# The same instances as lean/ShuttleRef.lean (probabilities as exact fractions).
INSTANCES = {
    "two_arrival_values": dict(capacity=3, dispatch_cost=10, holding_cost=1, arrivals=(1, 2),
                               probs=(Fraction(1, 2), Fraction(1, 2)), max_customers=8, gamma=Fraction(1, 2), horizons=8),
    "non_uniform_arrivals": dict(capacity=2, dispatch_cost=5, holding_cost=2, arrivals=(0, 1, 3),
                                 probs=(Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)), max_customers=6, gamma=Fraction(9, 10), horizons=6),
    "deterministic_arrivals": dict(capacity=4, dispatch_cost=12, holding_cost=Fraction(3, 2), arrivals=(2,),
                                   probs=(Fraction(1),), max_customers=10, gamma=Fraction(3, 4), horizons=10),
    "handout_parameters_small_station": dict(capacity=15, dispatch_cost=100, holding_cost=2, arrivals=(1, 2, 3, 4, 5),
                                             probs=(Fraction(1, 5),) * 5, max_customers=20, gamma=Fraction(19, 20), horizons=4),
}


def exact_values(inst: dict, horizon: int) -> list[list[Fraction]]:
    """V_n(s) for n = 0..horizon by backward induction in exact rational arithmetic."""
    M, K = inst["max_customers"], inst["capacity"]
    c_f, c_h, gamma = Fraction(inst["dispatch_cost"]), Fraction(inst["holding_cost"]), inst["gamma"]
    V = [Fraction(0)] * (M + 1)
    table = [V]
    for _ in range(horizon):
        new = []
        for s in range(M + 1):
            best = None
            for a in (0, 1):
                left = max(s - a * K, 0)
                expect = sum(p * V[min(left + arr, M)] for arr, p in zip(inst["arrivals"], inst["probs"]))
                q = a * c_f + c_h * left + gamma * expect
                best = q if best is None else min(best, q)
            new.append(best)
        V = new
        table.append(V)
    return table


def params_from_reference(p: dict) -> ShuttleParams:
    """ShuttleParams from the "params" entry of lean/reference/shuttle_reference.json (exact fractions)."""
    return ShuttleParams(capacity=p["capacity"], dispatch_cost=float(Fraction(p["dispatch_cost"])),
                         holding_cost=float(Fraction(p["holding_cost"])), arrivals=tuple(p["arrivals"]),
                         arrival_probs=tuple(float(Fraction(x)) for x in p["arrival_probs"]),
                         max_customers=p["max_customers"], gamma=float(Fraction(p["gamma"])))


def params_of(inst: dict) -> ShuttleParams:
    return ShuttleParams(capacity=inst["capacity"], dispatch_cost=float(inst["dispatch_cost"]), holding_cost=float(inst["holding_cost"]),
                         arrivals=inst["arrivals"], arrival_probs=tuple(float(p) for p in inst["probs"]),
                         max_customers=inst["max_customers"], gamma=float(inst["gamma"]), horizon=inst["horizons"])


class ExactFractionTests(unittest.TestCase):
    def test_enumeration_matches_exact_rational_backward_induction(self):
        for name, inst in INSTANCES.items():
            mdp = build_mdp(params_of(inst))
            table = exact_values(inst, inst["horizons"])
            for n, exact in enumerate(table):
                got = solvers.enumeration(mdp, n).values
                np.testing.assert_allclose(got, [float(v) for v in exact], rtol=1e-12, atol=1e-12, err_msg=f"{name}, horizon {n}")

    def test_exact_values_are_monotone(self):
        """The properties proved in Lean (value_mono_horizon, value_mono_state) hold for the exact values too."""
        for inst in INSTANCES.values():
            table = exact_values(inst, inst["horizons"])
            for n in range(len(table) - 1):
                self.assertTrue(all(a <= b for a, b in zip(table[n], table[n + 1])))
            for row in table:
                self.assertTrue(all(a <= b for a, b in zip(row, row[1:])))


class LeanReferenceTests(unittest.TestCase):
    def setUp(self):
        if not REFERENCE.exists():
            self.skipTest("lean/reference/shuttle_reference.json not generated (cd lean && lake env lean --run ShuttleRef.lean > reference/shuttle_reference.json)")
        self.data = json.loads(REFERENCE.read_text())

    def test_reference_covers_the_known_instances(self):
        self.assertEqual({i["name"] for i in self.data["instances"]}, set(INSTANCES))

    def test_enumeration_matches_the_lean_values(self):
        for inst in self.data["instances"]:
            mdp = build_mdp(params_from_reference(inst["params"]))
            for n, row in enumerate(inst["values"]):
                result = solvers.enumeration(mdp, n)
                exact = [float(Fraction(x)) for x in row]
                np.testing.assert_allclose(result.values, exact, rtol=1e-12, atol=1e-12, err_msg=f"{inst['name']}, horizon {n}")
                if n >= 1:
                    np.testing.assert_array_equal(result.policy, inst["first_actions"][n - 1], err_msg=f"{inst['name']} policy, horizon {n}")

    def test_lean_values_equal_the_fraction_re_implementation(self):
        for inst in self.data["instances"]:
            table = exact_values(INSTANCES[inst["name"]], len(inst["values"]) - 1)
            for n, row in enumerate(inst["values"]):
                self.assertEqual([Fraction(x) for x in row], table[n], f"{inst['name']}, horizon {n}")


if __name__ == "__main__":
    unittest.main()
