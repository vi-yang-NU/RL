import RLVerify.Shuttle

/-!
# Reference values for the unit tests

    lake env lean --run ShuttleRef.lean > reference/shuttle_reference.json

evaluates the verified definition `RLVerify.Shuttle.Params.value` (optimal expected discounted cost with `n`
periods to go, exact rationals) on a few small instances and prints them as JSON, every number as an exact
fraction "num/den". `tests/unit/test_dp_lean_reference.py` compares `rl.dp.solvers.enumeration` with them.
The instances are small because `value` is the plain recursive specification (no memoisation): its cost grows
like (2 · #arrival values)^horizon, and this script runs in Lean's interpreter.
-/

open RLVerify.Shuttle

structure Instance where
  name : String
  params : Params
  horizons : Nat

def uniform (k : Nat) : Fin k → ℚ := fun _ => (1 : ℚ) / k

def instances : List Instance := [
  { name := "two_arrival_values",
    params := { capacity := 3, dispatchCost := 10, holdingCost := 1, k := 2, arrivals := ![1, 2],
                probs := uniform 2, maxCustomers := 8, gamma := 1/2 },
    horizons := 8 },
  { name := "non_uniform_arrivals",
    params := { capacity := 2, dispatchCost := 5, holdingCost := 2, k := 3, arrivals := ![0, 1, 3],
                probs := ![1/4, 1/4, 1/2], maxCustomers := 6, gamma := 9/10 },
    horizons := 6 },
  { name := "deterministic_arrivals",
    params := { capacity := 4, dispatchCost := 12, holdingCost := 3/2, k := 1, arrivals := ![2],
                probs := ![1], maxCustomers := 10, gamma := 3/4 },
    horizons := 10 },
  { name := "handout_parameters_small_station",
    params := { capacity := 15, dispatchCost := 100, holdingCost := 2, k := 5, arrivals := ![1, 2, 3, 4, 5],
                probs := uniform 5, maxCustomers := 20, gamma := 19/20 },
    horizons := 4 }
]

/-- A rational as the JSON string "num/den". -/
def frac (x : ℚ) : String := s!"\"{x.num}/{x.den}\""

def listJson (xs : List String) : String := "[" ++ ", ".intercalate xs ++ "]"

def finList {α : Type} (k : Nat) (f : Fin k → α) : List α := (List.finRange k).map f

def paramsJson (p : Params) : String :=
  "{" ++ ", ".intercalate [
    s!"\"capacity\": {p.capacity}",
    s!"\"dispatch_cost\": {frac p.dispatchCost}",
    s!"\"holding_cost\": {frac p.holdingCost}",
    s!"\"arrivals\": {listJson ((finList p.k p.arrivals).map toString)}",
    s!"\"arrival_probs\": {listJson ((finList p.k p.probs).map frac)}",
    s!"\"max_customers\": {p.maxCustomers}",
    s!"\"gamma\": {frac p.gamma}"
  ] ++ "}"

def instanceJson (inst : Instance) : String :=
  let p := inst.params
  let states := List.range (p.maxCustomers + 1)
  let horizons := List.range (inst.horizons + 1)
  let table := horizons.map fun n => states.map fun s => p.value n s
  let policies := (List.range inst.horizons).map fun n => states.map fun s => p.policy n s
  "{" ++ ", ".intercalate [
    s!"\"name\": \"{inst.name}\"",
    s!"\"params\": {paramsJson p}",
    s!"\"horizons\": {listJson (horizons.map toString)}",
    s!"\"values\": {listJson (table.map fun row => listJson (row.map frac))}",
    s!"\"first_actions\": {listJson (policies.map fun row => listJson (row.map toString))}"
  ] ++ "}"

def main : IO Unit := do
  let body := ", ".intercalate (instances.map instanceJson)
  IO.println ("{\"generator\": \"lake env lean --run ShuttleRef.lean (lean/ShuttleRef.lean)\", "
    ++ "\"definition\": \"RLVerify.Shuttle.Params.value: optimal expected discounted cost with n periods to go, exact rationals\", "
    ++ "\"notes\": \"values[n][s] is the value with n periods to go in state s, as an exact fraction; first_actions[n][s] is the optimal action with n+1 periods to go (1 = dispatch)\", "
    ++ s!"\"instances\": [{body}]" ++ "}")
