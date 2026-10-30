import Mathlib

/-!
# The shuttle dispatching MDP, formally

This file is the formal counterpart of `rl/dp/shuttle.py` (the model) and `rl/dp/solvers.py` (the solvers).
It defines the same model over exact rationals and proves the properties the Python code relies on and the
unit tests check numerically:

* `next_le_max`: the number of customers never exceeds the cap (the state space is closed);
* `cost_nonneg`, `value_nonneg`: one-step costs and optimal values are non-negative;
* `value_mono_horizon`: more periods to go never lowers the optimal cost;
* `value_mono_state`: more customers waiting never lowers the optimal cost (the value function is
  non-decreasing in the state, as the plots show);
* `bellman_contraction`: the Bellman operator `T` is a `gamma`-contraction in the sup norm;
* `value_succ_sub_bound`: hence the changes between successive value-iteration iterates shrink at least
  geometrically (`||V_{n+1} - V_n|| <= gamma^n ||V_1 - V_0||`);
* `vi_error_bound`: the stopping rule of value iteration is sound: when `||T V - V|| <= d`, the iterate `T V`
  is within `gamma / (1 - gamma) * d` of the fixed point (this is `info["error_bound"]` in the Python code);
* `policy_improvement`: a greedy policy for the value of the current policy is at least as good
  (the step of policy iteration).

The model has two actions (dispatch or not), exactly the handout's problem. The Python code additionally
allows several shuttles per period and other holding-cost timings; those variants are outside this file.

`ShuttleRef.lean` evaluates `value` on small instances and writes `lean/reference/shuttle_reference.json`;
`tests/unit/test_dp_lean_reference.py` checks the floating-point enumeration against those exact values.
-/

namespace RLVerify.Shuttle

/-- Problem parameters over exact rationals. `arrivals i` customers arrive with probability `probs i`. -/
structure Params where
  capacity : ℕ
  dispatchCost : ℚ
  holdingCost : ℚ
  k : ℕ
  arrivals : Fin k → ℕ
  probs : Fin k → ℚ
  maxCustomers : ℕ
  gamma : ℚ

namespace Params

variable (p : Params)

/-- Customers left after dispatching `a` shuttles: `(s - a K)⁺` (natural-number subtraction truncates at 0). -/
def left (s a : ℕ) : ℕ := s - a * p.capacity

/-- The next state: the customers left plus the arrivals, capped at `maxCustomers`. -/
def next (s a arr : ℕ) : ℕ := min (p.left s a + arr) p.maxCustomers

/-- One-step cost: `a` shuttles plus holding for the customers left after the dispatch. -/
def cost (s a : ℕ) : ℚ := (a : ℚ) * p.dispatchCost + p.holdingCost * (p.left s a : ℚ)

/-- Expected value at the next state. -/
def expect (V : ℕ → ℚ) (s a : ℕ) : ℚ := ∑ i, p.probs i * V (p.next s a (p.arrivals i))

/-- `Q(s, a) = c(s, a) + γ E[V(s')]`. -/
def Q (V : ℕ → ℚ) (s a : ℕ) : ℚ := p.cost s a + p.gamma * p.expect V s a

/-- The Bellman optimality operator for the two actions: do not dispatch (0) or dispatch (1). -/
def T (V : ℕ → ℚ) (s : ℕ) : ℚ := min (p.Q V s 0) (p.Q V s 1)

/-- The one-step operator of a deterministic policy `π` (actions `0` or `1`). -/
def Tpi (π : ℕ → ℕ) (V : ℕ → ℚ) (s : ℕ) : ℚ := p.Q V s (π s)

/-- Optimal value with `n` periods to go: enumeration / backward induction (`rl.dp.solvers.enumeration`). -/
def value : ℕ → ℕ → ℚ
  | 0, _ => 0
  | n + 1, s => p.T (value n) s

theorem value_zero (s : ℕ) : p.value 0 s = 0 := rfl

theorem value_succ (n s : ℕ) : p.value (n + 1) s = p.T (p.value n) s := rfl

/-- The optimal first action with `n + 1` periods to go (dispatch iff it is strictly cheaper). -/
def policy (n s : ℕ) : ℕ := if p.Q (p.value n) s 1 < p.Q (p.value n) s 0 then 1 else 0

/-- Well-posed parameters: non-negative costs, a probability distribution of arrivals, `0 ≤ γ < 1`. -/
structure Valid : Prop where
  dispatch_nonneg : 0 ≤ p.dispatchCost
  holding_nonneg : 0 ≤ p.holdingCost
  probs_nonneg : ∀ i, 0 ≤ p.probs i
  probs_sum : ∑ i, p.probs i = 1
  gamma_nonneg : 0 ≤ p.gamma
  gamma_lt_one : p.gamma < 1

/-! ### The state space is closed and costs are non-negative -/

theorem left_le (s a : ℕ) : p.left s a ≤ s := Nat.sub_le _ _

theorem next_le_max (s a arr : ℕ) : p.next s a arr ≤ p.maxCustomers := min_le_right _ _

theorem left_mono (a : ℕ) : Monotone fun s => p.left s a := fun _ _ h => Nat.sub_le_sub_right h _

theorem next_mono (a arr : ℕ) : Monotone fun s => p.next s a arr := fun _ _ h =>
  min_le_min_right _ (Nat.add_le_add_right (p.left_mono a h) _)

theorem cost_nonneg (hv : p.Valid) (s a : ℕ) : 0 ≤ p.cost s a :=
  add_nonneg (mul_nonneg (Nat.cast_nonneg _) hv.dispatch_nonneg) (mul_nonneg hv.holding_nonneg (Nat.cast_nonneg _))

theorem cost_mono (hv : p.Valid) (a : ℕ) : Monotone fun s => p.cost s a := fun s t h => by
  unfold cost
  have : (p.left s a : ℚ) ≤ p.left t a := Nat.cast_le.mpr (p.left_mono a h)
  nlinarith [hv.holding_nonneg]

/-! ### Monotonicity of the Bellman operator -/

theorem expect_mono (hv : p.Valid) {V W : ℕ → ℚ} (h : ∀ s, V s ≤ W s) (s a : ℕ) :
    p.expect V s a ≤ p.expect W s a :=
  Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (h _) (hv.probs_nonneg i)

theorem expect_nonneg (hv : p.Valid) {V : ℕ → ℚ} (h : ∀ s, 0 ≤ V s) (s a : ℕ) : 0 ≤ p.expect V s a :=
  Finset.sum_nonneg fun i _ => mul_nonneg (hv.probs_nonneg i) (h _)

theorem expect_const (hv : p.Valid) (c : ℚ) (s a : ℕ) : p.expect (fun _ => c) s a = c := by
  simp [expect, ← Finset.sum_mul, hv.probs_sum]

theorem Q_mono (hv : p.Valid) {V W : ℕ → ℚ} (h : ∀ s, V s ≤ W s) (s a : ℕ) : p.Q V s a ≤ p.Q W s a := by
  unfold Q
  exact add_le_add (le_refl _) (mul_le_mul_of_nonneg_left (p.expect_mono hv h s a) hv.gamma_nonneg)

theorem T_mono (hv : p.Valid) {V W : ℕ → ℚ} (h : ∀ s, V s ≤ W s) (s : ℕ) : p.T V s ≤ p.T W s :=
  min_le_min (p.Q_mono hv h s 0) (p.Q_mono hv h s 1)

theorem Tpi_mono (hv : p.Valid) (π : ℕ → ℕ) {V W : ℕ → ℚ} (h : ∀ s, V s ≤ W s) (s : ℕ) :
    p.Tpi π V s ≤ p.Tpi π W s :=
  p.Q_mono hv h s (π s)

theorem Q_nonneg (hv : p.Valid) {V : ℕ → ℚ} (h : ∀ s, 0 ≤ V s) (s a : ℕ) : 0 ≤ p.Q V s a :=
  add_nonneg (p.cost_nonneg hv s a) (mul_nonneg hv.gamma_nonneg (p.expect_nonneg hv h s a))

theorem value_nonneg (hv : p.Valid) : ∀ n s, 0 ≤ p.value n s
  | 0, _ => le_refl 0
  | n + 1, s => by
    rw [value_succ]
    exact le_min (p.Q_nonneg hv (value_nonneg hv n) s 0) (p.Q_nonneg hv (value_nonneg hv n) s 1)

/-- More periods to go never lowers the optimal cost. -/
theorem value_mono_horizon (hv : p.Valid) : ∀ n s, p.value n s ≤ p.value (n + 1) s
  | 0, s => p.value_nonneg hv 1 s
  | n + 1, s => by
    rw [value_succ, value_succ]
    exact p.T_mono hv (value_mono_horizon hv n) s

/-! ### Monotonicity in the state -/

theorem expect_mono_state (hv : p.Valid) {V : ℕ → ℚ} (hV : Monotone V) (a : ℕ) :
    Monotone fun s => p.expect V s a := fun _ _ h =>
  Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hV (p.next_mono a (p.arrivals i) h)) (hv.probs_nonneg i)

theorem Q_mono_state (hv : p.Valid) {V : ℕ → ℚ} (hV : Monotone V) (a : ℕ) : Monotone fun s => p.Q V s a :=
  fun _ _ h => add_le_add (p.cost_mono hv a h) (mul_le_mul_of_nonneg_left (p.expect_mono_state hv hV a h) hv.gamma_nonneg)

theorem T_mono_state (hv : p.Valid) {V : ℕ → ℚ} (hV : Monotone V) : Monotone (p.T V) := fun _ _ h =>
  min_le_min (p.Q_mono_state hv hV 0 h) (p.Q_mono_state hv hV 1 h)

/-- More customers waiting never lowers the optimal cost: `value n` is non-decreasing in the state. -/
theorem value_mono_state (hv : p.Valid) : ∀ n, Monotone (p.value n)
  | 0 => fun _ _ _ => le_refl _
  | n + 1 => fun s t h => by
    rw [value_succ, value_succ]
    exact p.T_mono_state hv (value_mono_state hv n) h

/-! ### The Bellman operator is a γ-contraction -/

/-- Only states up to the cap matter: `expect` reads `V` at states `≤ maxCustomers`. -/
theorem expect_sub_bound (hv : p.Valid) {V W : ℕ → ℚ} {d : ℚ}
    (h : ∀ s ≤ p.maxCustomers, |V s - W s| ≤ d) (s a : ℕ) :
    |p.expect V s a - p.expect W s a| ≤ d := by
  unfold expect
  rw [← Finset.sum_sub_distrib]
  calc |∑ i, (p.probs i * V (p.next s a (p.arrivals i)) - p.probs i * W (p.next s a (p.arrivals i)))|
      ≤ ∑ i, |p.probs i * V (p.next s a (p.arrivals i)) - p.probs i * W (p.next s a (p.arrivals i))| :=
        Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, p.probs i * |V (p.next s a (p.arrivals i)) - W (p.next s a (p.arrivals i))| := by
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [← mul_sub, abs_mul, abs_of_nonneg (hv.probs_nonneg i)]
    _ ≤ ∑ i, p.probs i * d :=
        Finset.sum_le_sum fun i _ =>
          mul_le_mul_of_nonneg_left (h _ (p.next_le_max s a (p.arrivals i))) (hv.probs_nonneg i)
    _ = d := by rw [← Finset.sum_mul, hv.probs_sum, one_mul]

theorem Q_sub_bound (hv : p.Valid) {V W : ℕ → ℚ} {d : ℚ}
    (h : ∀ s ≤ p.maxCustomers, |V s - W s| ≤ d) (s a : ℕ) :
    |p.Q V s a - p.Q W s a| ≤ p.gamma * d := by
  have e : p.Q V s a - p.Q W s a = p.gamma * (p.expect V s a - p.expect W s a) := by
    unfold Q; ring
  rw [e, abs_mul, abs_of_nonneg hv.gamma_nonneg]
  exact mul_le_mul_of_nonneg_left (p.expect_sub_bound hv h s a) hv.gamma_nonneg

/-- **Contraction.** If `V` and `W` differ by at most `d` on the state space, `T V` and `T W` differ by at
most `γ d` everywhere. -/
theorem bellman_contraction (hv : p.Valid) {V W : ℕ → ℚ} {d : ℚ}
    (h : ∀ s ≤ p.maxCustomers, |V s - W s| ≤ d) (s : ℕ) :
    |p.T V s - p.T W s| ≤ p.gamma * d := by
  unfold T
  calc |min (p.Q V s 0) (p.Q V s 1) - min (p.Q W s 0) (p.Q W s 1)|
      ≤ max |p.Q V s 0 - p.Q W s 0| |p.Q V s 1 - p.Q W s 1| := abs_min_sub_min_le_max _ _ _ _
    _ ≤ p.gamma * d := max_le (p.Q_sub_bound hv h s 0) (p.Q_sub_bound hv h s 1)

/-- Successive iterates of value iteration (= `value n`) get closer at least geometrically on the state space. -/
theorem value_succ_sub_bound (hv : p.Valid) {d : ℚ} (h0 : ∀ s ≤ p.maxCustomers, |p.value 1 s - p.value 0 s| ≤ d) :
    ∀ n, ∀ s ≤ p.maxCustomers, |p.value (n + 1) s - p.value n s| ≤ p.gamma ^ n * d
  | 0, s, hs => by simpa using h0 s hs
  | n + 1, s, _ => by
    have ih : ∀ t ≤ p.maxCustomers, |p.value (n + 1) t - p.value n t| ≤ p.gamma ^ n * d :=
      value_succ_sub_bound hv h0 n
    have hc := p.bellman_contraction hv ih s
    calc |p.value (n + 1 + 1) s - p.value (n + 1) s|
        = |p.T (p.value (n + 1)) s - p.T (p.value n) s| := by rw [value_succ, value_succ (p := p) n]
      _ ≤ p.gamma * (p.gamma ^ n * d) := hc
      _ = p.gamma ^ (n + 1) * d := by ring

/-- **Soundness of the stopping rule.** If `Vstar` is a fixed point of `T` on the state space and the
Bellman residual of `V` is at most `d`, then `T V` is within `γ d / (1 - γ)` of `Vstar`. -/
theorem vi_error_bound (hv : p.Valid) {V Vstar : ℕ → ℚ}
    (hfix : ∀ s ≤ p.maxCustomers, p.T Vstar s = Vstar s) {d : ℚ}
    (h : ∀ s ≤ p.maxCustomers, |p.T V s - V s| ≤ d) :
    ∀ s ≤ p.maxCustomers, |p.T V s - Vstar s| ≤ p.gamma / (1 - p.gamma) * d := by
  set S := Finset.range (p.maxCustomers + 1) with hS
  have hne : S.Nonempty := ⟨0, by simp [hS]⟩
  set e := S.sup' hne fun s => |V s - Vstar s| with he_def
  have mem : ∀ s, s ≤ p.maxCustomers → s ∈ S := fun s hs => by simp [hS, Nat.lt_succ_of_le hs]
  have he : ∀ s ≤ p.maxCustomers, |V s - Vstar s| ≤ e := fun s hs =>
    Finset.le_sup' (fun s => |V s - Vstar s|) (mem s hs)
  have hcontr : ∀ s, |p.T V s - p.T Vstar s| ≤ p.gamma * e := p.bellman_contraction hv he
  have h1g : 0 < 1 - p.gamma := by linarith [hv.gamma_lt_one]
  -- e ≤ d + γ e, because |V s - V* s| ≤ |V s - T V s| + |T V s - T V* s| on the state space
  have hrec : e ≤ d + p.gamma * e := by
    refine Finset.sup'_le hne _ fun s hs => ?_
    have hs' : s ≤ p.maxCustomers := Nat.lt_succ_iff.mp (Finset.mem_range.mp hs)
    calc |V s - Vstar s| = |(V s - p.T V s) + (p.T V s - p.T Vstar s)| := by rw [hfix s hs']; ring_nf
      _ ≤ |V s - p.T V s| + |p.T V s - p.T Vstar s| := abs_add_le _ _
      _ ≤ d + p.gamma * e := add_le_add (by rw [abs_sub_comm]; exact h s hs') (hcontr s)
  have hebound : e ≤ d / (1 - p.gamma) := by
    rw [le_div_iff₀ h1g]; nlinarith
  intro s hs
  calc |p.T V s - Vstar s| = |p.T V s - p.T Vstar s| := by rw [hfix s hs]
    _ ≤ p.gamma * e := hcontr s
    _ ≤ p.gamma * (d / (1 - p.gamma)) := mul_le_mul_of_nonneg_left hebound hv.gamma_nonneg
    _ = p.gamma / (1 - p.gamma) * d := by ring

/-! ### Policy improvement (the step of policy iteration) -/

theorem T_le_Tpi {π : ℕ → ℕ} (hπ : ∀ s, π s ≤ 1) (V : ℕ → ℚ) (s : ℕ) : p.T V s ≤ p.Tpi π V s := by
  unfold T Tpi
  rcases Nat.le_one_iff_eq_zero_or_eq_one.mp (hπ s) with h | h
  · rw [h]; exact min_le_left _ _
  · rw [h]; exact min_le_right _ _

/-- If `V` is the value of policy `π` (a fixed point of `Tpi π`) and `π'` is greedy with respect to `V`
(`Tpi π' V = T V`), then every iterate of `Tpi π'` from `V` stays below `V`; since those iterates converge
to the value of `π'`, the new policy is at least as good in every state. -/
theorem policy_improvement (hv : p.Valid) {π π' : ℕ → ℕ} (hπ : ∀ s, π s ≤ 1) {V : ℕ → ℚ}
    (hfix : ∀ s, p.Tpi π V s = V s) (hgreedy : ∀ s, p.Tpi π' V s = p.T V s) :
    ∀ n s, (p.Tpi π')^[n] V s ≤ V s
  | 0, _ => le_refl _
  | n + 1, s => by
    rw [Function.iterate_succ_apply']
    calc p.Tpi π' ((p.Tpi π')^[n] V) s ≤ p.Tpi π' V s := p.Tpi_mono hv π' (policy_improvement hv hπ hfix hgreedy n) s
      _ = p.T V s := hgreedy s
      _ ≤ p.Tpi π V s := p.T_le_Tpi hπ V s
      _ = V s := hfix s

/-- The greedy policy of `policy` indeed attains the minimum. -/
theorem Q_policy_eq_T (n s : ℕ) : p.Q (p.value n) s (p.policy n s) = p.T (p.value n) s := by
  unfold policy T
  split_ifs with h
  · exact (min_eq_right h.le).symm
  · exact (min_eq_left (not_lt.mp h)).symm

end Params

end RLVerify.Shuttle
