import Mathlib

/-!
# Discounted returns

`rl.pg.reinforce.discounted_returns` computes the reward-to-go returns of an episode with the backward
recursion `G_t = r_t + γ G_{t+1}`. This file proves that the recursion gives the definition
`G_t = Σ_{k ≥ 0} γ^k r_{t+k}` at every step.
-/

namespace RLVerify.Returns

/-- The discounted return of a list of rewards: `r₀ + γ r₁ + γ² r₂ + ⋯` (recursive definition). -/
def discounted (γ : ℚ) : List ℚ → ℚ
  | [] => 0
  | r :: rs => r + γ * discounted γ rs

/-- `discounted` is the usual sum `Σ_k γ^k r_k` (with `getD`, which reads `l[k]` and defaults to 0). -/
theorem discounted_eq_sum (γ : ℚ) (l : List ℚ) :
    discounted γ l = ∑ k ∈ Finset.range l.length, γ ^ k * l.getD k 0 := by
  induction l with
  | nil => simp [discounted]
  | cons r rs ih =>
    rw [discounted, ih, List.length_cons, Finset.sum_range_succ']
    simp only [List.getD_cons_succ, List.getD_cons_zero, pow_zero, one_mul, pow_succ, Finset.mul_sum]
    rw [add_comm]
    congr 1
    exact Finset.sum_congr rfl fun k _ => by ring

/-- Reward-to-go returns of every step by the backward recursion (what the Python code implements). -/
def returnsBackward (γ : ℚ) : List ℚ → List ℚ
  | [] => []
  | r :: rs =>
    let rest := returnsBackward γ rs
    (r + γ * rest.headD 0) :: rest

@[simp] theorem returnsBackward_nil (γ : ℚ) : returnsBackward γ [] = [] := rfl

theorem returnsBackward_cons (γ r : ℚ) (rs : List ℚ) :
    returnsBackward γ (r :: rs) = (r + γ * (returnsBackward γ rs).headD 0) :: returnsBackward γ rs := rfl

@[simp] theorem returnsBackward_length (γ : ℚ) (l : List ℚ) : (returnsBackward γ l).length = l.length := by
  induction l with
  | nil => rfl
  | cons r rs ih => simp [returnsBackward_cons, ih]

/-- The first entry is the discounted return of the whole episode. -/
theorem returnsBackward_headD (γ : ℚ) (l : List ℚ) : (returnsBackward γ l).headD 0 = discounted γ l := by
  induction l with
  | nil => rfl
  | cons r rs ih => rw [returnsBackward_cons, List.headD_cons, discounted, ih]

/-- **Entry `t` of the backward recursion is the discounted return from step `t` on.** -/
theorem returnsBackward_getElem (γ : ℚ) (l : List ℚ) (t : ℕ) (ht : t < l.length) :
    (returnsBackward γ l)[t]'(by simpa using ht) = discounted γ (l.drop t) := by
  induction l generalizing t with
  | nil => simp at ht
  | cons r rs ih =>
    cases t with
    | zero =>
      simp only [returnsBackward_cons, List.getElem_cons_zero, List.drop_zero, discounted, returnsBackward_headD]
    | succ t =>
      have ht' : t < rs.length := by simpa using ht
      simp only [returnsBackward_cons, List.getElem_cons_succ, List.drop_succ_cons, ih t ht']

/-- Combined with `discounted_eq_sum`: `G_t = Σ_k γ^k r_{t+k}` (`(l.drop t).getD k 0` is `r_{t+k}`). -/
theorem returnsBackward_getElem_eq_sum (γ : ℚ) (l : List ℚ) (t : ℕ) (ht : t < l.length) :
    (returnsBackward γ l)[t]'(by simpa using ht) =
      ∑ k ∈ Finset.range (l.drop t).length, γ ^ k * (l.drop t).getD k 0 := by
  rw [returnsBackward_getElem γ l t ht, discounted_eq_sum]

/-- With `γ = 0` the return is the immediate reward. -/
theorem discounted_zero (l : List ℚ) (h : l ≠ []) : discounted 0 l = l.headD 0 := by
  cases l with
  | nil => exact absurd rfl h
  | cons r rs => simp [discounted]

/-- With `γ = 1` the return is the plain sum of the rewards. -/
theorem discounted_one (l : List ℚ) : discounted 1 l = l.sum := by
  induction l with
  | nil => simp [discounted]
  | cons r rs ih => simp [discounted, ih]

end RLVerify.Returns
