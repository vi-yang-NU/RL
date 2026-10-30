import Mathlib

/-!
# Baselines do not bias the policy gradient

REINFORCE estimates `∇J` by `Σ_t ∇ log π(a_t | s_t) (G_t - b_t)`. For a softmax policy over finitely many
actions, `∂ log π_i / ∂ z_k = δ_{ik} - π_k` (the score). This file proves, for any probability vector `π`
and that score, that

* the expected score is zero (`expected_score_zero`), hence
* subtracting any baseline `b` that does not depend on the action leaves the expected gradient unchanged
  (`baseline_unbiased`), which is why Part 2 can subtract `rl/pg/baselines.py` without changing what the
  algorithm optimises, and
* the softmax really is a probability vector (`softmax_sum`, `softmax_pos`).

The score's closed form is the textbook derivative of `log softmax`; `tests/unit/test_pg_train.py`
checks that PyTorch's automatic differentiation of the implemented loss matches it numerically.
-/

namespace RLVerify.Baseline

open Finset

variable {n : ℕ}

/-- The softmax policy with logits `z`. -/
noncomputable def softmax (z : Fin n → ℝ) (i : Fin n) : ℝ := Real.exp (z i) / ∑ j, Real.exp (z j)

theorem softmax_sum_exp_pos [NeZero n] (z : Fin n → ℝ) : 0 < ∑ j, Real.exp (z j) :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty

theorem softmax_pos [NeZero n] (z : Fin n → ℝ) (i : Fin n) : 0 < softmax z i :=
  div_pos (Real.exp_pos _) (softmax_sum_exp_pos z)

theorem softmax_sum [NeZero n] (z : Fin n → ℝ) : ∑ i, softmax z i = 1 := by
  unfold softmax
  rw [← Finset.sum_div, div_self (softmax_sum_exp_pos z).ne']

/-- `∂ log π_i / ∂ z_k = δ_{ik} - π_k`: the score function of the softmax policy. -/
def score (π : Fin n → ℝ) (i k : Fin n) : ℝ := (if i = k then 1 else 0) - π k

/-- The expected score is zero: `Σ_i π_i (δ_{ik} - π_k) = π_k - π_k`. -/
theorem expected_score_zero {π : Fin n → ℝ} (hπ : ∑ i, π i = 1) (k : Fin n) :
    ∑ i, π i * score π i k = 0 := by
  simp only [score, mul_sub, Finset.sum_sub_distrib, mul_ite, mul_one, mul_zero, Finset.sum_ite_eq',
    Finset.mem_univ, ite_true, ← Finset.sum_mul, hπ, one_mul, sub_self]

/-- **A baseline that does not depend on the action does not change the expected gradient.** -/
theorem baseline_unbiased {π : Fin n → ℝ} (hπ : ∑ i, π i = 1) (G : Fin n → ℝ) (b : ℝ) (k : Fin n) :
    ∑ i, π i * ((G i - b) * score π i k) = ∑ i, π i * (G i * score π i k) := by
  have h0 : ∑ i, π i * (b * score π i k) = 0 := by
    have : ∀ i, π i * (b * score π i k) = b * (π i * score π i k) := fun i => by ring
    simp_rw [this]
    rw [← Finset.mul_sum, expected_score_zero hπ k, mul_zero]
  have : ∀ i, π i * ((G i - b) * score π i k) = π i * (G i * score π i k) - π i * (b * score π i k) :=
    fun i => by ring
  simp_rw [this]
  rw [Finset.sum_sub_distrib, h0, sub_zero]

/-- The same for the softmax policy itself. -/
theorem softmax_baseline_unbiased [NeZero n] (z : Fin n → ℝ) (G : Fin n → ℝ) (b : ℝ) (k : Fin n) :
    ∑ i, softmax z i * ((G i - b) * score (softmax z) i k) =
      ∑ i, softmax z i * (G i * score (softmax z) i k) :=
  baseline_unbiased (softmax_sum z) G b k

/-- A state-dependent baseline is a constant for each state, so the same holds per state: with a
baseline `b s` in state `s`, the expected gradient in every state is unchanged. -/
theorem state_baseline_unbiased {S : Type*} (π : S → Fin n → ℝ) (hπ : ∀ s, ∑ i, π s i = 1)
    (G : S → Fin n → ℝ) (b : S → ℝ) (s : S) (k : Fin n) :
    ∑ i, π s i * ((G s i - b s) * score (π s) i k) = ∑ i, π s i * (G s i * score (π s) i k) :=
  baseline_unbiased (hπ s) (G s) (b s) k

end RLVerify.Baseline
