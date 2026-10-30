# Lean proofs for the assignment's algorithms

This folder is a Lean 4 project (Mathlib) that states the models the Python code implements over exact
rationals and proves the properties the solvers and the tests rely on. Nothing in the Python code needs Lean:
the one artefact Python reads, `reference/shuttle_reference.json`, is committed.

## What is proved

**`RLVerify/Shuttle.lean`**: the shuttle dispatching MDP (`Params`: capacity, costs, arrival distribution,
cap on the queue, γ) with `left`, `next`, `cost`, `expect`, `Q`, the Bellman operator `T`, the policy operator
`Tpi`, and `value n s` (optimal expected discounted cost with `n` periods to go, i.e. enumeration / backward
induction). Under `Params.Valid` (non-negative costs, a probability distribution of arrivals, 0 ≤ γ < 1):

| Theorem | Statement | Where Python relies on it |
|---|---|---|
| `next_le_max` | the next state never exceeds the cap | `build_mdp`: the state space 0..200 is closed |
| `cost_nonneg`, `value_nonneg` | costs and optimal values are ≥ 0 | sanity checks in the tests |
| `value_mono_horizon` | `value n s ≤ value (n+1) s` | "first 10 steps" experiment |
| `value_mono_state` | `value n` is monotone in the state | monotone value plots; tests |
| `bellman_contraction` | `|V − W| ≤ d` on the state space ⇒ `|T V − T W| ≤ γ d` | value iteration converges |
| `value_succ_sub_bound` | `|V_{n+1} − V_n| ≤ γ^n |V_1 − V_0|` | residual ratios ≤ γ (contraction experiment) |
| `vi_error_bound` | `|T V − V| ≤ d` and `T V* = V*` ⇒ `|T V − V*| ≤ γ d / (1 − γ)` | `info["error_bound"]` of `value_iteration` and the stopping rule |
| `T_le_Tpi`, `policy_improvement` | the greedy policy for `V^π` is no worse than π | the improvement step of `policy_iteration` |
| `Q_policy_eq_T` | the greedy action attains the minimum | tie-breaking of the exported first actions |

**`RLVerify/Returns.lean`**: `discounted γ l = Σ_k γ^k l[k]` (`discounted_eq_sum`), and the backward recursion
`returnsBackward` (what `rl.pg.reinforce.discounted_returns` implements) satisfies
`(returnsBackward γ l)[t] = discounted γ (l.drop t)` (`returnsBackward_getElem`), hence
`G_t = Σ_k γ^k r_{t+k}` (`returnsBackward_getElem_eq_sum`); also the γ = 0 and γ = 1 special cases.

**`RLVerify/Baseline.lean`**: the softmax is a probability vector (`softmax_pos`, `softmax_sum`); with the score
`δ_{ik} − π_k` of a softmax, the expected score is zero (`expected_score_zero`), so subtracting any baseline that
does not depend on the action leaves the expected gradient unchanged (`baseline_unbiased`,
`softmax_baseline_unbiased`, `state_baseline_unbiased`). This is why Part 2 may subtract a baseline. The closed
form of the score is the textbook derivative of `log softmax`; `tests/unit/test_pg_train.py` checks PyTorch's
gradient against it numerically.

**`ShuttleRef.lean`** (`lake env lean --run ShuttleRef.lean`) evaluates `value` on four small instances with exact rationals and
prints `reference/shuttle_reference.json`: values as exact fractions for every horizon and state,
plus the optimal first actions. `tests/unit/test_dp_lean_reference.py` and the `dp` validation experiment
"Exact rational arithmetic agrees with the floating-point solvers" compare `rl.dp.solvers.enumeration` with it.
The instances are small because `value` is the plain recursive specification (no memoisation) and the script runs in Lean's interpreter (no native build of Mathlib is needed).

The formal model has the handout's two actions (dispatch or not) and holding charged after the dispatch; the
Python code's extra structural switches (several shuttles per period, other holding-cost timings) are outside it.

## Building (optional; only to re-check the proofs or regenerate the reference)

1. Install `elan`, the Lean version manager (<https://github.com/leanprover/elan>). `lean-toolchain` pins
   Lean `v4.35.0`; elan installs it on first use.
2. From this folder:

   ```bash
   lake exe cache get      # downloads Mathlib's prebuilt files (several GB, once)
   lake build              # checks every proof; a clean build prints no errors
   lake env lean --run ShuttleRef.lean > reference/shuttle_reference.json
   ```

`lakefile.toml` pins Mathlib to the tag `v4.35.0`. The `.lake/` folder (dependencies and build products) is
ignored by git.
