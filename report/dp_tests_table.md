<!-- Section 2 validation-test summary, one row per test. `python scripts/tests_table.py` renders this table to
results/dp/tests_table.png, the figure shown above the harness's results in the report and the dashboard.
A `\|` is a literal bar inside a cell. -->

| # | Test | What it checks | How it validates | Pass criterion | Result |
|---|---|---|---|---|---|
| T1 | Model sanity | The transition and cost arrays are built correctly | Rows sum to 1; cap behaviour; costs checked against hand calculations | Exact | Row-sum error 0; P[200, hold, 200] = 1; P[198, hold, 200] = 0.8; C[20, hold] = 40; C[20, dispatch] = 110 |
| T2 | Hand-computed backups | The Bellman backup is coded correctly | Compare with values worked out on paper | Equal to 0.01 | 1 period left: V(s) = 2s. 2 left: V(20) = 83.70 (hold), dispatch = 125.20 |
| T3 | Bellman residual | Each method’s answer satisfies the optimality equation | max over s of \|V − min(C + 0.95·P·V)\| | ≤ 10⁻⁶ | Enumeration 2×10⁻¹⁰, VI 9×10⁻⁷, PI 5×10⁻¹³ |
| T4 | Cross-method agreement | All three methods reach the same answer | Largest difference in V; number of states where the policies differ | ≤ 10⁻⁴; 0 states | Max gap 1.8×10⁻⁵; 0 disagreements; all thresholds 13 |
| T5 | PI monotone improvement | Policy improvement never makes a policy worse | V never increases in any state from one round to the next; try several starting policies | Increase ≤ 0; same final policy | V(0): 2,204 → 1,330 → 891 → 749 → 582 → 577. Starts from never, always and threshold 50 all end at threshold 13 |
| T6 | VI contraction | Each value-iteration sweep shrinks the error | Ratio of successive changes ≤ γ; actual error stays within γ·δ/(1−γ) | Ratio ≤ 0.95; error/bound ≤ 1 | Max ratio 0.9500; max error/bound 0.99999 |
| T7–T8 | Monte Carlo | The computed values match what actually happens | Simulate the optimal policy 20,000 times (400 steps each) and compare the mean discounted cost with V(s) | Exact value inside the 95% CI | s = 0: 576.67 vs 576.67 ± 0.43. s = 13, 100, 200 also inside |
| T9 | Optimal vs alternatives | The policy beats every reasonable competitor | Exactly evaluate 42 other policies (thresholds 1–40, never, always) | No policy cheaper in any state | Best is threshold 13; thresholds 12 and 14 cost 581.7 and 581.6 vs 576.7 |
| T10 | Structure | The answer has the shape theory predicts | V nondecreasing in s; policy monotone; V flat on 13–15 | All true | True, true; V(13) = V(14) = V(15) = 676.67 |
| T11 | Known-answer edge cases | The solver gives the right answer when the answer is obvious | c_h = 0, c_f = 0, γ = 0, deterministic arrivals | Matches the predicted answer | c_h = 0: never dispatch, V = 0. c_f = 0: dispatch at ≥ 1. γ = 0: V = 2s. Always-dispatch cost = 100/0.05 = 2,000 exactly |
| T12 | Sensitivity | The policy moves in the right direction when a parameter changes | Re-solve with each parameter changed | Direction matches intuition | c_f 50/100/200 → threshold 11/13/14. c_h 1/4 → 14/11. K 10/30 → 9/17 |
| T13 | Horizon and terminal value | T = 500 is effectively infinite | Gap to V* for several T; a $1 trillion terminal value | Gap ≈ 0 at T = 500; shift = 0.95⁵⁰⁰·terminal | T = 30: 146. T = 100: 4.0. T = 500: 5×10⁻⁹. Trillion shift 7.2745, predicted 7.2745 |
