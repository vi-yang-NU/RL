"""Every question and deliverable in the assignment handout, in report order.

The dashboard shows each one with an answer box; report/build.py turns them into report.md.
`figures` names entries of rl.outputs.FIGURES shown with the question, `evals` names
policy-gradient runs whose evaluation mean and standard deviation belong with it,
`tests` names the harness section whose results belong with it, and `source` names a
Markdown file in report/ that holds the answer instead of the dashboard's answer box.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    points: int | None = None
    breakdown: str = ""


@dataclass(frozen=True)
class Question:
    id: str
    section: str
    title: str
    prompt: str
    figures: tuple[str, ...] = ()
    evals: tuple[str, ...] = ()
    tests: str | None = None
    source: str | None = None


SECTIONS = [
    Section("submission", "Submission details"),
    Section("dp", "2. Value and Policy Iteration", 30, "Edge cases and tests: 15 points · Plots: 15 points"),
    Section("pg1", "3. Policy Gradient: Part 1", 40, "Edge cases and tests: 20 points · Plots: 20 points"),
    Section("pg2", "Policy Gradient: Part 2", 20, "Edge cases and tests: 10 points · Plots: 10 points"),
]
SECTION_BY_KEY = {s.key: s for s in SECTIONS}

# Asked of every part in the handout's Summary.
DESCRIPTION = "Provide a description of your solution and any corresponding plots."
EDGE_CASES = (
    "What edge cases may arise from the problem and how did you handle them? Document the edge cases of the "
    "problem that you considered and explain how you ensured that your solution has handled them correctly."
)
VALIDATION = (
    "Beyond asking AI, how do you know that your algorithm is working correctly? Test whether your implementation "
    "is behaving as predicted. For example, you may change certain parameters or the structure of the problem and "
    "see if the output matches your expectation. You may also reduce the stochasticity of the environment or policy "
    "so that the process is close to deterministic. You may also find it useful to isolate the first 10 steps and "
    "observe how the states, values, rewards, and policy interact at each step. Provide directions and documentation "
    "for how to execute validation and test experiments (what does each test do, how does it validate, etc). Provide "
    "quantitative and convincing evidence. (Not unit tests or basic bug fixes, such as checking the data format.)"
)
EXTENDED = "Given an additional 3 months (extended period), what other validation tests and experiments would you perform?"
PG_PLOTS = (
    "Plot the episode reward against the number of training episodes, overlaid with the moving average of the "
    "rewards over 100 episodes. When you finish training, roll out 500 episodes using your trained model. Plot the "
    "histogram of the 500 episode rewards and report the mean and the standard deviation."
)
PONG = (
    "Use frames of the game as input to a neural net which models the policy. The game has six actions [NOOP, FIRE, "
    "RIGHT, LEFT, RIGHTFIRE, LEFTFIRE], but you should train a network that chooses the best action only between "
    "[RIGHT, LEFT] (actions 2 and 3). You may pre-process the frames with the provided preprocess function. Use the "
    "specified environment version (Pong-v5); on deepdish, use a single GPU card."
)


def _shared(section: str, validation_figures: tuple[str, ...] = ()) -> list[Question]:
    """The edge-case, validation and extended-validation questions asked of every part; `validation_figures`
    are shown with the validation question, above the harness's results."""
    return [
        Question(f"{section}_edge_cases", section, "Edge cases", EDGE_CASES),
        Question(f"{section}_validation", section, "Tests for validating the correctness of your algorithm", VALIDATION,
                 figures=validation_figures, tests=section),
        Question(f"{section}_extended", section, "Additional validation with 3 more months", EXTENDED),
    ]


QUESTIONS: list[Question] = [
    Question("how_to_run", "submission", "Instructions for running the code", "Instructions of how to run your code.",
             source="instructions.md"),
    # Section 2
    Question("dp_description", "dp", "Solution overview", DESCRIPTION),
    Question(
        "dp_model", "dp", "MDP formulation",
        "Model the shuttle dispatching problem as a Markov Decision Process, given K = 15, c_f = 100, c_h = 2 and "
        "A_t ~ Unif{1, 2, 3, 4, 5}. The number of people in the station cannot exceed 200; use discount rate γ = 0.95.",
    ),
    Question(
        "dp_enumeration", "dp", "(a) Enumeration, T = 500",
        "Solve using enumeration (with time period T = 500). Plot the optimal value function at time 0 versus number "
        "of customers waiting.",
        figures=("dp_enumeration",),
    ),
    Question(
        "dp_value_iteration", "dp", "(b) Value iteration, T = ∞",
        "Solve using value iteration (T = ∞). Plot optimal value function versus number of customers waiting.",
        figures=("dp_value_iteration",),
    ),
    Question(
        "dp_policy_iteration", "dp", "(c) Policy iteration",
        "Solve using policy iteration. Plot optimal policy versus number of customers waiting.",
        figures=("dp_policy_iteration",),
    ),
    Question("dp_assumptions", "dp", "Assumptions", "If you make any other assumption, please state it in your report."),
    *_shared("dp", validation_figures=("dp_tests_table",)),
    # Policy Gradient Part 1
    Question(
        "pg1_description", "pg1", "Solution overview",
        "Using PyTorch, implement the basic policy gradient algorithm: train a simple neural network that models the "
        "policy and takes actions based on the provided states. Set the discounted reward over an episode as your "
        "objective function and use Adam as your gradient optimizer with learning rate 0.01 (if you use a different "
        "learning rate because training is more stable, mention it). " + DESCRIPTION,
    ),
    Question(
        "pg1_cartpole", "pg1", "(a) CartPole-v1, discount factor 0.95", PG_PLOTS,
        figures=("part1_cartpole_train", "part1_cartpole_eval"), evals=("part1_cartpole",),
    ),
    Question(
        "pg1_pong", "pg1", "(b) Pong-v5, discount factor 0.99", f"{PONG} {PG_PLOTS}",
        figures=("part1_pong_train", "part1_pong_eval"), evals=("part1_pong",),
    ),
    *_shared("pg1", validation_figures=("part1_cartpole_steps", "part1_pong_steps")),
    # Policy Gradient Part 2
    Question(
        "pg2_baseline", "pg2", "Baseline: choice and implementation",
        "The training suffers from high variance of the rewards in the sampled episodes; using a baseline helps to "
        "reduce the variance. Repeat the experiments of Part 1 using a constant or non-constant baseline of your "
        "choice. Explain which baseline you have chosen and how you implemented it.",
    ),
    Question("pg2_description", "pg2", "Solution overview", DESCRIPTION),
    Question(
        "pg2_cartpole", "pg2", "(a) CartPole-v1 with baseline", f"Repeat Part 1 (a) with your baseline. {PG_PLOTS}",
        figures=("part2_cartpole_train", "part2_cartpole_eval"), evals=("part2_cartpole",),
    ),
    Question(
        "pg2_pong", "pg2", "(b) Pong-v5 with baseline", f"Repeat Part 1 (b) with your baseline. {PG_PLOTS}",
        figures=("part2_pong_train", "part2_pong_eval"), evals=("part2_pong",),
    ),
    *_shared("pg2", validation_figures=("part2_cartpole_steps", "part2_pong_steps")),
]
QUESTION_BY_ID = {q.id: q for q in QUESTIONS}


def questions_for(section: str) -> list[Question]:
    return [q for q in QUESTIONS if q.section == section]
