# Assignment 1: Reinforcement Learning

Code for the three parts of MLDS Assignment 1 (due Oct 26, 2026, 11:59 pm), plus a dashboard that connects
to deepdish, runs the code there, and collects the plots, validation results and discussion answers for the report.

| Part | Code | Points |
|---|---|---|
| 2. Value and Policy Iteration (shuttle dispatching) | `rl/dp/` | 30 |
| 3. Policy Gradient: Part 1 (CartPole-v1, Pong-v5) | `rl/pg/` | 40 |
| Policy Gradient: Part 2 (with a baseline) | `rl/pg/` | 20 |

> **Status:** setup, dashboard, test harness and report builder are in place. The algorithm code is still
> TODO stubs (see [Where your code goes](#5-where-your-code-goes)), the test files are empty, and the report
> has no answers yet.

```
run.py               single entry point: python run.py {dp,pg,tests,report,check,ui}
setup.sh             one-time setup (laptop or deepdish)
config/deepdish.json server names, project folder on deepdish, upload exclusions
rl/                  assignment code: dp/ (Section 2), pg/ (Parts 1 and 2), plotting, output paths
tests/               validation-experiment harness + one test file per part
ui/                  Streamlit dashboard and the deepdish connection code
scripts/jobs.sh      background job runner (single GPU), used by the dashboard and from the terminal
report/              questions from the handout, saved answers, report builder, report.md
results/             everything the runs produce (figures and JSON are committed; checkpoints are not)
```

## 1. Setup (once, after cloning)

```bash
git clone <this repository> && cd RL
./setup.sh                    # or: bash setup.sh
source .venv/bin/activate
python run.py ui              # dashboard at http://localhost:8501
```

`setup.sh` on your computer:

1. creates `.venv` with the newest Python ≥ 3.9 it finds and installs `requirements.txt` and `requirements-ui.txt`;
2. sets up SSH for deepdish, asking before each change to `~/.ssh/config` (and backing it up first):
   - `deepdish1` … `deepdish5` shortcuts for `mlds-deepdishN.ads.northwestern.edu` with your NetID, unless you already have them;
   - **connection sharing** (`ControlMaster`), so the dashboard can reuse SSH sessions you open from a terminal or VS Code;
   - an SSH key, copied to deepdish with `ssh-copy-id` if key login does not work yet;
3. runs `python run.py check`.

Options: `--server` (on deepdish: no dashboard, no SSH steps), `--no-ssh`, `--python /path/to/python3`, `-y`.

## 2. Connecting to deepdish

You need the campus network or the Northwestern VPN (GlobalProtect). Any of these counts as connected, and the
dashboard sidebar shows a green dot next to that server:

| How you connect | What the sidebar shows |
|---|---|
| **Connect** button in the dashboard sidebar | `shared SSH connection open` |
| `ssh deepdish3` in a terminal | `terminal ssh` (and `shared SSH connection open` with connection sharing on) |
| VS Code **Remote-SSH** to host `deepdish3` | `VS Code Remote-SSH` (likewise) |
| Dashboard started on deepdish itself (see below) | `dashboard is running on this server` |

How it works: the dashboard checks for an open SSH control socket for each server (`ssh -O check`), scans this
computer's processes for `ssh` sessions to `deepdish1`…`deepdish5` (telling terminal sessions from VS Code by the
parent process), and compares the machine's own hostname. Press **Refresh status** after connecting elsewhere; the
status is cached for 10 seconds.

- **Any of the five servers works.** Pick one in the sidebar; **Make deepdishN the default** remembers it.
- **SSH key or password.** With a key, **Connect** needs nothing else. Without one, type your NetID password in the
  sidebar: it is used once to open the shared connection (valid 4 hours) and is never saved.
- **Running the dashboard on deepdish.** In VS Code, connect with Remote-SSH, open the project folder there, and run
  `python run.py ui`. VS Code forwards port 8501 to your browser. Uploading is then unnecessary and jobs run in place.

## 3. Running code on deepdish

In the dashboard's **Run** tab, with a deepdish server selected:

1. **Upload code** copies this project to the folder on deepdish (`~/RL` by default). The server's `results/`,
   `logs/` and `.venv/` are never overwritten.
2. First time only: launch the job **Install the Python environment** (`setup.sh --server`). deepdish's GPUs are
   GTX 1080s (compute capability 6.1), which current default PyTorch CUDA builds no longer support, so the script
   detects this and installs the newest CUDA 12.6 build of PyTorch (`torch<2.15`) instead.
3. Launch **Environment check**: it should end with `ready` and `test computation on the GPU: ok`.
4. **Show GPUs** and pick one card. The handout requires a **single GPU** on deepdish: each job sees only the chosen
   card through `CUDA_VISIBLE_DEVICES`.
5. Pick a job (Section 2, Part 1/2 × CartPole/Pong, validation tests), optionally add a run tag or extra arguments,
   and press **Launch**. Jobs keep running after you disconnect or close the dashboard.
6. Under **Jobs**, follow the log (turn on auto-refresh) and, for policy-gradient runs, the live training curve.
7. **Download results** copies `results/` back, and the plots appear in the section tabs.

The same from a terminal:

```bash
rsync -az --exclude .git --exclude .venv --exclude results --exclude logs ./ deepdish3:RL/   # or git clone there
ssh deepdish3
cd ~/RL && ./setup.sh --server
scripts/jobs.sh start part1_cartpole 0 python run.py pg --part 1 --env cartpole   # job name, GPU index, command
scripts/jobs.sh status                  # name, state, exit code, pid, gpu, start time, host, command
scripts/jobs.sh tail part1_cartpole 50
scripts/jobs.sh stop part1_cartpole
```

On your laptop, choose **Run on: this computer** to run the same jobs locally (CPU).

## 4. Commands

| Command | What it does | Output |
|---|---|---|
| `python run.py dp [--method enumeration\|value_iteration\|policy_iteration]` | Section 2 (a)–(c) | `results/dp/` |
| `python run.py pg --part 1 --env cartpole` | Part 1 (a); `--env pong` for (b) | `results/pg/part1_cartpole/` |
| `python run.py pg --part 2 --env pong --baseline <name>` | Part 2 with a baseline | `results/pg/part2_pong/` |
| `python run.py tests --section dp\|pg1\|pg2\|all [-k text] [--list]` | validation experiments | `results/tests/<section>.json` |
| `python run.py report` | rebuild the report | `report/report.md` |
| `python run.py check` | packages, GPU, CartPole-v1 and Pong-v5 | console |
| `python run.py ui [--port 8501]` | dashboard | browser |

`pg` options: `--episodes`, `--lr` (default 0.01, as in the handout), `--gamma` (default 0.95 CartPole / 0.99 Pong),
`--eval-episodes` (default 500), `--seed`, `--device auto|cpu|cuda`, `--tag NAME` (keeps runs side by side in
`results/pg/part1_cartpole_NAME/`), `--eval-only` (re-evaluate the saved `policy.pt`), and `--set KEY=VALUE`
(repeatable; available to your code as `cfg.extra`).

A policy-gradient run writes `config.json`, `progress.jsonl` (one line per episode, read by the live curve),
`train.json`, `policy.pt`, `eval.json` (500 rewards, mean, std), `training_curve.png` and `eval_histogram.png`.

## 5. Where your code goes

Each stub raises `NotImplementedError`; the commands above report it as a TODO.

| File | To implement | What the rest of the pipeline expects |
|---|---|---|
| `rl/dp/shuttle.py` | `build_mdp(params)` | any structure your solvers accept; `ShuttleParams` holds the handout's constants |
| `rl/dp/solvers.py` | `enumeration`, `value_iteration`, `policy_iteration` | a `SolverResult` with `states` and `values` (or `policy`) |
| `rl/pg/policy.py` | `build_policy(cfg, env)` | a `torch.nn.Module` (saved as `policy.pt`) |
| `rl/pg/reinforce.py` | `train(...)`, `evaluate(...)` | `train` returns one reward per training episode and calls `logger.log(episode, reward)`; `evaluate` returns the rewards of the evaluation episodes |
| `rl/pg/baselines.py` | `make_baseline(...)`, `DEFAULT_BASELINE` | whatever object your `train` uses |

`rl/pg/envs.py` already creates `CartPole-v1` and `ALE/Pong-v5` and contains the handout's `preprocess` function.
`rl/plotting.py` draws the required figures with consistent titles, axis labels and legends.

## 6. Validation-test harness

The handout asks, for each part, for experiments that show the algorithm behaves as predicted, documented with
*what each test does and how it validates*, plus quantitative evidence. Write each experiment as a function in
`tests/test_dp.py`, `tests/test_pg_part1.py` or `tests/test_pg_part2.py`:

```python
from tests.harness import validation_test

@validation_test(
    section="dp",                      # dp, pg1 or pg2
    title="Short name of the experiment",
    what="What the experiment changes or measures.",
    how="Why the outcome shows the implementation is correct, and what outcome is expected.",
)
def my_experiment(ctx):
    ...
    ctx.metric("some quantity", value)
    ctx.check("expected behaviour holds", condition, detail)
```

| `ctx` method | Records |
|---|---|
| `check(name, condition, detail="")` | a pass/fail assertion (the test keeps running) |
| `check_close(name, actual, expected, tol)` | a check on the max absolute difference of numbers or arrays |
| `metric(name, value)` | a number for the report |
| `table(name, rows)` | a list of dicts or a DataFrame, e.g. a step-by-step trace |
| `figure(fig_or_path, name)` | a matplotlib figure or an image file |
| `note(text)` | a free-form observation |

Run with `python run.py tests --section dp` (or the dashboard's *Validation tests* jobs, for long experiments on
deepdish). Each test ends as **passed**, **failed** (a check failed), **error** (it raised) or **recorded** (no
checks, evidence only). Results go to `results/tests/<section>.json`, with figures in `results/tests/<section>/<test>/`.
The section tabs show them, and the report includes a table of each test's description, validation and result.

## 7. Report

Each section tab lists that part's questions from the handout, with its plots, evaluation statistics and validation
results, and an answer box. Answers save to `report/answers.json` when you click outside the box. In the **Report**
tab, fill in the submission details (name, GitHub username, repository name and link), then **Build report** to
regenerate `report/report.md` (or run `python run.py report`). To make the PDF for Canvas:

```bash
cd report && pandoc report.md -o report.pdf
```

The questions themselves live in `report/questions.py`.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| Connect times out / "Cannot reach" | Connect to the campus network or GlobalProtect VPN. |
| "did not accept an SSH key" | Type your NetID password and press Connect, or rerun `./setup.sh` to copy your key. |
| Status doesn't show a session you just opened | Press **Refresh status** (status is cached for 10 s). |
| `no kernel image is available` / GPU test fails | The torch build doesn't support the GPU: rerun `./setup.sh --server` on deepdish, or set `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126`. |
| "already running" when launching | A job with that name is running; stop it or set a run tag. |
| Port 8501 in use | `python run.py ui --port 8502` |
