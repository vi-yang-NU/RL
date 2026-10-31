# Assignment 1: Reinforcement Learning

Code for the three parts of MLDS Assignment 1 (due Oct 26, 2026, 11:59 pm), plus a dashboard that connects
to deepdish, runs the code there, and collects the plots, validation results and discussion answers for the report.

| Part | Code | Points |
|---|---|---|
| 2. Value and Policy Iteration (shuttle dispatching) | `rl/dp/` | 30 |
| 3. Policy Gradient: Part 1 (CartPole-v1, Pong-v5) | `rl/pg/` | 40 |
| Policy Gradient: Part 2 (with a baseline) | `rl/pg/` | 20 |

> **Status:** all algorithms are implemented (Section 2 solvers; REINFORCE with and without baselines), with
> unit tests (`python run.py unittest`), validation experiments for each part (`python run.py tests`) and Lean
> proofs of the properties they rely on (`lean/`, see [section 7](#7-formal-verification-with-lean)). The Part 1
> runs in `results/pg/` were made on a laptop CPU. CartPole-v1 reaches the 500-step cap with the handout's settings.
> Pong-v5 with the handout's Adam learning rate 0.01 collapsed onto one action within about 150 episodes and lost
> every game 0 : 21 (that run is kept as `results/pg/part1_pong_lr0.01/`), so Pong now trains with learning rate
> 10⁻⁴ and an entropy bonus of 0.01 per step, and every run logs the policy's entropy and its fraction of RIGHT
> actions per episode (lower panel of the training curve, live chart in the dashboard). Still to do: the Pong runs
> with these settings and the Part 2 runs on deepdish, and the written answers in the report.

```
run.py               single entry point: python run.py {dp,pg,tests,report,check,ui,save}
setup.sh             one-time setup (laptop or deepdish)
config/deepdish.json server names, project folder on deepdish, upload exclusions
.env.example         your deepdish settings (NetID, default server, key) go in .env, which git ignores; see section 2
rl/                  assignment code: dp/ (Section 2), pg/ (Parts 1 and 2), plotting, output paths, --set overrides
tests/               validation-experiment harness + one experiment file per part; tests/unit/ = unit tests
lean/                Lean 4 proofs about the algorithms and the exact reference values they produce
ui/                  Streamlit dashboard and the deepdish connection code
scripts/jobs.sh      background job runner (single GPU), used by the dashboard and from the terminal
scripts/explore_collage.py  the report's collage of the explorer's first ten periods (headless Chrome)
scripts/tests_table.py      the report's pictures of the validation-test summary tables (Section 2, Parts 1 and 2; headless Chrome)
report/              questions from the handout, saved answers, run instructions, report builder, report.md/.pdf
results/             everything the runs produce (figures and JSON are committed; checkpoints are not)
```

## 1. Setup (once, after cloning)

```bash
git clone <this repository> && cd RL
./setup.sh                    # or: bash setup.sh
source .venv/bin/activate
python run.py ui              # opens the dashboard at http://localhost:8501
```

`setup.sh` on your computer:

1. creates `.venv` with the newest Python ≥ 3.9 it finds and installs `requirements.txt` and `requirements-ui.txt`;
2. sets up SSH for deepdish, asking before each change to `~/.ssh/config` (and backing it up first):
   - `deepdish1` … `deepdish5` shortcuts for `mlds-deepdishN.ads.northwestern.edu` with your NetID (saved to `.env`), unless you already have them;
   - **connection sharing** (`ControlMaster`), so the dashboard can reuse SSH sessions you open from a terminal or VS Code;
   - an SSH key, copied to deepdish with `ssh-copy-id` if key login does not work yet;
3. runs `python run.py check`.

Options: `--server` (on deepdish: no dashboard, no SSH steps), `--no-ssh`, `--python /path/to/python3`,
`--venv-dir DIR`, `-y`. On deepdish the home directory is on NFS, from which `import torch` alone takes about
150 s per process, so `--server` creates the environment on the local disk (`/tmp/$USER/RL-venv`) and makes
`.venv` a symlink to it; `/tmp` survives logouts, and if a reboot clears it, rerun `./setup.sh --server`.

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
- **Your settings live in `.env`** (git ignores it; `.env.example` lists every variable): `DEEPDISH_USER` (NetID),
  `DEEPDISH_DEFAULT_SERVER`, `DEEPDISH_REMOTE_DIR`, and `DEEPDISH_IDENTITY_FILE` for a machine without the
  `~/.ssh/config` shortcuts. `setup.sh` and the sidebar write it. A `DEEPDISH_*` variable set in the environment
  (a VS Code launch configuration, a CI secret, a cloud session's environment settings) overrides `.env`, so any
  tool that runs commands in this folder connects the same way as the dashboard. Passwords are never stored.
- **Without the dashboard:** `python run.py deepdish` prints the settings in use, where each one comes from, how
  ssh reaches the chosen server and every server's status; `python run.py deepdish --connect` opens the shared
  connection from a terminal (your key, or the NetID password at the prompt), after which `ssh deepdish2`, `rsync`
  and the dashboard reuse it for 4 hours.

## 3. Running code on deepdish

The sidebar (hide it with **«** at its top, bring it back with **»** at the top left) holds the connection and
**Jobs run on**: the server or this computer, and the GPU. The handout requires a **single GPU** on deepdish: each
job sees only the chosen card through `CUDA_VISIBLE_DEVICES`. **Show GPU usage** lists each card's memory use.

Once per server, under **Set up deepdish** in the sidebar (shown when a deepdish server is selected under
*Jobs run on*):

1. **Upload code** copies this project to the folder on deepdish (`~/RL` by default). The server's `results/`,
   `logs/` and `.venv/` are never overwritten.
2. **Install** runs `setup.sh --server` there (uploading the code first). deepdish's GPUs are GTX 1080s (compute
   capability 6.1), which current default PyTorch CUDA builds no longer support, so the script detects this and
   installs the newest CUDA 12.6 build of PyTorch (`torch<2.15`) instead.
3. **Check** runs `python run.py check` there: it should end with `ready` and `test computation on the GPU: ok`.

Then each part's tab (Section 2, Part 1, Part 2) has its own **Run** panel above the questions:

1. Pick the job (e.g. *(a) CartPole-v1*, *(b) Pong-v5* or *Validation tests*), optionally add a run tag or extra
   arguments, and press **Launch**. *Upload code first* (on by default) sends your latest code before launching.
   Jobs keep running after you disconnect or close the dashboard.
2. Under **Jobs** (that part's jobs only), follow the log (turn on auto-refresh) and, for policy-gradient runs, the
   live training curve.
3. **Download results** copies `results/` back, and the plots below update.

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

On your laptop, choose **Jobs run on: this computer** to run the same jobs locally (CPU).

## 4. Commands

| Command | What it does | Output |
|---|---|---|
| `python run.py dp [--method enumeration\|value_iteration\|policy_iteration] [--set KEY=VALUE] [--tag NAME] [--backend numpy\|cuda] [--explorer FILE]` | Section 2 (a)–(c); `--set` changes any `ShuttleParams` field; `--tag` keeps that run (figures included) in `results/dp_NAME/` so the handout's results stay; `--backend cuda` uses the kernels in `rl/cuda`; `--explorer` also writes the interactive explorer page (below) for those parameters | `results/dp/` or `results/dp_NAME/` |
| `python run.py pg --part 1 --env cartpole` | Part 1 (a); `--env pong` for (b) | `results/pg/part1_cartpole/` |
| `python run.py pg --part 2 --env pong [--baseline value\|mean\|time]` | Part 2 with a baseline (default `value`) | `results/pg/part2_pong/` |
| `python run.py pg --part 2 --env cartpole --compare` | Part 1 (no baseline) against Part 2 (baseline) from the two finished runs' logs: rewards and the size of the weights on ∇ log π; drawn automatically after either run | `results/pg/compare_cartpole.png` |
| `python run.py pg --part 2 --env cartpole --replot` | redraw a finished run's figures (training curve, histogram, first-steps figure, comparison) from its saved numbers, without training | the run's folder |
| `python run.py play --env cartpole\|pong [--run part1_pong\|--untrained] [--speed X] [--greedy] [--seed N]` | the real game live in a window with the learned policy at the controls; the arrow keys take over (Pong: the paddle; CartPole: the cart), A hands back, Space pauses, R restarts, G toggles greedy actions, +/− change the speed, Q quits | a window |
| `python run.py watch --env cartpole\|pong [--run part1_pong] [--seed N] [--steps N] [--greedy] [--out FILE] [--save FILE.npz] [--figure FILE.png]` | one episode of the real environment, frame by frame, as the viewer page the dashboard's **Watch the game** panel shows (below); `--run` plays with a finished run's `policy.pt`, otherwise an untrained policy plays | `results/pg/watch_<env>.html` |
| `python run.py tests --section dp\|pg1\|pg2\|all [-k text] [--list]` | validation experiments | `results/tests/<section>.json` |
| `python run.py unittest [-k text] [-v]` | unit tests and edge cases (about 130 tests; a few seconds, plus about a minute for the CUDA kernels when they run in the simulator) | console |
| `python run.py report` | rebuild the report, Markdown and PDF | `report/report.md`, `report.pdf` |
| `python run.py check` | packages, GPU, CartPole-v1 and Pong-v5 (creation and frame rendering) | console |
| `python run.py ui [--port 8501]` | dashboard, on this computer only (localhost); opens your browser | browser |
| `python run.py deepdish [-n N] [--connect\|--disconnect]` | your deepdish settings (`.env`, `DEEPDISH_*` variables) with where each comes from, the ssh target and every server's connection status; `--connect` opens the shared SSH connection without the dashboard | console |
| `python run.py save [-m "message"] [--no-push] [--no-report]` | commit every change and push it to GitHub in one step; rebuilds the report first when the answers or results are newer than it, and pulls GitHub's newer commits before pushing | the repository on GitHub |

`pg` options: `--episodes`, `--lr` (default 0.01 for CartPole, as in the handout; 10⁻⁴ for Pong, where 0.01 collapsed the
policy onto one action within about 150 episodes, see `DEFAULT_LR` in `rl/pg/run.py`), `--gamma` (default 0.95 CartPole / 0.99 Pong),
`--eval-episodes` (default 500), `--seed`, `--device auto|cpu|cuda|mps` (`cuda` and `mps` fail if the GPU is not
available; the policy, the baseline network and every tensor they touch are placed on that device, and the run
prints the GPU it uses), `--tag NAME` (keeps runs side by side in
`results/pg/part1_cartpole_NAME/`), `--eval-only` (re-evaluate the saved `policy.pt`), and `--set KEY=VALUE`
(repeatable) for every tunable listed in `rl/pg/options.py`: `hidden=[128,64]`, `arch=cnn` (Pong), `frame_diff`,
`batch_episodes`, `normalize`, `reward_to_go`, `discount_grad`, `entropy_coef` (an entropy bonus per step; Pong's default is
0.01, CartPole's 0), `max_grad_norm`, `max_steps`,
`baseline_lr`, `baseline_hidden`, `baseline_epochs`, `baseline_decay`, `kernels`, `cpu_threads` (CPU runs use 2
PyTorch threads by default: the tiny per-step forward passes get slower with more), `eval_greedy`, `log_every`.
Quote lists in zsh: `--set 'hidden=[128,64]'`.

A policy-gradient run writes `config.json`, `progress.jsonl` (one line per episode, read by the live curve),
`train.json`, `policy.pt`, `eval.json` (the 500 rewards with mean, std, min, max and median, how long each episode
lasted in steps with the seconds of game time per step, and a note when every episode scored the same),
`training_curve.png` and `eval_histogram.png`, then
`episode.npz` (one episode played by the trained policy, every frame and what the network saw and did, for the
dashboard's **Watch the game** panel; about 0.3–1 MB) and `first_steps.png` (its first ten steps, a figure for the report).
Once both parts of an environment have finished, `results/pg/compare_<env>.png` sets them side by side for Part 2.

## 5. What is implemented, and how to change the problems

| Module | Contents |
|---|---|
| `rl/dp/shuttle.py` | `ShuttleParams`: every constant of the problem plus structural switches (arrival values and probabilities, shuttles per period, when the holding cost is charged), and `build_mdp`, which turns them into a generic `TabularMDP` (transition tensor `P[a, s, s']`, cost matrix, γ) |
| `rl/dp/solvers.py` | `enumeration` (finite-horizon backward induction), `value_iteration` (stops on the sup-norm change; saves the proven error bound γΔ/(1−γ)), `policy_iteration` (exact policy evaluation by a linear solve, greedy improvement that keeps the current action on ties), plus `evaluate_policy`, `simulate_policy`, `bellman_residual`, `greedy_policy`, `policy_threshold` |
| `rl/dp/explore.py`, `rl/dp/explore.html`, `rl/dp/explore_engine.js` | the interactive explorer shown in the dashboard's tab 2 (below): `explore` collects what the page shows from the real solvers, `render_html` embeds it in the page; the JavaScript only draws and replays |
| `rl/pg/policy.py` | `MLP` and `CNN` policy networks, `build_policy`, `build_value_net` |
| `rl/pg/reinforce.py` | REINFORCE: `discounted_returns`, `run_episode` (until the environment ends the episode or after `max_steps`, default 1000; Pong 10000 because a game lasts 1000-3000 steps), `policy_gradient_loss` (the pseudo-loss −Σ log π(a_t\|s_t)(G_t − b_t), reward-to-go weights, optional γ^t factor, baseline, normalisation, entropy bonus), `train`, `evaluate` |
| `rl/pg/baselines.py` | Part 2 baselines: `mean` (constant moving average of the returns), `time` (mean return per time step), `value` (state-value network fitted to the returns; the default) |
| `rl/pg/options.py` | every tunable of the policy-gradient code with its default, set with `--set KEY=VALUE` |
| `rl/pg/envs.py`, `rl/pg/toy.py` | CartPole-v1 / Pong-v5 creation (`gymnasium.make("CartPole-v1")`, `gymnasium.make("ALE/Pong-v5")`: the gymnasium ids of the handout's versions), the handout's `preprocess`, the adapters (frame differences, the RIGHT/LEFT restriction); bandit and corridor environments with known optima, used by the tests |
| `rl/pg/watch.py`, `rl/pg/watch.html`, `ui/watch.py` | the game viewer shown in the dashboard's policy-gradient tabs (below): `record_episode` plays one episode with rendering on and keeps every frame, Pong's preprocessed frames and network inputs, the probabilities, actions, rewards and reward-to-go; `save_recording`/`load_recording` (`episode.npz`), `render_html` (the page), `first_steps_figure` (the report's figure) |
| `rl/params.py` | `--set KEY=VALUE` parsing and dataclass overrides shared by `dp` and `pg` |
| `rl/save.py` | `python run.py save`: stage, commit and push in one step; rebuilds the report when it is older than the answers or results, refuses files over 50 MB, rebases onto GitHub's newer commits before pushing |
| `rl/cuda/` | CUDA versions written as explicit kernels (Numba): `backend.py` (device detection, simulator fallback, grid/block helpers), `dp.py` (state grid initialised on the device, persistent single-block value-iteration kernel with the value function in shared memory and the convergence loop inside the kernel, grid-wide sweeps for larger state spaces, greedy improvement, one thread per Monte Carlo trajectory), `pg.py` (discounted returns as a parallel suffix scan, fused Pong preprocessing, block reductions for normalisation, softmax sampling) |

Changing the parameters or the structure of a problem:

- **Section 2:** `python run.py dp --set capacity=10 --set 'arrivals=[0,1,2]' --set 'arrival_probs=[0.5,0.25,0.25]'
  --set gamma=0.9 --set max_dispatch=2 --set holding_charged=before_dispatch` (any field of `ShuttleParams`); in
  code, `ShuttleParams(...).with_(...)`. The solvers accept any `TabularMDP`, so other problems can be solved with
  the same code (the tests solve the corridor environment this way and compare REINFORCE with the exact optimum).
- **Policy gradient:** `--lr`, `--gamma`, `--episodes`, `--baseline`, and `--set` for anything in `rl/pg/options.py`.
  `rl.pg.reinforce.train` works with any environment that follows the gymnasium API and has a 1-D observation
  space (see `rl/pg/toy.py`).

`rl/plotting.py` draws the required figures with consistent titles, axis labels and legends.

### Interactive explorer (dashboard tab 2, "Explore the problem")

The **2 · Value & Policy Iteration** tab has a collapsible panel (marked, like **Run**, by the blue bar on its left) for
changing the problem and watching what the algorithms do with it. The widgets on its left set every field of `ShuttleParams` (capacity, costs, the arrival distribution,
the station limit, γ, T, shuttles per period, when the holding cost is charged) plus the value-iteration
tolerance; presets **Edge case 1–3** reproduce the report's Edge cases section (arrivals that overflow the station;
4 or 5 arrivals every period; a full platform with 5 arrivals every period, the replay starting at s = M), and the
other presets are the variations the tests use (deterministic or non-uniform arrivals, γ = 0, γ = 1, a tiny
station, cheap shuttles or expensive waiting, a 5-period horizon, two shuttles per period). On every
change the three solvers rerun (milliseconds) and the page on the right is rebuilt from their output:

- **The two stations.** Customers waiting at Station A (the state), the shuttle with its seats, and the customers
  delivered to Station B. **Step** takes the policy's action for the current period and animates it (boarding,
  the trip, the arrivals), **Play** runs the remaining periods, **Back** undoes one, and the action buttons take
  a *different* action in that period (overrides are marked in red). The summary tiles show the period, the
  state, what the policy says with the Q-values behind it, the period's cost with its breakdown, the discounted
  total so far next to what the optimal policy would have paid on the same random draws, and V*(s_t) with the
  check "cost so far + γ^t V*(s_t)" against V*(s_0).
- **Policy to replay:** the optimal policy, the policy from the algorithm panel at its current step, the exact
  finite-horizon policy (π with T − t periods to go in period t), never/always dispatch, or a custom threshold.
  The random draw for period t depends only on the seed and t (common random numbers), so changing the policy
  or overriding an action changes only what follows from it.
- **Period by period:** t, s_t, V*(s_t), Q*(s_t, ·), the suggested and taken action, the cost, its discounted
  value, the running total, the arrivals and s_{t+1}, plus Σ γ^t [Q*(s_t, a_t) − V*(s_t)], which is zero
  exactly when every taken action was optimal.
- **The algorithm's first steps:** enumeration (V_k and the decision with k periods to go), value iteration
  (V_k after k sweeps and its greedy policy; from V_0 ≡ 0 these are the same numbers as backward induction, which
  the panel points out) and policy iteration (each policy visited, its exact value, and how many states changed),
  for k = 0 … 10, with the dispatch threshold at every step.
- **Charts:** V* with the selected step's V_k or V^{π_k} over it and the current state marked; the policy bands
  (π*, the selected step's policy, the replayed policy) with thresholds; and the convergence history on a log
  scale with the first ten steps marked and the tolerance line. A values table lists V*, π*, Q* for every state.
- **Run this case** (under the preset) copies the current parameters into the Run panel at the top of the tab as
  `python run.py dp --set …`, with a run tag (`results/dp_<tag>/`, so the handout's results and figures stay), and
  scrolls up to it; **Launch** there runs the edge case for real, on this computer or deepdish. The **Edge cases**
  box lower on the tab has a button that scrolls up to the explorer.

`python run.py dp --explorer results/dp/explore.html [--set …]` writes the same page as a standalone file
(`results/dp/explore.html` holds it for the handout's parameters); opening it with `#auto` after the file name
replays all ten periods at once. `python scripts/explore_collage.py` opens that page in headless Chrome, screenshots
it at the start of each of the first ten periods and tiles them into `results/dp/explore_steps.png`, the figure the
report's Section 2 validation answer shows. The page is problem-specific only in its scene and cost breakdown; the replay,
tables and charts work for any `TabularMDP`, so the component can be reused for other problems.

### CUDA kernels

`python run.py dp --backend cuda` solves Section 2 with the kernels in `rl/cuda/dp.py`, and
`python run.py pg ... --set kernels=true` makes the policy-gradient code compute the discounted returns with the
scan kernel and the Pong frames with the preprocessing kernel (the network itself already runs on the GPU
through PyTorch). Thread-level design:

- **Grid initialisation.** `init_shuttle_tables` runs one thread per (state, action) on a 2-D grid
  (`launch_2d(201, 2)` → 7 × 1 blocks of 32 × 4 threads) and fills that pair's cost and successor table from the
  problem constants; the dense 201 × 201 transition matrix is never materialised (5 successors per pair).
- **Value iteration in one launch.** For state spaces that fit in a block (≤ 1024 states; the handout has 201),
  `sweep_persistent` keeps the value function in shared memory, each thread owns one state and applies its
  Bellman backup, the block computes max |ΔV| with a tree reduction, and the loop runs inside the kernel until
  the tolerance is met: no host round trip per iteration. Enumeration is the same kernel for exactly T sweeps,
  and policy evaluation the same kernel with the policy's action (`mode = 1`). Larger state spaces use
  `sweep_kernel`, one launch per sweep over a grid of blocks with a per-block reduction.
- **Policy improvement** is `greedy_kernel` (one thread per state, tie-breaking that keeps the current action,
  an atomic counter of changed states); **Monte Carlo** is `simulate_kernel`, one thread per trajectory with its
  own xoroshiro128+ stream (100 000 trajectories are 391 blocks of 256 threads).
- **Discounted returns** are a parallel suffix scan: the recurrence G_t = r_t + γG_{t+1} is sequential, but
  (sum, decay) pairs combine associatively, so a block scans up to 1024 steps in log₂ rounds in shared memory
  and a fix-up kernel adds the carry from the following chunk. **Pong preprocessing** is one thread per output
  pixel on a 3-D grid (frame, row, column), fused with the frame difference.

Without an NVIDIA GPU, `rl/cuda/backend.py` sets `NUMBA_ENABLE_CUDASIM=1` before importing Numba, so the same
kernels run in Numba's CPU simulator (every CUDA thread is a Python thread): correct, but hundreds of times
slower, which is how `tests/unit/test_cuda_kernels.py` checks them on a laptop. On deepdish, Numba needs the
CUDA toolkit's NVVM library (`nvidia-cuda-nvcc-cu12` from `requirements.txt`, or `CUDA_HOME`) and, with the
servers' CUDA 13 driver, NVIDIA's `cuda-python` bindings (also in `requirements.txt`; `rl/cuda/backend.py`
switches Numba to them, since Numba 0.60's own driver binding segfaults there); check with
`python -c "from numba import cuda; cuda.detect()"`.

The `dp` validation experiment "The CUDA kernels reproduce the solvers" and the kernel unit tests take about
a minute each in the simulator (seconds on a GPU); the rest of the test suites are unaffected.

### Game viewer (dashboard tabs "Policy Gradient Part 1" and "Part 2", "Watch the game")

Both policy-gradient tabs have a panel (marked by the blue bar, like **Run** and **Explore the problem**) that plays one
episode of the real gym environment step by step. On the left, choose the game (**CartPole-v1** or **Pong-v5**) and
the policy: the episode `python run.py pg` recorded at the end of the run with the trained policy (`episode.npz`,
so the trained agent can be watched on a computer that does not have the checkpoint), a checkpoint (`policy.pt`,
which plays a new episode here with the seed you choose), or an untrained policy (what training starts from); a
cap on the steps shown and greedy actions are optional. The page on the right shows, for the current step t:

- **the frame** the environment renders (for Pong the observation itself, 210×160, with the rows that `preprocess`
  keeps marked; for CartPole `env.render()` of the state, with the four state variables as bars that run to the
  limits at which the episode ends, |x| = 2.4 and |θ| = 12°);
- **for Pong, what the network sees:** `preprocess(frame)` from the handout (crop rows 35–194, every second pixel of
  the red channel, background erased, everything else 1) and the network's input, that image minus the previous
  one (blue +1 where something is now, orange −1 where it was), with the counts of non-zero pixels;
- **tiles:** the step, the action taken (for Pong also the environment action id, 2 or 3) with the policy's two
  probabilities as bars, the reward r_t, the reward-to-go G_t = Σ γ^k r (what weights log π(a_t|s_t) in the gradient)
  and the reward so far (for Pong the score);
- **controls:** Back, Step, Play/Pause (real time by default: 50 steps/s for CartPole, 15 for Pong), Next reward (Pong), End,
  a speed menu, a step slider, and the keyboard (← → space Home End);
- **two charts** over the whole episode with the current step marked, which also seek on click or drag: the reward so
  far with each point (Pong) or the cart position and pole angle as a fraction of their limits (CartPole), and
  π(a | s_t) with a strip showing the action taken at every step;
- **the first 10 steps** as a table (t, a_t, both probabilities, r_t, G_t and the state), each row a link to that step;
- **for Pong, the points:** who scored each one, at which step, how long the rally lasted in steps and game time (a
  step is 4 Atari frames, 1/15 s), and the score after it, with the average, shortest and longest rally; the Game
  time tile and the header give the same clock for the whole game (CartPole: 0.02 s per step, 10 s at the cap).

**Play it live** (the button under the widgets, or `python run.py play --env pong --run part1_pong`) opens a pygame
window on the computer running the dashboard with the real game running at its own speed (CartPole slowed to 20
steps/s so a hand can follow) and the chosen policy at the controls; the panel beside the game shows the policy's
probabilities, the action, the score and the games won and lost by the agent and by you. The arrow keys take the
controls (Pong: Up/Down move the paddle, sent as RIGHT (2) / LEFT (3); CartPole: Left/Right push the cart) and A
hands them back. Pong's opponent is the cartridge's built-in computer player: ale-py exposes a single joystick, so
you play the agent's side against the same opponent the network faces, not against the network itself.

`python run.py watch --env pong --run part1_pong` writes the same page as a standalone file (`--save` keeps the
episode as `.npz`, `--figure` draws the first-steps figure); `python run.py pg` writes `episode.npz` and
`first_steps.png` into every run's folder. Rendering CartPole needs pygame (`gymnasium[classic-control]`, in
`requirements.txt`); the code points pygame at SDL's dummy video driver, so no window opens and it works over ssh
and from the dashboard's worker threads.

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

The experiments currently registered (all run on a laptop CPU in about two minutes in total): for `dp`, agreement
of the three solvers within the proven error bound, the γ-contraction of value iteration, a step-by-step trace
of the first 10 backward steps and 10 periods of the optimal policy, exact rational arithmetic (Python fractions
and Lean) versus floating point, sweeps of γ and of the costs, structural variants (two shuttles, holding-cost
timing, a cap of 400, deterministic arrivals), Monte Carlo rollouts versus the value function, and the monotone
improvement of policy iteration; for `pg1`, the gradient against its closed form, bandits, REINFORCE against the
exact optimum of the corridor computed by value iteration, the role of the discount factor, the three ways of
weighting the log-probabilities, a short CartPole run and the Pong pipeline; for `pg2`, unbiasedness and variance
with several baselines, lock-in on the first rewarded arm with and without a baseline, the value network against
the exact values, learning curves, and CartPole with the baseline.

**Unit tests** (standard-library `unittest`, no extra package) live in `tests/unit/` and cover the building blocks
and edge cases: the MDP construction (costs, transitions, the cap, degenerate parameters, invalid input), the
solvers on a two-state problem with a closed-form solution and on the shuttle problem (agreement, error bound,
contraction ratios, monotonicity, threshold policy, γ = 0, γ = 1, horizon 0, single state, iteration caps),
exact-arithmetic references, the explorer's data (its recorded steps are the solvers' own iterates, its finite-horizon
policies are exact, and, when Node is installed, the JavaScript replay only makes moves the model allows),
discounted returns, policy networks and adapters (Pong frames, action mapping),
training on toy environments (gradient against the closed form, reproducibility, truncation, constant returns
with normalisation, a perfect baseline giving a zero gradient), and the baselines. Run them with
`python run.py unittest` (or `python -m unittest discover -s tests/unit -t .`).

## 7. Formal verification with Lean

`lean/` is a Lean 4 project (Mathlib) that states the models the code implements over exact rationals and proves
the properties the solvers and the tests rely on, so the numerical checks compare against theorems rather than
expectations. `lean/README.md` lists every theorem; in short:

| File | Proves |
|---|---|
| `lean/RLVerify/Shuttle.lean` | for the shuttle MDP: the state never leaves 0..200; costs and values are non-negative; the optimal value is monotone in the horizon and in the number waiting; the Bellman operator is a γ-contraction, so value-iteration changes decay geometrically and the stopping rule's error bound γΔ/(1−γ) is sound; the improvement step of policy iteration never makes a policy worse |
| `lean/RLVerify/Returns.lean` | the backward recursion G_t = r_t + γG_{t+1} used by `discounted_returns` equals Σ_k γ^k r_{t+k} |
| `lean/RLVerify/Baseline.lean` | for a softmax policy, subtracting any action-independent baseline leaves the expected policy gradient unchanged |
| `lean/ShuttleRef.lean` | an executable that evaluates the verified value definition on small instances; its output `lean/reference/shuttle_reference.json` is committed and compared with the Python solvers by `tests/unit/test_dp_lean_reference.py` and a `dp` experiment |

Lean is not needed to run anything in Python. To re-check the proofs: install `elan` (<https://github.com/leanprover/elan>),
then `cd lean && lake exe cache get && lake build` (the first build downloads Mathlib's prebuilt files, several GB),
and `lake env lean --run ShuttleRef.lean > reference/shuttle_reference.json` regenerates the reference values.

## 8. Report

The report and the dashboard are written for the instructors grading the assignment. The report opens with the
submission details, the instructions for running the code and a table of contents, then follows the handout's
order; the dashboard has the same sections, one tab per part.

Each section tab lists that part's questions from the handout, with its plots, evaluation statistics and validation
results, and the answer, shown formatted. **Contents** at the right edge of the page lists the open tab's sections
and highlights the one on screen (`ui/contents.js`). Write or change an answer under **Write answer** / **Edit answer**; it
saves to `report/answers.json` when you click outside the box. In the **Overview** tab, check the submission details
(name, GitHub username, repository name and link; **Edit submission details** changes them), then press **Build
report** (or run `python run.py report`). It writes:

| File | What it is |
|---|---|
| `report/report.md` | the report in Markdown |
| `report/report.pdf` | the same report as a PDF, for Canvas, with a table of contents (page numbers) and an outline in the viewer's sidebar |

The run instructions, the report's "Instructions for running the code" section that the handout requires, live
in `report/instructions.md` (edit them there). The PDF is made with `reportlab` (in `requirements-ui.txt`), so
pandoc and LaTeX are not needed.

The Section 2 validation answer opens with a picture of the validation-test summary table (one row per test:
what it checks, how it validates, the pass criterion and the result). Its rows are the Markdown table in
`report/dp_tests_table.md`; `python scripts/tests_table.py` sets it in the report's typeface in headless Chrome and
saves `results/dp/tests_table.png`, which the report and the dashboard show above the harness's results; `report/pg1_tests_table.md` and `report/pg2_tests_table.md` are the same tables for the policy-gradient parts (`python scripts/tests_table.py --source report/pg1_tests_table.md --out results/pg/pg1_tests_table.png`, likewise for pg2). To change
the table, edit the Markdown file and rerun the script.

The questions themselves live in `report/questions.py`.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| Connect times out / "Cannot reach" | Connect to the campus network or GlobalProtect VPN. |
| "did not accept an SSH key" | Type your NetID password and press Connect, or rerun `./setup.sh` to copy your key. |
| A NetID or server you set is not used | A `DEEPDISH_*` variable in the environment overrides `.env`: `python run.py deepdish` shows where each value comes from. |
| Status doesn't show a session you just opened | Press **Refresh status** (status is cached for 10 s). |
| `no kernel image is available` / GPU test fails | The torch build doesn't support the GPU: rerun `./setup.sh --server` on deepdish, or set `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126`. |
| "already running" when launching | A job with that name is running; stop it or set a run tag. |
| Port 8501 in use | `python run.py ui --port 8502` (the launcher says so and stops). |
| The dashboard page is blank, says "Please wait" or shows a connection error | Wait for `Dashboard: http://localhost:8501` in the terminal, then reload the page (R); open the URL by hand; try `--port 8502` or another browser. The report has everything if it still does not load. |
| The sidebar is gone | Click **»** at the top left of the page. |
| `reportlab requires 3.10+` | reportlab 5 was installed on Python 3.9: `pip install -r requirements-ui.txt` reinstalls 4.x. |
