"""Section 2 on the GPU: CUDA kernels for the shuttle MDP and the three solvers.

Data layout (device arrays). Every (state, action) pair has at most D successor states, so the MDP is stored
sparsely: `cost[s, a]`, `succ_idx[s, a, j]` (int32) and `succ_p[s, a, j]` (probability, 0 for padding). For the
shuttle problem D = number of arrival values (5), against 201 columns of the dense matrix.

Kernels (one thread per state unless noted):

* `init_shuttle_tables`: initialises the state grid. A 2-D grid of threads, one per (state, action), computes
  that pair's cost and successor table from the problem constants, so the MDP is built on the device without
  ever materialising a dense transition matrix.
* `sweep_persistent`: the whole state space in ONE block (S <= 1024, the handout's 201 states). The value
  function lives in shared memory; each thread owns one state and applies the Bellman backup
  (`mode = 0`: min over actions, writes the greedy action; `mode = 1`: the given policy's action), the block
  computes max_s |V_new(s) - V(s)| with a tree reduction in shared memory, and the loop continues inside the
  kernel until the residual is below `tol` or `max_iter` sweeps were done. One kernel launch runs all of value
  iteration (or exactly T sweeps for enumeration, with `tol < 0`), with no host round trip per iteration.
* `sweep_kernel`: the same backup for state spaces larger than a block (one launch per sweep, grid-wide, with a
  per-block residual reduction that the host finishes).
* `greedy_kernel`: policy improvement with tie-breaking that keeps the current action; counts changed states
  with an atomic add.
* `simulate_kernel`: Monte Carlo evaluation, one thread per trajectory with its own xoroshiro128+ stream.

The Python wrappers `enumeration_gpu`, `value_iteration_gpu`, `policy_iteration_gpu` and `simulate_policy_gpu`
return the same `SolverResult` / dictionaries as rl/dp/solvers.py, so the plots, tests and report do not care
which backend produced them.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from rl.cuda.backend import MAX_THREADS, backend_name, block_for_states, device_summary, launch_1d, launch_2d  # noqa: I001  (before numba: simulator switch)
from numba import cuda, float64  # noqa: E402
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64  # noqa: E402
from rl.dp.shuttle import HOLDING_CHARGED, ShuttleParams, TabularMDP
from rl.dp.solvers import SolverResult, bellman_residual, policy_threshold

BIG = 1e300


# --- host-side layout ------------------------------------------------------------------------------------


def sparse_tables(mdp: TabularMDP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cost, succ_idx, succ_p) for any TabularMDP: the non-zero successors of every (s, a), zero-padded."""
    P = mdp.transitions
    A, S, _ = P.shape
    counts = (P > 0).sum(axis=2)
    D = int(counts.max()) if counts.size else 1
    succ_idx = np.zeros((S, A, D), dtype=np.int32)
    succ_p = np.zeros((S, A, D), dtype=np.float64)
    for a in range(A):
        for s in range(S):
            nz = np.flatnonzero(P[a, s] > 0)
            succ_idx[s, a, : len(nz)] = nz
            succ_p[s, a, : len(nz)] = P[a, s, nz]
    return np.ascontiguousarray(mdp.costs, dtype=np.float64), succ_idx, succ_p


# --- kernels ----------------------------------------------------------------------------------------------


@cuda.jit
def init_shuttle_tables(capacity, dispatch_cost, holding_cost, arrivals, probs, max_customers, holding_mode,
                        mean_arrivals, cost, succ_idx, succ_p):
    """Thread (s, a) fills cost[s, a] and the successor table of (s, a) from the problem constants."""
    s, a = cuda.grid(2)
    if s < cost.shape[0] and a < cost.shape[1]:
        served = min(s, a * capacity)
        left = s - served
        if holding_mode == 0:  # after the dispatch
            waiting = float(left)
        elif holding_mode == 1:  # before the dispatch
            waiting = float(s)
        else:  # end of the period, in expectation
            waiting = left + mean_arrivals
        cost[s, a] = a * dispatch_cost + holding_cost * waiting
        for i in range(arrivals.shape[0]):
            succ_idx[s, a, i] = min(left + arrivals[i], max_customers)
            succ_p[s, a, i] = probs[i]


@cuda.jit(device=True)
def backup(cost, succ_idx, succ_p, gamma, values, s, a):
    """Q(s, a) = c(s, a) + gamma * sum_j p_j V(succ_j), reading V from `values` (shared or global)."""
    q = cost[s, a]
    for j in range(succ_idx.shape[2]):
        p = succ_p[s, a, j]
        if p > 0.0:
            q += gamma * p * values[succ_idx[s, a, j]]
    return q


@cuda.jit
def sweep_persistent(cost, succ_idx, succ_p, gamma, tol, max_iter, mode, policy, V, residuals, out):
    """All states in one block: Bellman sweeps with the value function in shared memory until convergence."""
    s = cuda.threadIdx.x
    nthreads = cuda.blockDim.x
    S = V.shape[0]
    A = cost.shape[1]
    sh_v = cuda.shared.array(MAX_THREADS, float64)  # current values
    sh_r = cuda.shared.array(MAX_THREADS, float64)  # per-thread |change|, then the tree reduction
    if s < S:
        sh_v[s] = V[s]
    cuda.syncthreads()
    it = 0
    resid = 0.0
    new_v = 0.0
    best_a = 0
    while it < max_iter:
        if s < S:
            if mode == 0:
                best = BIG
                best_a = 0
                for a in range(A):
                    q = backup(cost, succ_idx, succ_p, gamma, sh_v, s, a)
                    if q < best:
                        best = q
                        best_a = a
                new_v = best
            else:
                new_v = backup(cost, succ_idx, succ_p, gamma, sh_v, s, policy[s])
            sh_r[s] = abs(new_v - sh_v[s])
        else:
            sh_r[s] = 0.0
        cuda.syncthreads()  # everyone has read the old values
        if s < S:
            sh_v[s] = new_v
            if mode == 0:
                policy[s] = best_a
        stride = nthreads // 2
        while stride > 0:
            cuda.syncthreads()
            if s < stride and sh_r[s + stride] > sh_r[s]:
                sh_r[s] = sh_r[s + stride]
            stride //= 2
        cuda.syncthreads()
        resid = sh_r[0]  # the same value in every thread: the break below is uniform
        if s == 0:
            residuals[it] = resid
        it += 1
        if resid < tol:
            break
        cuda.syncthreads()  # nobody overwrites sh_r[0] before everyone has read it
    if s < S:
        V[s] = sh_v[s]
    if s == 0:
        out[0] = it
        out[1] = resid


@cuda.jit
def sweep_kernel(cost, succ_idx, succ_p, gamma, mode, policy_in, V_in, V_out, policy_out, block_max):
    """One Bellman sweep over a grid of blocks (state spaces larger than one block); V_in -> V_out."""
    s = cuda.grid(1)
    t = cuda.threadIdx.x
    sh_r = cuda.shared.array(MAX_THREADS, float64)
    delta = 0.0
    if s < V_in.shape[0]:
        if mode == 0:
            best = BIG
            best_a = 0
            for a in range(cost.shape[1]):
                q = backup(cost, succ_idx, succ_p, gamma, V_in, s, a)
                if q < best:
                    best = q
                    best_a = a
            policy_out[s] = best_a
            new_v = best
        else:
            new_v = backup(cost, succ_idx, succ_p, gamma, V_in, s, policy_in[s])
        V_out[s] = new_v
        delta = abs(new_v - V_in[s])
    sh_r[t] = delta
    stride = cuda.blockDim.x // 2
    while stride > 0:
        cuda.syncthreads()
        if t < stride and sh_r[t + stride] > sh_r[t]:
            sh_r[t] = sh_r[t + stride]
        stride //= 2
    cuda.syncthreads()
    if t == 0:
        block_max[cuda.blockIdx.x] = sh_r[0]


@cuda.jit
def greedy_kernel(cost, succ_idx, succ_p, gamma, V, current, tie_tol, new_policy, changed):
    """Policy improvement: thread s picks argmin_a Q(s, a), keeping its current action on near-ties."""
    s = cuda.grid(1)
    if s < V.shape[0]:
        best = BIG
        best_a = 0
        q_current = 0.0
        for a in range(cost.shape[1]):
            q = backup(cost, succ_idx, succ_p, gamma, V, s, a)
            if a == current[s]:
                q_current = q
            if q < best:
                best = q
                best_a = a
        slack = tie_tol * max(1.0, abs(best))
        chosen = current[s] if q_current <= best + slack else best_a
        new_policy[s] = chosen
        if chosen != current[s]:
            cuda.atomic.add(changed, 0, 1)


@cuda.jit
def simulate_kernel(cost, succ_idx, succ_p, policy, start, horizon, gamma, rng_states, out):
    """One thread per trajectory: follow `policy` for `horizon` periods and accumulate the discounted cost."""
    tid = cuda.grid(1)
    if tid < out.shape[0]:
        s = start
        disc = 1.0
        total = 0.0
        for _ in range(horizon):
            a = policy[s]
            total += disc * cost[s, a]
            u = xoroshiro128p_uniform_float64(rng_states, tid)
            acc = 0.0
            nxt = succ_idx[s, a, 0]
            for j in range(succ_idx.shape[2]):
                p = succ_p[s, a, j]
                if p > 0.0:
                    nxt = succ_idx[s, a, j]
                    acc += p
                    if u < acc:
                        break
            s = nxt
            disc *= gamma
        out[tid] = total


# --- wrappers ---------------------------------------------------------------------------------------------


def shuttle_tables_gpu(params: ShuttleParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the cost and successor tables of the shuttle problem on the device (init_shuttle_tables)."""
    arrivals, probs = params.arrival_distribution()
    S, A, D = params.max_customers + 1, params.max_dispatch + 1, len(arrivals)
    d_cost = cuda.device_array((S, A), dtype=np.float64)
    d_idx = cuda.device_array((S, A, D), dtype=np.int32)
    d_p = cuda.device_array((S, A, D), dtype=np.float64)
    blocks, threads = launch_2d(S, A)
    init_shuttle_tables[blocks, threads](
        int(params.capacity), float(params.dispatch_cost), float(params.holding_cost),
        cuda.to_device(arrivals.astype(np.int32)), cuda.to_device(probs.astype(np.float64)),
        int(params.max_customers), HOLDING_CHARGED.index(params.holding_charged), float(params.mean_arrivals),
        d_cost, d_idx, d_p,
    )
    return d_cost.copy_to_host(), d_idx.copy_to_host(), d_p.copy_to_host()


def mdp_from_tables(cost: np.ndarray, succ_idx: np.ndarray, succ_p: np.ndarray, gamma: float, action_labels: tuple, name: str = "shuttle-gpu") -> TabularMDP:
    """A dense TabularMDP from sparse tables (to compare the device-built model with build_mdp)."""
    S, A, D = succ_idx.shape
    P = np.zeros((A, S, S))
    for s in range(S):
        for a in range(A):
            for j in range(D):
                if succ_p[s, a, j] > 0:
                    P[a, s, succ_idx[s, a, j]] += succ_p[s, a, j]
    return TabularMDP(P, cost, gamma, np.arange(S), action_labels, name=name)


def _info(extra: dict[str, Any], start: float, kernel: str) -> dict[str, Any]:
    return {"backend": backend_name(), "device": device_summary(), "kernel": kernel, "time_s": round(time.time() - start, 4), **extra}


def _persistent_sweeps(tables: tuple, gamma: float, tol: float, max_iter: int, mode: int, policy: np.ndarray, values: np.ndarray):
    """Run `sweep_persistent` once; returns (values, policy, residuals, iterations, last residual)."""
    cost, succ_idx, succ_p = tables
    S = cost.shape[0]
    block = block_for_states(S)
    if block is None:
        raise ValueError(f"the persistent kernel holds at most {MAX_THREADS} states in one block; got {S}")
    d_policy = cuda.to_device(np.ascontiguousarray(policy, dtype=np.int32))
    d_V = cuda.to_device(np.ascontiguousarray(values, dtype=np.float64))
    d_res = cuda.device_array(max(1, int(max_iter)), dtype=np.float64)
    d_out = cuda.to_device(np.zeros(2))
    sweep_persistent[1, block](cuda.to_device(cost), cuda.to_device(succ_idx), cuda.to_device(succ_p),
                               float(gamma), float(tol), int(max_iter), int(mode), d_policy, d_V, d_res, d_out)
    out = d_out.copy_to_host()
    iterations = int(out[0])
    return d_V.copy_to_host(), d_policy.copy_to_host(), d_res.copy_to_host()[:iterations], iterations, float(out[1])


def _grid_sweeps(tables: tuple, gamma: float, tol: float, max_iter: int, mode: int, policy: np.ndarray, values: np.ndarray, threads: int = 256):
    """The multi-block path: one `sweep_kernel` launch per sweep, residual finished on the host."""
    cost, succ_idx, succ_p = tables
    S = cost.shape[0]
    blocks, threads = launch_1d(S, threads)
    d_cost, d_idx, d_p = cuda.to_device(cost), cuda.to_device(succ_idx), cuda.to_device(succ_p)
    d_policy = cuda.to_device(np.ascontiguousarray(policy, dtype=np.int32))
    d_policy_out = cuda.to_device(np.ascontiguousarray(policy, dtype=np.int32))
    d_a = cuda.to_device(np.ascontiguousarray(values, dtype=np.float64))
    d_b = cuda.device_array(S, dtype=np.float64)
    d_block_max = cuda.device_array(blocks, dtype=np.float64)
    residuals = []
    resid = float("inf")
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        sweep_kernel[blocks, threads](d_cost, d_idx, d_p, float(gamma), int(mode), d_policy, d_a, d_b, d_policy_out, d_block_max)
        resid = float(d_block_max.copy_to_host().max())
        residuals.append(resid)
        d_a, d_b = d_b, d_a
        if mode == 0:
            d_policy, d_policy_out = d_policy_out, d_policy
        if resid < tol:
            break
    return d_a.copy_to_host(), d_policy.copy_to_host(), np.array(residuals), iterations, resid


def _sweeps(tables, gamma, tol, max_iter, mode, policy, values, force_grid=False):
    S = tables[0].shape[0]
    if not force_grid and block_for_states(S) is not None:
        return (*_persistent_sweeps(tables, gamma, tol, max_iter, mode, policy, values), "persistent (one block, shared memory)")
    return (*_grid_sweeps(tables, gamma, tol, max_iter, mode, policy, values), "grid (one launch per sweep)")


def enumeration_gpu(mdp: TabularMDP, horizon: int, force_grid: bool = False) -> SolverResult:
    """(a) Backward induction: exactly `horizon` optimal sweeps on the device (tol < 0 never stops early)."""
    start = time.time()
    tables = sparse_tables(mdp)
    if horizon == 0:
        return SolverResult(mdp.states, np.zeros(mdp.n_states), None, 0, {"max |V_t − V_{t+1}|": []}, _info({"horizon": 0}, start, "none"))
    V, policy, residuals, iterations, _, kernel = _sweeps(tables, mdp.gamma, -1.0, horizon, 0, np.zeros(mdp.n_states, np.int32), np.zeros(mdp.n_states), force_grid)
    return SolverResult(mdp.states, V, policy, iterations, {"max |V_t − V_{t+1}|": residuals.tolist()}, _info({"horizon": horizon}, start, kernel))


def value_iteration_gpu(mdp: TabularMDP, tol: float = 1e-8, max_iter: int = 100_000, force_grid: bool = False) -> SolverResult:
    """(b) Value iteration on the device; same stopping rule and error bound as the CPU solver."""
    if mdp.gamma >= 1.0:
        raise ValueError("value iteration needs gamma < 1")
    start = time.time()
    tables = sparse_tables(mdp)
    V, policy, residuals, iterations, resid, kernel = _sweeps(tables, mdp.gamma, tol, max_iter, 0, np.zeros(mdp.n_states, np.int32), np.zeros(mdp.n_states), force_grid)
    converged = resid < tol
    return SolverResult(
        mdp.states, V, policy, iterations, {"‖V_{n+1} − V_n‖∞": residuals.tolist()},
        _info({"tol": tol, "converged": bool(converged), "final_change": resid, "error_bound": mdp.gamma * resid / (1 - mdp.gamma),
               "bellman_residual": bellman_residual(mdp, V)}, start, kernel),
    )


def evaluate_policy_gpu(mdp: TabularMDP, policy: np.ndarray, tol: float = 1e-13, max_iter: int = 200_000, force_grid: bool = False) -> np.ndarray:
    """V^pi by Jacobi sweeps on the device (mode 1 of the sweep kernels)."""
    tables = sparse_tables(mdp)
    V, _, _, _, resid, _ = _sweeps(tables, mdp.gamma, tol, max_iter, 1, np.asarray(policy, np.int32), np.zeros(mdp.n_states), force_grid)
    return V


def policy_iteration_gpu(mdp: TabularMDP, max_iter: int = 1_000, eval_tol: float = 1e-13, tie_tol: float = 1e-9, force_grid: bool = False) -> SolverResult:
    """(c) Policy iteration: device policy evaluation (Jacobi sweeps) and device greedy improvement."""
    if mdp.gamma >= 1.0:
        raise ValueError("policy iteration needs gamma < 1")
    start = time.time()
    tables = sparse_tables(mdp)
    cost, succ_idx, succ_p = tables
    S = mdp.n_states
    d_cost, d_idx, d_p = cuda.to_device(cost), cuda.to_device(succ_idx), cuda.to_device(succ_p)
    policy = np.zeros(S, dtype=np.int32)
    policies = [policy.copy()]
    changes: list[float] = []
    blocks, threads = launch_1d(S)
    converged = False
    V = np.zeros(S)
    iterations = 0
    kernel = ""
    for iterations in range(1, int(max_iter) + 1):
        V, _, _, _, _, kernel = _sweeps(tables, mdp.gamma, eval_tol, 200_000, 1, policy, V, force_grid)
        d_new = cuda.device_array(S, dtype=np.int32)
        d_changed = cuda.to_device(np.zeros(1, dtype=np.int32))
        greedy_kernel[blocks, threads](d_cost, d_idx, d_p, float(mdp.gamma), cuda.to_device(V), cuda.to_device(policy), float(tie_tol), d_new, d_changed)
        changed = int(d_changed.copy_to_host()[0])
        changes.append(float(changed))
        if changed == 0:
            converged = True
            break
        policy = d_new.copy_to_host()
        policies.append(policy.copy())
    return SolverResult(
        mdp.states, V, policy, iterations, {"states whose action changed": changes},
        _info({"converged": converged, "evaluation": "jacobi sweeps on the device", "policies": [p.tolist() for p in policies],
               "bellman_residual": bellman_residual(mdp, V), "threshold": policy_threshold(policy) if mdp.n_actions == 2 else None}, start, kernel),
    )


def simulate_policy_gpu(mdp: TabularMDP, policy: np.ndarray, start: int, horizon: int, episodes: int = 100_000, seed: int = 0, threads: int = 256) -> dict[str, Any]:
    """Monte Carlo estimate of V^pi(start) with one device thread per trajectory."""
    t0 = time.time()
    cost, succ_idx, succ_p = sparse_tables(mdp)
    blocks, threads = launch_1d(episodes, threads)
    rng_states = create_xoroshiro128p_states(blocks * threads, seed=seed)
    d_out = cuda.device_array(int(episodes), dtype=np.float64)
    simulate_kernel[blocks, threads](cuda.to_device(cost), cuda.to_device(succ_idx), cuda.to_device(succ_p),
                                     cuda.to_device(np.asarray(policy, np.int32)), int(start), int(horizon), float(mdp.gamma), rng_states, d_out)
    costs = d_out.copy_to_host()
    return {
        "mean": float(costs.mean()),
        "stderr": float(costs.std(ddof=1) / np.sqrt(episodes)) if episodes > 1 else float("nan"),
        "costs": costs,
        "truncation_bound": float(mdp.gamma ** horizon * mdp.costs.max() / (1 - mdp.gamma)) if mdp.gamma < 1 else float("inf"),
        "backend": backend_name(), "device": device_summary(), "time_s": round(time.time() - t0, 4),
        "grid": {"blocks": blocks, "threads": threads},
    }
