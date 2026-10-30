"""Policy gradient on the GPU: CUDA kernels for the parts of REINFORCE that are not already cuBLAS/cuDNN work.

The policy network's forward and backward passes already run on the GPU through PyTorch; what the Python code
does step by step is the rest. Kernels:

* `returns_scan_kernel` + `returns_fixup_kernel`: discounted returns G_t = sum_k gamma^k r_{t+k} for a whole
  episode as a parallel suffix scan. The recurrence G_t = r_t + gamma G_{t+1} is sequential, but the pairs
  (sum, decay) = (r_t, gamma) combine associatively, (a1, g1) ⊕ (a2, g2) = (a1 + g1 a2, g1 g2), so a block scans
  a chunk of up to 1024 steps in log2(chunk) rounds in shared memory; longer episodes are split into chunks,
  the chunk aggregates are combined, and a second kernel adds each chunk's carry (decay × return at the start
  of the next chunk).
* `preprocess_kernel`: the handout's Pong preprocessing (crop, downsample by 2, erase the two background
  colours, binarise) fused with the frame difference, one thread per output pixel, on a batch of frames
  (3-D grid: frame, row, column).
* `block_sum_kernel` + `normalize_kernel`: mean and standard deviation of the advantages by block reductions,
  then the elementwise standardisation.
* `sample_actions_kernel`: one thread per state computes the softmax of its logits (stable, max-subtracted)
  and draws an action by inverse CDF from a given uniform number.

`discounted_returns_gpu`, `preprocess_frames_gpu`, `normalize_gpu` and `sample_actions_gpu` are the wrappers;
`--set kernels=true` makes rl/pg/reinforce.py and the Pong adapter use the first two.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from rl.cuda.backend import MAX_THREADS, launch_1d, launch_3d  # noqa: I001  (before numba: simulator switch)
from numba import cuda, float64  # noqa: E402

CHUNK = 1024  # steps per block in the returns scan (shared memory: 2 double buffers of CHUNK doubles each)
CHUNK2 = 2 * CHUNK  # cuda.shared.array needs a literal constant size (a module constant, not an expression)
FRAME = 80


# --- discounted returns: parallel suffix scan ---------------------------------------------------------------


@cuda.jit
def returns_scan_kernel(rewards, gamma, chunk, out, suffix_pow, agg_a, agg_g):
    """Block c scans rewards[c*chunk : (c+1)*chunk] from the right: out[t] = sum_{k>=t in chunk} gamma^(k-t) r_k,
    suffix_pow[t] = gamma^(chunk_end - t), and the chunk aggregate (agg_a, agg_g) = (out[chunk start], gamma^len)."""
    c = cuda.blockIdx.x
    t = cuda.threadIdx.x
    i = c * chunk + t
    T = rewards.shape[0]
    n = min(chunk, T - c * chunk)  # valid steps in this chunk
    sa = cuda.shared.array(CHUNK2, float64)
    sg = cuda.shared.array(CHUNK2, float64)
    valid = t < n
    if valid:
        sa[t] = rewards[i]
        sg[t] = gamma
    else:
        sa[t] = 0.0
        sg[t] = 1.0
    cuda.syncthreads()
    offset = 1
    buf = 0
    while offset < n:
        src = buf * CHUNK
        dst = (1 - buf) * CHUNK
        if t + offset < n:
            sa[dst + t] = sa[src + t] + sg[src + t] * sa[src + t + offset]
            sg[dst + t] = sg[src + t] * sg[src + t + offset]
        else:
            sa[dst + t] = sa[src + t]
            sg[dst + t] = sg[src + t]
        cuda.syncthreads()
        buf = 1 - buf
        offset *= 2
    src = buf * CHUNK
    if valid:
        out[i] = sa[src + t]
        suffix_pow[i] = sg[src + t]
    if t == 0:
        agg_a[c] = sa[src]
        agg_g[c] = sg[src]


@cuda.jit
def returns_fixup_kernel(out, suffix_pow, carry, chunk):
    """out[t] += gamma^(chunk_end - t) * (return at the start of the next chunk)."""
    i = cuda.grid(1)
    if i < out.shape[0]:
        c = i // chunk
        out[i] += suffix_pow[i] * carry[c + 1]


def discounted_returns_gpu(rewards: Any, gamma: float, chunk: int = CHUNK) -> np.ndarray:
    """G_t for every step of an episode, computed by the scan kernels (same result as discounted_returns)."""
    r = np.ascontiguousarray(np.asarray(rewards, dtype=np.float64))
    T = len(r)
    if T == 0:
        return np.zeros(0)
    if not 1 <= chunk <= CHUNK:
        raise ValueError(f"chunk must be between 1 and {CHUNK}")
    chunks = math.ceil(T / chunk)
    d_r = cuda.to_device(r)
    d_out = cuda.device_array(T, dtype=np.float64)
    d_pow = cuda.device_array(T, dtype=np.float64)
    d_agg_a = cuda.device_array(chunks, dtype=np.float64)
    d_agg_g = cuda.device_array(chunks, dtype=np.float64)
    returns_scan_kernel[chunks, chunk](d_r, float(gamma), int(chunk), d_out, d_pow, d_agg_a, d_agg_g)
    if chunks > 1:
        agg_a, agg_g = d_agg_a.copy_to_host(), d_agg_g.copy_to_host()
        carry = np.zeros(chunks + 1)  # return at the start of each chunk, carry[chunks] = 0
        for c in range(chunks - 1, -1, -1):
            carry[c] = agg_a[c] + agg_g[c] * carry[c + 1]
        blocks, threads = launch_1d(T)
        returns_fixup_kernel[blocks, threads](d_out, d_pow, cuda.to_device(carry), int(chunk))
    return d_out.copy_to_host()


# --- Pong preprocessing --------------------------------------------------------------------------------------


@cuda.jit
def preprocess_kernel(frames, previous, use_previous, binarized, out):
    """Thread (n, y, x): binarised pixel of frame n at (35 + 2y, 2x) and its difference with the previous frame."""
    x, y, n = cuda.grid(3)
    if n < frames.shape[0] and y < FRAME and x < FRAME:
        v = frames[n, 35 + 2 * y, 2 * x, 0]
        b = 0.0 if (v == 144 or v == 109 or v == 0) else 1.0
        binarized[n, y, x] = b
        if use_previous:
            out[n, y, x] = b - previous[n, y, x]
        else:
            out[n, y, x] = 0.0


def preprocess_frames_gpu(frames: np.ndarray, previous: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(binarised frames, frame differences) for a batch of raw 210x160x3 frames; with `previous` None the
    differences are zero (first frame of an episode)."""
    f = np.ascontiguousarray(np.asarray(frames, dtype=np.uint8))
    if f.ndim == 3:
        f = f[None]
    N = f.shape[0]
    prev = np.zeros((N, FRAME, FRAME), np.float32) if previous is None else np.ascontiguousarray(np.asarray(previous, np.float32).reshape(N, FRAME, FRAME))
    d_bin = cuda.device_array((N, FRAME, FRAME), dtype=np.float32)
    d_out = cuda.device_array((N, FRAME, FRAME), dtype=np.float32)
    blocks, threads = launch_3d(FRAME, FRAME, N, threads=(16, 16, 1))
    preprocess_kernel[blocks, threads](cuda.to_device(f), cuda.to_device(prev), previous is not None, d_bin, d_out)
    return d_bin.copy_to_host(), d_out.copy_to_host()


# --- advantage normalisation ---------------------------------------------------------------------------------


@cuda.jit
def block_sum_kernel(x, shift, partial_sums, partial_squares):
    """Each block reduces sum(x - shift) and sum((x - shift)^2) over its slice in shared memory."""
    i = cuda.grid(1)
    t = cuda.threadIdx.x
    sh_s = cuda.shared.array(MAX_THREADS, float64)
    sh_q = cuda.shared.array(MAX_THREADS, float64)
    v = 0.0
    if i < x.shape[0]:
        v = x[i] - shift
    sh_s[t] = v
    sh_q[t] = v * v
    stride = cuda.blockDim.x // 2
    while stride > 0:
        cuda.syncthreads()
        if t < stride:
            sh_s[t] += sh_s[t + stride]
            sh_q[t] += sh_q[t + stride]
        stride //= 2
    cuda.syncthreads()
    if t == 0:
        partial_sums[cuda.blockIdx.x] = sh_s[0]
        partial_squares[cuda.blockIdx.x] = sh_q[0]


@cuda.jit
def normalize_kernel(x, mean, inv_std, out):
    i = cuda.grid(1)
    if i < x.shape[0]:
        out[i] = (x[i] - mean) * inv_std


def mean_std_gpu(x: Any, threads: int = 256) -> tuple[float, float]:
    """Mean and (population) standard deviation by block reductions, shifted by x[0] for accuracy."""
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    n = len(a)
    if n == 0:
        return 0.0, 0.0
    blocks, threads = launch_1d(n, threads)
    d_s = cuda.device_array(blocks, dtype=np.float64)
    d_q = cuda.device_array(blocks, dtype=np.float64)
    shift = float(a[0])
    block_sum_kernel[blocks, threads](cuda.to_device(a), shift, d_s, d_q)
    s, q = float(d_s.copy_to_host().sum()), float(d_q.copy_to_host().sum())
    mean_shifted = s / n
    var = max(q / n - mean_shifted ** 2, 0.0)
    return mean_shifted + shift, math.sqrt(var)


def normalize_gpu(x: Any, eps: float = 1e-8) -> np.ndarray:
    """(x - mean) / std with the reductions and the elementwise pass on the device; std < eps leaves the scale."""
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if len(a) < 2:
        return a - a.mean() if len(a) else a
    mean, std = mean_std_gpu(a)
    inv = 1.0 / std if std > eps else 1.0
    blocks, threads = launch_1d(len(a))
    d_out = cuda.device_array(len(a), dtype=np.float64)
    normalize_kernel[blocks, threads](cuda.to_device(a), float(mean), float(inv), d_out)
    return d_out.copy_to_host()


# --- softmax sampling ----------------------------------------------------------------------------------------


@cuda.jit
def sample_actions_kernel(logits, uniforms, actions, probs_out):
    """Thread i: softmax of logits[i, :] (max-subtracted) and the action whose CDF interval contains uniforms[i]."""
    i = cuda.grid(1)
    if i < logits.shape[0]:
        A = logits.shape[1]
        m = logits[i, 0]
        for a in range(1, A):
            if logits[i, a] > m:
                m = logits[i, a]
        z = 0.0
        for a in range(A):
            z += math.exp(logits[i, a] - m)
        acc = 0.0
        chosen = A - 1
        picked = False
        for a in range(A):
            p = math.exp(logits[i, a] - m) / z
            probs_out[i, a] = p
            acc += p
            if not picked and uniforms[i] < acc:
                chosen = a
                picked = True
        actions[i] = chosen


def sample_actions_gpu(logits: Any, uniforms: Any) -> tuple[np.ndarray, np.ndarray]:
    """(actions, probabilities) for a batch of logits and one uniform number per row."""
    L = np.ascontiguousarray(np.asarray(logits, dtype=np.float64))
    u = np.ascontiguousarray(np.asarray(uniforms, dtype=np.float64))
    if L.ndim != 2 or len(u) != L.shape[0]:
        raise ValueError("logits must be (rows, actions) with one uniform number per row")
    blocks, threads = launch_1d(L.shape[0])
    d_actions = cuda.device_array(L.shape[0], dtype=np.int32)
    d_probs = cuda.device_array(L.shape, dtype=np.float64)
    sample_actions_kernel[blocks, threads](cuda.to_device(L), cuda.to_device(u), d_actions, d_probs)
    return d_actions.copy_to_host(), d_probs.copy_to_host()
