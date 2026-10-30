"""CUDA versions of the algorithms, written as explicit kernels (Numba CUDA): grid/block initialisation and
thread-level code.

    rl/cuda/backend.py   device detection, the CPU simulator fallback, launch-configuration helpers
    rl/cuda/dp.py        Section 2: table initialisation, value iteration / enumeration / policy evaluation
                         sweeps (one thread per state, shared-memory residual reduction, in-kernel convergence
                         loop), greedy improvement, Monte Carlo simulation (one thread per trajectory)
    rl/cuda/pg.py        policy gradient: discounted returns as a parallel suffix scan, fused Pong frame
                         preprocessing, mean/std reductions for advantage normalisation, softmax sampling

Use: `python run.py dp --backend cuda`, `python run.py pg ... --set kernels=true`. On a machine without an
NVIDIA GPU the same kernels run in Numba's CUDA simulator (slow, for tests): see backend.py.
"""
from rl.cuda import backend as backend  # noqa: E402,F401  must run before anything imports numba (simulator switch)
