"""Numba CUDA backend: where the kernels run, and helpers to initialise grids.

Importing this module decides the target:

* with an NVIDIA GPU visible to PyTorch (`torch.cuda.is_available()`), kernels are compiled for it and run
  natively (deepdish; Numba needs the CUDA toolkit's NVVM library: `pip install nvidia-cuda-nvcc-cu12`, or set
  CUDA_HOME to a toolkit, then `python -c "from numba import cuda; cuda.detect()"`). With a CUDA 13 driver
  Numba 0.60 must use NVIDIA's `cuda-python` bindings (requirements.txt installs them on Linux; this module
  turns them on), because its built-in driver binding segfaults at the first kernel launch;
* otherwise `NUMBA_ENABLE_CUDASIM=1` is set before Numba is imported, so the very same kernels run in Numba's
  CUDA simulator, where every CUDA thread is a Python thread. That is correct but slow; it is how the kernels
  are unit-tested on a laptop. `RL_CUDA_SIMULATOR=1` forces the simulator even when a GPU is present.

Grid initialisation: `launch_1d(n)` returns `(blocks, threads)` for one thread per element, `launch_2d` and
`launch_3d` the same for 2-D / 3-D index spaces, and `block_for_states(n)` the (power-of-two) block size for
the persistent kernels that keep a whole state space in one block.
"""
from __future__ import annotations

import importlib.util
import math
import os
import warnings

SIMULATOR_ENV = "NUMBA_ENABLE_CUDASIM"


def _gpu_visible() -> bool:
    if os.environ.get("RL_CUDA_SIMULATOR", "") not in ("", "0", "false"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch missing or broken: no GPU
        return False


SIMULATOR = not _gpu_visible()
if SIMULATOR:
    os.environ[SIMULATOR_ENV] = "1"
elif importlib.util.find_spec("cuda") is not None:
    # Numba 0.60's built-in ctypes driver binding crashes (segmentation fault at the first kernel launch) with
    # CUDA 13 drivers such as deepdish's; NVIDIA's own bindings (pip package cuda-python) work.
    os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

from numba import cuda  # noqa: E402  (must come after the environment variables are set)
from numba.core.errors import NumbaPerformanceWarning, NumbaWarning  # noqa: E402

# The persistent solver kernels run the whole (small) state space in one block on purpose; Numba's
# "grid size 1 ... low occupancy" warning is expected there.
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

if SIMULATOR:
    # Numba's simulator seeds its random generators with CPU fallbacks that emit harmless warnings.
    warnings.filterwarnings("ignore", category=NumbaWarning)
    warnings.filterwarnings("ignore", message="overflow encountered", category=RuntimeWarning)

MAX_THREADS = 1024  # threads per block on every CUDA device since compute capability 2.0
WARP = 32


def device_summary() -> str:
    if SIMULATOR:
        return "Numba CUDA simulator (CPU)"
    dev = cuda.get_current_device()
    cc = ".".join(map(str, dev.compute_capability))
    name = dev.name
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    name = str(name).split("\0", 1)[0].strip()  # NVIDIA's bindings return a NUL-padded buffer
    return f"{name} (compute capability {cc})"


def backend_name() -> str:
    return "cuda-simulator" if SIMULATOR else "cuda"


def blocks_for(n: int, threads: int) -> int:
    """Number of blocks so that blocks * threads >= n."""
    return max(1, math.ceil(int(n) / int(threads)))


def launch_1d(n: int, threads: int = 256) -> tuple[int, int]:
    """(blocks, threads per block) for one thread per element of a 1-D index space of size n."""
    return blocks_for(n, threads), int(threads)


def launch_2d(nx: int, ny: int, threads: tuple = (32, 4)) -> tuple[tuple, tuple]:
    """Blocks and threads for a 2-D index space (x fastest: consecutive threads get consecutive x)."""
    return (blocks_for(nx, threads[0]), blocks_for(ny, threads[1])), tuple(int(t) for t in threads)


def launch_3d(nx: int, ny: int, nz: int, threads: tuple = (16, 16, 1)) -> tuple[tuple, tuple]:
    return (blocks_for(nx, threads[0]), blocks_for(ny, threads[1]), blocks_for(nz, threads[2])), tuple(int(t) for t in threads)


def block_for_states(n: int, cap: int = MAX_THREADS) -> int | None:
    """Smallest power of two >= n (at least one warp) that fits in a block, or None if n is too large."""
    size = WARP
    while size < n:
        size *= 2
    return size if size <= cap else None
