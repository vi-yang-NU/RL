"""`python run.py check`: confirm this machine (laptop or deepdish) can run every part."""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import socket


def _version(module: str) -> str | None:
    try:
        return getattr(importlib.import_module(module), "__version__", "installed")
    except Exception:  # noqa: BLE001 - any import failure means "not usable here"
        return None


def main(args: argparse.Namespace) -> int:
    ok = True
    print(f"host      {socket.gethostname()}")
    print(f"python    {platform.python_version()} ({platform.platform()})")
    for module in ("numpy", "matplotlib", "torch", "gymnasium", "ale_py"):
        version = _version(module)
        ok &= version is not None
        print(f"{module:<9} {version or 'MISSING (run ./setup.sh)'}")

    if _version("torch"):
        import torch

        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        print(f"cuda      {'available' if torch.cuda.is_available() else 'not available (CPU only)'}"
              f"; CUDA_VISIBLE_DEVICES={visible if visible is not None else '(unset)'}")
        for i in range(torch.cuda.device_count()):
            print(f"  gpu {i}   {torch.cuda.get_device_name(i)} (compute capability {'.'.join(map(str, torch.cuda.get_device_capability(i)))})")
        if torch.cuda.is_available():
            try:
                (torch.ones(64, 64, device="cuda") @ torch.ones(64, 64, device="cuda")).sum().item()
                print("cuda      test computation on the GPU: ok")
            except RuntimeError as e:
                ok = False
                print(f"cuda      test computation FAILED: {str(e).splitlines()[0]}")
                print("          this torch build does not support the GPU; rerun setup.sh (it picks a matching build)")
        if torch.cuda.device_count() > 1:
            print("  note: deepdish jobs must use a single GPU; jobs started via the dashboard or scripts/jobs.sh see one card.")

    if _version("gymnasium"):
        from rl.pg.envs import ENV_IDS, make_env
        from rl.pg.watch import headless_pygame

        headless_pygame()
        for name, env_id in ENV_IDS.items():
            try:
                env = make_env(name, render_mode="rgb_array")  # frames are rendered for the dashboard's game viewer
                env.reset(seed=0)
                env.step(env.action_space.sample())
                frame = env.render()
                env.close()
                print(f"env       {env_id}: ok (renders {frame.shape[1]}x{frame.shape[0]} frames)")
            except Exception as e:  # noqa: BLE001
                ok = False
                hint = " - pip install 'gymnasium[classic-control]'" if "pygame" in str(e) else ""
                print(f"env       {env_id}: FAILED ({type(e).__name__}: {e}){hint}")

    print("ready" if ok else "some checks failed")
    return 0 if ok else 1
