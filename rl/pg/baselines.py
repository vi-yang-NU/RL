"""Baselines for variance reduction (Part 2)."""
from __future__ import annotations

from typing import Any

# Baseline that `python run.py pg --part 2` uses when --baseline is not given.
# TODO(Part 2): set this to the name of the baseline you implement below.
DEFAULT_BASELINE: str | None = None


def make_baseline(name: str, cfg: Any, env: Any) -> Any:
    """Return the baseline object the training loop uses, or None for "none" (Part 1).

    TODO(Part 2): implement the constant or non-constant baseline of your choice.
    """
    if name == "none":
        return None
    raise NotImplementedError(f"baseline '{name}' (rl/pg/baselines.py)")
