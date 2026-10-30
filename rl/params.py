"""Overriding experiment parameters from the command line (`--set KEY=VALUE`).

Both `python run.py dp` and `python run.py pg` accept repeatable `--set KEY=VALUE` options. Values are parsed
as JSON when possible (so `--set capacity=10`, `--set arrivals=[1,2,3]`, `--set holding_charged=before_dispatch`
all work), and `apply_overrides` copies a dataclass with those fields replaced, rejecting unknown names.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """`["a=1", "b=[1,2]", "c=text"]` -> `{"a": 1, "b": [1, 2], "c": "text"}`."""
    overrides: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        try:
            overrides[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key.strip()] = value
    return overrides


def apply_overrides(obj: Any, overrides: dict[str, Any]) -> Any:
    """A copy of dataclass `obj` with `overrides` applied; values are coerced to the field's default type."""
    if not overrides:
        return obj
    fields = {f.name: f for f in dataclasses.fields(obj)}
    unknown = sorted(set(overrides) - set(fields))
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown}; choose from {sorted(fields)}")
    coerced = {name: _coerce(getattr(obj, name), value) for name, value in overrides.items()}
    return dataclasses.replace(obj, **coerced)


def _coerce(default: Any, value: Any) -> Any:
    if value is None or default is None:
        return value
    if isinstance(default, tuple) and isinstance(value, (list, tuple)):
        return tuple(value)
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value)
        return value
    if isinstance(default, float) and isinstance(value, (int, str)):
        try:
            return float(value)
        except ValueError:
            return value
    return value
