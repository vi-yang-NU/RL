"""Validation-experiment harness.

For every part, the handout asks for experiments showing the algorithm behaves as
predicted, documented with what each test does and how it validates, plus quantitative
evidence. This harness runs those experiments and records the evidence:

  * Register an experiment with @validation_test(section=..., title=..., what=..., how=...).
    `section` is "dp", "pg1" or "pg2"; `what` and `how` are the documentation the
    dashboard shows and the report copies.
  * Inside the experiment, record evidence on the context it receives:
        ctx.check(name, condition, detail)          pass/fail assertion (does not stop the test)
        ctx.check_close(name, actual, expected, tol)
        ctx.metric(name, value)                     a number worth reporting
        ctx.table(name, rows)                       list of dicts, e.g. a step-by-step trace
        ctx.figure(fig_or_path, name)               a matplotlib figure or an existing image
        ctx.note(text)                              free-form observation
  * `python run.py tests --section dp` runs them and writes results/tests/dp.json.

Experiments live in tests/test_*.py.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import platform
import shutil
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Iterable

from rl.outputs import ROOT, SECTIONS, read_json, tests_dir, write_json

STATUSES = ("passed", "failed", "error", "recorded")  # recorded = ran without checks, evidence only


@dataclass(frozen=True)
class TestSpec:
    section: str
    name: str
    title: str
    what: str
    how: str
    func: Callable[["TestContext"], Any]
    tags: tuple[str, ...] = ()


_REGISTRY: dict[str, TestSpec] = {}


def validation_test(section: str, *, title: str | None = None, what: str = "", how: str = "", tags: Iterable[str] = ()):
    """Register the decorated function as a validation experiment for `section`."""
    if section not in SECTIONS:
        raise ValueError(f"section must be one of {list(SECTIONS)}, got {section!r}")

    def register(func: Callable[[TestContext], Any]) -> Callable[[TestContext], Any]:
        spec = TestSpec(
            section=section,
            name=func.__name__,
            title=title or func.__name__.replace("_", " ").capitalize(),
            what=dedent(what).strip(),
            how=dedent(how).strip(),
            func=func,
            tags=tuple(tags),
        )
        _REGISTRY[f"{section}/{spec.name}"] = spec
        return func

    return register


class TestContext:
    """Collects the evidence one experiment produces."""

    def __init__(self, spec: TestSpec, out_dir: Path):
        self.spec = spec
        self.out_dir = out_dir
        self.checks: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.figures: list[str] = []
        self.notes: list[str] = []

    def check(self, name: str, condition: Any, detail: Any = "") -> bool:
        passed = bool(condition)
        self.checks.append({"name": name, "passed": passed, "detail": str(detail)})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f": {detail}" if detail != "" else ""))
        return passed

    def check_close(self, name: str, actual: Any, expected: Any, tol: float) -> bool:
        import numpy as np

        diff = float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))))
        return self.check(name, diff <= tol, f"max |actual - expected| = {diff:.3g} (tolerance {tol:g})")

    def metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value.item() if hasattr(value, "item") else value

    def table(self, name: str, rows: Any) -> None:
        if hasattr(rows, "to_dict"):  # pandas DataFrame
            rows = rows.to_dict("records")
        self.tables[name] = [dict(r) for r in rows]

    def figure(self, fig_or_path: Any, name: str | None = None) -> Path:
        """Save a matplotlib figure (or copy an existing image) into this test's output folder."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(fig_or_path, "savefig"):
            from rl.plotting import save

            path = save(fig_or_path, self.out_dir / f"{name or f'figure_{len(self.figures) + 1}'}.png")
        else:
            src = Path(fig_or_path)
            path = self.out_dir / f"{name or src.stem}{src.suffix}"
            if src.resolve() != path.resolve():
                shutil.copyfile(src, path)
        self.figures.append(str(path.relative_to(ROOT)))
        return path

    def note(self, text: str) -> None:
        self.notes.append(str(text))


class _Tee(io.TextIOBase):
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, s: str) -> int:
        for stream in self.streams:
            stream.write(s)
        return len(s)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def discover() -> list[TestSpec]:
    """Import every tests/test_*.py module so their experiments register."""
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        importlib.import_module(f"tests.{path.stem}")
    return list(_REGISTRY.values())


def _run_one(spec: TestSpec) -> dict[str, Any]:
    ctx = TestContext(spec, tests_dir() / spec.section / spec.name)
    output = io.StringIO()
    error = None
    start = time.time()
    print(f"-- {spec.section}/{spec.name}: {spec.title}")
    try:
        with contextlib.redirect_stdout(_Tee(sys.stdout, output)):
            spec.func(ctx)
    except Exception:  # noqa: BLE001 - a crashing experiment is reported, not fatal
        error = traceback.format_exc()
        print(error)
    if error:
        status = "error"
    elif not ctx.checks:
        status = "recorded"
    else:
        status = "passed" if all(c["passed"] for c in ctx.checks) else "failed"
    duration = time.time() - start
    print(f"   {status.upper()} in {duration:.1f}s")
    return {
        "name": spec.name,
        "title": spec.title,
        "what": spec.what,
        "how": spec.how,
        "tags": list(spec.tags),
        "status": status,
        "checks": ctx.checks,
        "metrics": ctx.metrics,
        "tables": ctx.tables,
        "figures": ctx.figures,
        "notes": ctx.notes,
        "error": error,
        "stdout": output.getvalue()[-20_000:],
        "duration_s": round(duration, 2),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def run(sections: Iterable[str] | None = None, match: str | None = None) -> int:
    """Run the selected experiments, write results/tests/<section>.json, return an exit code."""
    specs = discover()
    exit_code = 0
    for section in sections or SECTIONS:
        registered = [s for s in specs if s.section == section]
        chosen = [s for s in registered if not match or match.lower() in f"{s.name} {s.title}".lower()]
        print(f"== {SECTIONS[section]}: {len(chosen)} test(s)")
        if not registered:
            print(f"   no tests yet: add them to tests/test_{'dp' if section == 'dp' else 'pg_part' + section[-1]}.py")

        outcomes = {s.name: _run_one(s) for s in chosen}
        # A filtered run keeps earlier results for the tests it skipped.
        previous = {t["name"]: t for t in (read_json(tests_dir() / f"{section}.json", {}) or {}).get("tests", [])}
        tests = [outcomes.get(s.name) or previous.get(s.name) for s in registered]
        tests = [t for t in tests if t]
        write_json(
            tests_dir() / f"{section}.json",
            {
                "section": section,
                "title": SECTIONS[section],
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "host": socket.gethostname(),
                "python": platform.python_version(),
                "tests": tests,
            },
        )
        counts = {status: sum(t["status"] == status for t in outcomes.values()) for status in STATUSES}
        if outcomes:
            print("   " + ", ".join(f"{n} {status}" for status, n in counts.items() if n))
        if counts["failed"] or counts["error"]:
            exit_code = 1
    return exit_code


def list_tests(sections: Iterable[str] | None = None) -> int:
    specs = discover()
    for section in sections or SECTIONS:
        print(f"{SECTIONS[section]} ({section})")
        for spec in (s for s in specs if s.section == section):
            print(f"  {spec.name:<32} {spec.title}")
    return 0


def load_results(section: str) -> dict[str, Any] | None:
    return read_json(tests_dir() / f"{section}.json")
