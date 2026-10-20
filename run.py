#!/usr/bin/env python3
"""Single entry point for Assignment 1. `python run.py -h` lists the commands.

    python run.py dp                              Section 2: enumeration, value and policy iteration
    python run.py pg --part 1 --env cartpole      Policy gradient Part 1 / Part 2 (cartpole or pong)
    python run.py tests --section dp              validation experiments (dp, pg1, pg2 or all)
    python run.py report                          rebuild report/report.md
    python run.py check                           check packages, GPU and gym environments
    python run.py ui                              open the dashboard
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    from rl.dp import run as dp_run
    from rl.pg import run as pg_run

    parser = argparse.ArgumentParser(description="MLDS Assignment 1: reinforcement learning")
    sub = parser.add_subparsers(dest="command", required=True)
    dp_run.add_arguments(sub.add_parser("dp", help="Section 2: value and policy iteration"))
    pg_run.add_arguments(sub.add_parser("pg", help="policy gradient, Part 1 or Part 2"))
    tests = sub.add_parser("tests", help="run validation experiments")
    tests.add_argument("--section", choices=["dp", "pg1", "pg2", "all"], default="all")
    tests.add_argument("-k", "--match", help="only run tests whose name or title contains this text")
    tests.add_argument("--list", action="store_true", help="list the registered tests without running them")
    sub.add_parser("report", help="rebuild report/report.md from answers, figures and test results")
    sub.add_parser("check", help="check packages, GPU and gym environments on this machine")
    ui = sub.add_parser("ui", help="open the dashboard (Streamlit)")
    ui.add_argument("--port", type=int, default=8501)
    args = parser.parse_args(argv)

    if args.command == "dp":
        return dp_run.main(args)
    if args.command == "pg":
        return pg_run.main(args)
    if args.command == "tests":
        from tests import harness

        sections = None if args.section == "all" else [args.section]
        if args.list:
            return harness.list_tests(sections)
        return harness.run(sections, args.match)
    if args.command == "report":
        from report.build import build_report

        print(f"wrote {build_report().relative_to(ROOT)}")
        return 0
    if args.command == "check":
        from rl import check

        return check.main(args)
    if args.command == "ui":
        cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "ui" / "app.py"), "--server.port", str(args.port)]
        return subprocess.call(cmd, cwd=ROOT)
    return 2


if __name__ == "__main__":
    sys.exit(main())
