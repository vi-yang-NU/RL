#!/usr/bin/env python3
"""Single entry point for Assignment 1. `python run.py -h` lists the commands.

    python run.py dp                              Section 2: enumeration, value and policy iteration
    python run.py pg --part 1 --env cartpole      Policy gradient Part 1 / Part 2 (cartpole or pong)
    python run.py watch --env pong [--run part1_pong]   record one episode, frame by frame, as a viewer page
    python run.py play --env pong [--run part1_pong]    play the game live in a window: the policy at the controls, take over with the arrows
    python run.py tests --section dp              validation experiments (dp, pg1, pg2 or all)
    python run.py unittest [-k text]              unit tests and edge cases (tests/unit, seconds)
    python run.py report                          rebuild report/report.md and report.pdf
    python run.py check                           check packages, GPU and gym environments
    python run.py ui                              open the dashboard
    python run.py deepdish [-n N] [--connect]     deepdish settings (.env), ssh target and connection status; open the shared connection
    python run.py save [-m "message"]             commit every change and push it to GitHub, in one step
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
    from rl.pg import play as pg_play
    from rl.pg import watch as pg_watch
    from ui import deepdish

    parser = argparse.ArgumentParser(description="MLDS Assignment 1: reinforcement learning")
    sub = parser.add_subparsers(dest="command", required=True)
    dp_run.add_arguments(sub.add_parser("dp", help="Section 2: value and policy iteration"))
    pg_run.add_arguments(sub.add_parser("pg", help="policy gradient, Part 1 or Part 2"))
    pg_watch.add_arguments(sub.add_parser("watch", help="record one episode of CartPole-v1 or Pong-v5 frame by frame and write the viewer page"))
    pg_play.add_arguments(sub.add_parser("play", help="play CartPole-v1 or Pong-v5 live in a window with the learned policy at the controls; take over with the arrow keys"))
    tests = sub.add_parser("tests", help="run validation experiments")
    tests.add_argument("--section", choices=["dp", "pg1", "pg2", "all"], default="all")
    tests.add_argument("-k", "--match", help="only run tests whose name or title contains this text")
    tests.add_argument("--list", action="store_true", help="list the registered tests without running them")
    unit = sub.add_parser("unittest", help="run the unit tests in tests/unit (seconds; standard library unittest)")
    unit.add_argument("-k", "--match", action="append", help="only tests whose name contains this text (repeatable)")
    unit.add_argument("-v", "--verbose", action="store_true")
    sub.add_parser("report", help="rebuild report/report.md and report/report.pdf")
    sub.add_parser("check", help="check packages, GPU and gym environments on this machine")
    ui = sub.add_parser("ui", help="open the dashboard (Streamlit)")
    ui.add_argument("--port", type=int, default=8501)
    deepdish.add_arguments(sub.add_parser(
        "deepdish", help="deepdish settings (.env, DEEPDISH_* variables) and connection status; --connect opens the shared SSH connection"))
    save = sub.add_parser("save", help="commit every change and push it to GitHub, in one step")
    save.add_argument("-m", "--message", help="commit message (default: the date and the folders that changed)")
    save.add_argument("--no-push", action="store_true", help="commit on this computer only")
    save.add_argument("--no-report", action="store_true",
                      help="do not rebuild the report first, even when the answers or results are newer than it")
    args = parser.parse_args(argv)

    if args.command == "dp":
        return dp_run.main(args)
    if args.command == "pg":
        return pg_run.main(args)
    if args.command == "watch":
        return pg_watch.main(args)
    if args.command == "play":
        return pg_play.main(args)
    if args.command == "tests":
        from tests import harness

        sections = None if args.section == "all" else [args.section]
        if args.list:
            return harness.list_tests(sections)
        return harness.run(sections, args.match)
    if args.command == "unittest":
        import unittest

        loader = unittest.TestLoader()
        if args.match:
            loader.testNamePatterns = [f"*{m}*" for m in args.match]
        suite = loader.discover(str(ROOT / "tests" / "unit"), top_level_dir=str(ROOT))
        result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
        return 0 if result.wasSuccessful() else 1
    if args.command == "report":
        from report.build import build_pdfs, build_report

        print(f"wrote {build_report().relative_to(ROOT)}")
        try:
            pdfs = build_pdfs()
        except ImportError as e:
            print(f"skipped the PDF: {e} (pip install -r requirements-ui.txt)")
            return 1
        for path in pdfs:
            print(f"wrote {path.relative_to(ROOT)}")
        return 0
    if args.command == "check":
        from rl import check

        return check.main(args)
    if args.command == "ui":
        return run_ui(args.port)
    if args.command == "deepdish":
        return deepdish.main(args)
    if args.command == "save":
        from rl.save import save as save_all

        return save_all(ROOT, args.message, push=not args.no_push, report=not args.no_report)
    return 2


def run_ui(port: int) -> int:
    """Serve the dashboard on this machine only and open it in the browser.

    Headless mode skips Streamlit's first-run email prompt, which would otherwise wait for an answer in the
    terminal. Over ssh (e.g. VS Code Remote-SSH on deepdish) no browser is opened; forward the port instead.
    Whatever goes wrong, the message says what to do next; the report is the fallback when the page will not load.
    """
    import importlib.util
    import os
    import time
    import webbrowser

    url = f"http://localhost:{port}"
    fallback = (
        "If the page stays blank or shows a connection error: reload it once this line has appeared, open the URL\n"
        f"by hand, or start the dashboard on another port (python run.py ui --port {port + 1}). It is not needed to\n"
        "grade: report/report.pdf has every answer, plot and test result."
    )
    if importlib.util.find_spec("streamlit") is None:
        print(f"The dashboard needs Streamlit: pip install -r requirements-ui.txt\n{fallback}", file=sys.stderr)
        return 1
    if _listening(port):
        print(f"Port {port} is already in use, perhaps by a dashboard that is already open at {url}.\n"
              f"Open that one, or start this one on another port: python run.py ui --port {port + 1}", file=sys.stderr)
        return 1
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "ui" / "app.py"), "--server.port", str(port),
           "--server.address", "localhost", "--server.headless", "true"]
    proc = subprocess.Popen(cmd, cwd=ROOT)
    try:
        for _ in range(120):
            if proc.poll() is not None:
                if proc.returncode:
                    print(f"The dashboard did not start (exit code {proc.returncode}); the error is above.\n{fallback}",
                          file=sys.stderr)
                return proc.returncode
            if _listening(port):
                break
            time.sleep(0.5)
        else:
            print("The dashboard is taking more than a minute to start (the first start imports PyTorch); "
                  "keep waiting, then reload the page.", flush=True)
        print(f"Dashboard: {url}  (Ctrl+C stops it)\n{fallback}", flush=True)
        if not os.environ.get("SSH_CONNECTION"):
            webbrowser.open(url)
        return proc.wait()
    except KeyboardInterrupt:
        return proc.wait()


def _listening(port: int) -> bool:
    import socket

    try:
        socket.create_connection(("localhost", port), timeout=0.5).close()
        return True
    except OSError:
        return False


if __name__ == "__main__":
    sys.exit(main())
