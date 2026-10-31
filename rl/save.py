"""`python run.py save`: commit every change in this repository and push it to GitHub, in one step.

It is `git add -A && git commit && git push` with the checks a one-step save needs:

* the report is rebuilt first when the answers or the results are newer than it, so GitHub shows the same
  answers as the dashboard (`--no-report` skips this);
* files GitHub would refuse (over 50 MB) stop the save before anything is committed;
* when GitHub has commits this computer does not have (an edit made on github.com, a save from deepdish),
  they are pulled with a rebase and the push is tried again;
* if the push still fails, the commit stays on this computer and the message says what to fix.
"""
from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, TextIO

LIMIT_BYTES = 50 * 1024 * 1024  # GitHub warns at 50 MB and refuses files over 100 MB
REPORT_INPUTS = (".json", ".png", ".svg")  # the answers and results the report is built from


def git(root: Path, *args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def save(root: Path, message: str | None = None, *, push: bool = True, remote: str = "origin", report: bool = True,
         limit_bytes: int = LIMIT_BYTES, out: TextIO = sys.stdout) -> int:
    """Stage, commit and push everything under `root`. Returns a process exit code (0 = saved)."""

    def say(text: str) -> None:
        print(text, file=out, flush=True)

    top = git(root, "rev-parse", "--show-toplevel")
    if top.returncode:
        say(f"{root} is not inside a git repository: {top.stderr.strip()}")
        return 1
    ref = git(root, "symbolic-ref", "--short", "-q", "HEAD")
    if ref.returncode:
        say("HEAD is not on a branch (detached). Run `git switch main` first, then save again.")
        return 1
    branch = ref.stdout.strip()
    if report:
        refresh_report(root, say)

    add = git(root, "add", "-A")
    if add.returncode:
        say(add.stderr.strip())
        return 1
    staged = staged_files(root)
    too_big = [(path, (root / path).stat().st_size) for path in staged
               if (root / path).is_file() and (root / path).stat().st_size > limit_bytes]
    if too_big:
        git(root, "reset", "-q")
        say("Not saved: GitHub refuses files this large. Move them out of the repository or add them to .gitignore:")
        for path, size in too_big:
            say(f"  {size / 1e6:,.0f} MB  {path}")
        return 1

    if staged:
        text = message or auto_message(staged)
        commit = git(root, "commit", "-q", "-m", text)
        if commit.returncode:
            say(commit.stderr.strip() or commit.stdout.strip())
            return 1
        sha = git(root, "rev-parse", "--short", "HEAD").stdout.strip()
        say(f"committed {sha} on {branch}: {text.splitlines()[0]}")
    else:
        say("nothing new to commit")
    if not push:
        return 0
    url = git(root, "remote", "get-url", remote)
    if url.returncode:
        say(f"No remote named {remote!r}. Create the repository on GitHub, then\n"
            f"  git remote add {remote} git@github.com:<you>/<repo>.git\nand save again.")
        return 1
    return push_branch(root, remote, branch, url.stdout.strip(), say)


def push_branch(root: Path, remote: str, branch: str, url: str, say: Callable[[str], None]) -> int:
    pushed = git(root, "push", "-u", remote, branch)
    if pushed.returncode and re.search(r"rejected|non-fast-forward|fetch first", pushed.stderr):
        say(f"{remote} has commits this computer does not have; pulling them first")
        pulled = git(root, "pull", "--rebase", remote, branch)
        if pulled.returncode:
            git(root, "rebase", "--abort")
            say(pulled.stderr.strip() or pulled.stdout.strip())
            say("Both sides changed the same lines. Run `git pull --rebase`, resolve the conflicts, then save again.")
            return 1
        pushed = git(root, "push", "-u", remote, branch)
    page = github_page(url)
    if pushed.returncode == 0:
        unchanged = "Everything up-to-date" in pushed.stderr  # git reports this on stderr
        where = f"{page}/tree/{branch}" if page else f"{remote}/{branch}"
        if unchanged:
            say(f"GitHub already had everything: {where}")
        else:
            sha = git(root, "rev-parse", "HEAD").stdout.strip()
            say(f"pushed to {where}")
            if page:
                say(f"this commit: {page}/commit/{sha}")
        return 0
    say(pushed.stderr.strip())
    if re.search(r"403|Write access|Authentication failed|Permission denied|could not read Username", pushed.stderr):
        if page and url.startswith("http"):
            say("GitHub refused the stored HTTPS credentials. If you use SSH keys with GitHub (`gh auth status` says so):\n"
                f"  git remote set-url {remote} git@github.com:{page[len('https://github.com/'):]}.git")
        else:
            say("Check your GitHub access: `gh auth status`, or `ssh -T git@github.com`.")
    say("Your work is committed on this computer; run `python run.py save` again once the push works.")
    return 1


def staged_files(root: Path) -> list[str]:
    return [p for p in git(root, "diff", "--cached", "--name-only", "-z").stdout.split("\0") if p]


def auto_message(paths: list[str]) -> str:
    """'Save 2026-10-30 18:05: 23 files (rl, ui, tests, report)': the areas touched, most files first."""
    areas = Counter(p.split("/", 1)[0] for p in paths)
    top = [name for name, _ in areas.most_common(5)]
    more = f" +{len(areas) - len(top)} more" if len(areas) > len(top) else ""
    n = len(paths)
    return f"Save {dt.datetime.now():%Y-%m-%d %H:%M}: {n} file{'s' if n != 1 else ''} ({', '.join(top)}{more})"


def github_page(url: str) -> str | None:
    """https://github.com/owner/repo for an SSH or HTTPS GitHub remote URL; None for anything else."""
    m = re.match(r"^(?:git@github\.com:|ssh://git@github\.com/|https://github\.com/)([^/]+)/([^/]+?)(?:\.git)?/?$", url.strip())
    return f"https://github.com/{m.group(1)}/{m.group(2)}" if m else None


def refresh_report(root: Path, say: Callable[[str], None]) -> None:
    """Rebuild report/report.md and report.pdf when the answers or results they are built from are newer."""
    report_md = root / "report" / "report.md"
    inputs = [root / "report" / "answers.json"]
    if (root / "results").is_dir():
        inputs += [p for p in (root / "results").rglob("*") if p.suffix in REPORT_INPUTS]
    inputs = [p for p in inputs if p.is_file()]
    if not inputs or not (root / "report" / "build.py").is_file():
        return
    if report_md.exists() and report_md.stat().st_mtime >= max(p.stat().st_mtime for p in inputs):
        return
    say("the answers or results are newer than the report; rebuilding report/report.md and report.pdf")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from report.build import build_pdfs, build_report

        build_report()
        try:
            build_pdfs()
        except ImportError as e:
            say(f"  skipped the PDF: {e} (pip install -r requirements-ui.txt)")
    except Exception as e:  # noqa: BLE001 - saving must not depend on the report build
        say(f"  could not rebuild the report ({e}); saving what is there")
