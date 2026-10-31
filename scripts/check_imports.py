"""Check that the project only uses modules installable with pip/conda (the course's open-source rule).

    python scripts/check_imports.py            # tables in Markdown, then PASS or FAIL
    python scripts/check_imports.py --quiet    # only the verdict

What it checks, for every Python file in the repository (git's tracked and new files; the virtual environment,
caches and Lean build folders are never included):

1. Every `import` / `from ... import` at any depth is parsed with `ast` and its top-level module is classified as
   the Python standard library, one of the project's own packages, or a third-party module. A third-party module
   must be installed in the active environment's site-packages, and its distribution is looked up with
   importlib.metadata: version, installer and where it was installed from (`direct_url.json` exists only for
   installs from a URL, a git repository or a local folder; without it the package came from the package index).
2. Each third-party distribution is either listed in requirements*.txt or required by one that is.
3. requirements*.txt name packages only: no git+ / URL / local-path / editable entries and no extra package indexes.
4. No source file (.py, .js, .html, .sh, .css) carries a third-party license header or a "copied/adapted from"
   note, contains distinctive lines of widely copied REINFORCE code (Karpathy's pg-pong, the PyTorch example,
   Spinning Up, Stable-Baselines), or loads a script or stylesheet from outside the repository.

Exit status 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import subprocess
import sys
import sysconfig
from collections import defaultdict
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", ".lake", "node_modules", "logs", "results"}
LICENSE_HEADER = re.compile(r"copyright|SPDX-License|licensed under|MIT License|Apache License|GNU General Public", re.I)
ATTRIBUTION = re.compile(r"adapted from|copied from|taken from|borrowed from|stackoverflow\.com|gist\.github\.com", re.I)
# Distinctive names from widely copied REINFORCE code: Karpathy's pg-pong.py, the PyTorch examples' reinforce.py,
# OpenAI Spinning Up's simple policy gradient, Stable-Baselines.
KNOWN_CODE = re.compile(r"saved_log_probs|def finish_episode|np\.finfo\(np\.float32\)\.eps|self\.affine1|def discount_rewards|"
                        r"def policy_forward|def policy_backward|dlogps|rmsprop_cache|from spinup|import spinup|stable_baselines")
EXTERNAL_ASSET = re.compile(r"<script[^>]+src=[\"']?(https?:)?//|<link[^>]+href=[\"']?(https?:)?//|from\s+[\"']https?://|import\(\s*[\"']https?://")
NON_INDEX_REQUIREMENT = re.compile(r"git\+|https?://|^-e\b|^--?(index-url|extra-index-url|find-links)|file:|^\.{0,2}/")


def repo_files(suffixes: tuple[str, ...]) -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = [ROOT / f for f in out.splitlines()]
    except (OSError, subprocess.CalledProcessError):
        files = [p for p in ROOT.rglob("*") if p.is_file()]
    return sorted(p for p in files if p.suffix in suffixes and p.exists() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS))


def imports(files: list[Path]) -> dict[str, set[str]]:
    """Top-level module -> files importing it (absolute imports only; relative imports are the project's own)."""
    uses: dict[str, set[str]] = defaultdict(set)
    for f in files:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"), filename=str(f))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                uses[name.split(".")[0]].add(str(f.relative_to(ROOT)))
    return uses


def project_modules(files: list[Path]) -> set[str]:
    """The repository's own packages (top-level folders with Python files) and module names (for sibling imports)."""
    return {p.relative_to(ROOT).parts[0] for p in files if len(p.relative_to(ROOT).parts) > 1} | {p.stem for p in files}


def classify(module: str) -> str:
    if module in sys.builtin_module_names:
        return "stdlib"
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return "missing"
    location = spec.origin if spec.origin not in (None, "namespace", "built-in", "frozen") else next(iter(spec.submodule_search_locations or []), "")
    if spec.origin in ("built-in", "frozen"):
        return "stdlib"
    path = str(Path(location).resolve())
    paths = sysconfig.get_paths()
    if any(path.startswith(str(Path(paths[k]).resolve())) for k in ("purelib", "platlib")):
        return "site-packages"
    if path.startswith(str(Path(paths["stdlib"]).resolve())) or path.startswith(str(Path(paths["platstdlib"]).resolve())):
        return "stdlib"
    return f"elsewhere ({path})"


def distributions_by_module() -> dict[str, list[metadata.Distribution]]:
    found: dict[str, list[metadata.Distribution]] = defaultdict(list)
    for dist in metadata.distributions():
        tops = (dist.read_text("top_level.txt") or "").split()
        if not tops:
            tops = sorted({str(f).split("/")[0].split(".")[0] for f in (dist.files or [])
                           if not str(f).startswith("..") and ".dist-info" not in str(f) and "__pycache__" not in str(f)})
        for top in tops:
            found[top].append(dist)
    return found


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_names(lines: list[str]) -> dict[str, set[str]]:
    """Package name -> the extras requested for it, e.g. {"gymnasium": {"atari", "classic-control"}}."""
    names: dict[str, set[str]] = {}
    for line in lines:
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[([^\]]*)\])?", line)
        if m and not line.lstrip().startswith(("#", "-")):
            extras = {normalize(e) for e in (m.group(2) or "").split(",") if e.strip()}
            names.setdefault(normalize(m.group(1)), set()).update(extras)
    return names


def required_by(name: str, listed: dict[str, set[str]]) -> str | None:
    """A requirement-file package whose metadata requires `name`, unconditionally or through an extra the
    requirements actually ask for."""
    for parent, extras in listed.items():
        try:
            requires = metadata.distribution(parent).requires or []
        except metadata.PackageNotFoundError:
            continue
        for req in requires:
            m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)(.*)", req)
            if not m or normalize(m.group(1)) != name:
                continue
            extra = re.search(r"extra\s*==\s*[\"']([^\"']+)", m.group(2))
            if extra is None:
                return parent
            if normalize(extra.group(1)) in extras:
                return f"{parent}[{extra.group(1)}]"
    return None


def run(quiet: bool = False) -> int:
    say = (lambda *_: None) if quiet else print
    problems: list[str] = []
    py_files = repo_files((".py",))
    uses = imports(py_files)
    own = project_modules(py_files)
    by_module = distributions_by_module()

    req_files = sorted(ROOT.glob("requirements*.txt"), key=lambda p: (p.name != "requirements.txt", p.name))
    req_lines = {p.name: [line.strip() for line in p.read_text().splitlines() if line.strip() and not line.strip().startswith("#")] for p in req_files}
    listed: dict[str, set[str]] = {}
    for lines in req_lines.values():
        for n, extras in requirement_names(lines).items():
            listed.setdefault(n, set()).update(extras)

    stdlib, project, third, missing = [], [], {}, []
    for module in sorted(uses):
        if module in own:
            project.append(module)
            continue
        kind = classify(module)
        if kind == "stdlib":
            stdlib.append(module)
        elif kind == "site-packages" and by_module.get(module):
            third[module] = by_module[module][0]
        elif kind == "missing":
            missing.append(module)  # e.g. the dashboard's packages on a training server: reported, checked where installed
        else:
            problems.append(f"`{module}` is not the standard library, the project, or an installed pip package ({kind})")

    say(f"Python {sys.version.split()[0]} at {sys.prefix}")
    say(f"Scanned {len(py_files)} Python files: {len(uses)} distinct top-level imports — {len(stdlib)} standard library, "
        f"{len(project)} project, {len(third)} third-party" + (f", {len(missing)} not installed in this environment" if missing else "") + ".\n")
    say("| Module | Package | Version | Installed by | Source | In requirements | Imported in |")
    say("|---|---|---|---|---|---|---|")
    for module, dist in third.items():
        name = normalize(dist.metadata["Name"])
        direct = dist.read_text("direct_url.json")
        source = "package index" if not direct else json.loads(direct).get("url", "direct URL")
        if direct:
            problems.append(f"`{name}` was installed from {source}, not from the package index")
        listing = "listed" if name in listed else (f"via {required_by(name, listed)}" if required_by(name, listed) else "NOT LISTED")
        if listing == "NOT LISTED":
            problems.append(f"`{name}` is imported but neither listed in requirements*.txt nor required by a listed package")
        files = sorted(uses[module])
        shown = ", ".join(files[:2]) + (f" and {len(files) - 2} more" if len(files) > 2 else "")
        installer = (dist.read_text("INSTALLER") or "?").strip()
        say(f"| `{module}` | {dist.metadata['Name']} | {dist.version} | {installer} | {source} | {listing} | {shown} |")
    say(f"\nStandard library: {', '.join(stdlib)}.")
    say(f"\nProject packages and modules: {', '.join(project)}.")
    if missing:
        say("\nNot installed in this environment (run the check where they are installed to verify them): "
            + "; ".join(f"`{m}` (imported in {', '.join(sorted(uses[m]))})" for m in missing) + ".")

    say("\n| Requirements file | Entries | Non-index entries |")
    say("|---|---|---|")
    for fname, lines in req_lines.items():
        bad = [line for line in lines if NON_INDEX_REQUIREMENT.search(line)]
        problems += [f"{fname}: `{line}` does not come from the package index" for line in bad]
        say(f"| {fname} | {', '.join(f'`{line}`' for line in lines)} | {', '.join(bad) or 'none'} |")

    # this file names the patterns it searches for, so it is left out of the search
    text_files = [f for f in repo_files((".py", ".js", ".html", ".sh", ".css")) if f.resolve() != Path(__file__).resolve()]
    hits = defaultdict(list)
    for f in text_files:
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for label, pattern in (("license header", LICENSE_HEADER), ("attribution", ATTRIBUTION),
                                   ("external script or stylesheet", EXTERNAL_ASSET), ("line from well-known REINFORCE code", KNOWN_CODE)):
                if pattern.search(line):
                    hits[label].append(f"{f.relative_to(ROOT)}:{n}")
    say("\n| Source-file search | Files searched | Matches |")
    say("|---|---|---|")
    for label in ("license header", "attribution", "external script or stylesheet", "line from well-known REINFORCE code"):
        say(f"| {label} | {len(text_files)} | {', '.join(hits[label]) or 'none'} |")
        problems += [f"{label} at {where}" for where in hits[label]]

    verdict = ("PASS: every module is the standard library, the project's own code, or a package installed from a package "
               "index with pip") + (f" ({len(missing)} module(s) not installed here, so not checked)" if missing else "")
    print("\n" + verdict if not problems else "\nFAIL:\n" + "\n".join(f"- {p}" for p in problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quiet", action="store_true", help="print only the verdict")
    sys.exit(run(parser.parse_args().quiet))
