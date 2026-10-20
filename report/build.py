"""`python run.py report`: assemble report/report.md from the answers, figures and test results.

Convert it to the PDF for Canvas with e.g. `cd report && pandoc report.md -o report.pdf`.
"""
from __future__ import annotations

import os
from pathlib import Path

from report import answers
from report.questions import SECTIONS, Question, questions_for
from rl.outputs import FIGURE_BY_KEY, RESULTS, read_json, tests_dir

REPORT_FILE = Path(__file__).resolve().parent / "report.md"
PENDING = "*Answer pending.*"


def _quote(text: str) -> list[str]:
    return [f"> {line}" if line else ">" for line in text.splitlines()]


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|") or "–"


def _tests_block(section: str) -> list[str]:
    data = read_json(tests_dir() / f"{section}.json")
    command = f"`python run.py tests --section {section}`"
    if not data or not data.get("tests"):
        return [f"*No validation results yet. Add experiments to tests/ and run {command}.*", ""]
    lines = [
        f"Validation results from {command} (last run {data['updated']} on {data['host']}):",
        "",
        "| Test | What it does | How it validates | Result |",
        "|---|---|---|---|",
    ]
    for t in data["tests"]:
        passed = sum(c["passed"] for c in t["checks"])
        result = t["status"].upper() + (f" ({passed}/{len(t['checks'])} checks)" if t["checks"] else "")
        lines.append(f"| {_cell(t['title'])} | {_cell(t['what'])} | {_cell(t['how'])} | {result} |")
    lines.append("")
    for t in data["tests"]:
        if t["metrics"]:
            metrics = ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in t["metrics"].items())
            lines.append(f"- **{t['title']}**: {metrics}")
    return lines + [""]


def _question(q: Question, text: str, fig_no: int, out_dir: Path) -> tuple[list[str], int]:
    lines = [f"### {q.title}", "", *_quote(q.prompt), ""]
    for key in q.figures:
        fig = FIGURE_BY_KEY[key]
        fig_no += 1
        if fig.file.exists():
            lines += [f"![]({os.path.relpath(fig.file, out_dir)})", "", f"*Figure {fig_no}. {fig.caption}*", ""]
        else:
            lines += [f"*Figure {fig_no} (pending). {fig.caption} Generate it with `{fig.command}`.*", ""]
    for run in q.evals:
        stats = read_json(RESULTS / "pg" / run / "eval.json")
        if stats:
            lines += [f"**Evaluation over {stats['episodes']} episodes:** mean reward = {stats['mean']:.2f}, "
                      f"standard deviation = {stats['std']:.2f}.", ""]
        else:
            lines += ["**Evaluation over 500 episodes:** mean and standard deviation pending.", ""]
    if q.tests:
        lines += _tests_block(q.tests)
    lines += [text.strip() or PENDING, ""]
    return lines, fig_no


def build_report(path: Path = REPORT_FILE) -> Path:
    data = answers.load()
    meta, text = data["meta"], data["answers"]
    lines = ["# Assignment 1: Reinforcement Learning", ""]
    for key, label in answers.META_FIELDS.items():
        value = meta.get(key, "").strip()
        if key == "repo_url" and value:
            value = f"<{value}>"
        lines.append(f"**{label}:** {value or '*pending*'}  ")
    lines.append("")

    fig_no = 0
    for section in SECTIONS:
        if section.key != "submission":
            lines += [f"## {section.title} ({section.points} points)", ""]
        for q in questions_for(section.key):
            block, fig_no = _question(q, text.get(q.id, ""), fig_no, path.parent)
            if section.key == "submission":
                block[0] = f"## {q.title}"
            lines += block
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path
