"""`python run.py report`: assemble report/report.md from the answers, figures and test results, then render it
to report/report.pdf. The report opens with the submission details, the run instructions (report/instructions.md)
and a table of contents, then follows the handout's order. An answer places a registered figure inline with
`{{figure:KEY}}` (report/answers.py); it is numbered in sequence with the figures shown above the answers.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from report import answers
from report.answers import FIGURE_TOKEN
from report.questions import SECTIONS, Question, questions_for
from rl.outputs import FIGURE_BY_KEY, RESULTS, Figure, read_json, tests_dir

REPORT_DIR = Path(__file__).resolve().parent
REPORT_FILE = REPORT_DIR / "report.md"
REPORT_PDF = REPORT_DIR / "report.pdf"
TITLE = "Assignment 1: Reinforcement Learning"
CONTENTS = "Contents"  # heading of the table of contents; report/pdf.py sets the list under it with page numbers
PENDING = "*Answer pending.*"
HEADING = re.compile(r"(#{1,6})\s+(.*)")


def _quote(text: str) -> list[str]:
    return [f"> {line}" if line else ">" for line in text.splitlines()]


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|") or "–"


def _demote(markdown: str) -> str:
    """Headings one level deeper, so a whole document fits under a report section."""
    lines, fenced = [], False
    for line in markdown.splitlines():
        fenced ^= line.lstrip().startswith("```")
        lines.append(f"#{line}" if not fenced and line.startswith("#") else line)
    return "\n".join(lines)


def _anchor(text: str, seen: dict[str, int]) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces to hyphens, -1, -2, … for repeats."""
    slug = re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", re.sub(r"[`*]", "", text).strip().lower()))
    n = seen.get(slug, -1) + 1
    seen[slug] = n
    return slug if n == 0 else f"{slug}-{n}"


def _contents(lines: list[str]) -> list[str]:
    """The table of contents: a linked list of the report's h2 and h3 headings, in order."""
    seen: dict[str, int] = {}
    entries, fenced = [], False
    for line in lines:
        fenced ^= line.lstrip().startswith("```")
        m = HEADING.match(line)
        if fenced or not m:
            continue
        level, text = len(m.group(1)), m.group(2).strip()
        anchor = _anchor(text, seen)
        if level in (2, 3) and text != CONTENTS:
            entries.append(f"{'  ' * (level - 2)}- [{text}](#{anchor})")
    return [f"## {CONTENTS}", "", *entries, ""]


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


def _figure(fig: Figure, fig_no: int, out_dir: Path) -> list[str]:
    if fig.file.exists():
        return [f"![{fig.alt}]({os.path.relpath(fig.file, out_dir)})", "", f"*Figure {fig_no}. {fig.caption}*", ""]
    return [f"*Figure {fig_no} (pending). {fig.caption} Generate it with `{fig.command}`.*", ""]


def _question(q: Question, text: str, fig_no: int, out_dir: Path) -> tuple[list[str], int]:
    lines = [f"### {q.title}", "", *_quote(q.prompt), ""]
    for key in q.figures:
        fig_no += 1
        lines += _figure(FIGURE_BY_KEY[key], fig_no, out_dir)
    for run in q.evals:
        stats = read_json(RESULTS / "pg" / run / "eval.json")
        if stats:
            lines += [f"**Evaluation over {stats['episodes']} episodes:** mean reward = {stats['mean']:.2f}, "
                      f"standard deviation = {stats['std']:.2f}.", ""]
        else:
            lines += ["**Evaluation over 500 episodes:** mean and standard deviation pending.", ""]
    if q.tests:
        lines += _tests_block(q.tests)
    if q.source:
        text = _demote(text)

    def inline_figure(m: re.Match) -> str:
        nonlocal fig_no
        fig = FIGURE_BY_KEY.get(m.group(1))
        if fig is None:
            return f"*Unknown figure `{m.group(1)}` (rl/outputs.py lists the keys).*"
        fig_no += 1
        return "\n".join(_figure(fig, fig_no, out_dir)).strip()

    lines += [FIGURE_TOKEN.sub(inline_figure, text).strip() or PENDING, ""]
    return lines, fig_no


def build_report(path: Path = REPORT_FILE) -> Path:
    data = answers.load()
    meta = data["meta"]
    lines = [f"# {TITLE}", ""]
    for key, label in answers.META_FIELDS.items():
        value = meta.get(key, "").strip()
        if key == "repo_url" and value:
            value = f"<{value}>"
        lines.append(f"**{label}:** {value or '*pending*'}  ")
    lines.append("")

    fig_no, contents_at = 0, len(lines)
    for section in SECTIONS:
        if section.key != "submission":
            lines += [f"## {section.title} ({section.points} points)", ""]
        for q in questions_for(section.key):
            block, fig_no = _question(q, answers.text(q, data), fig_no, path.parent)
            if section.key == "submission":
                block[0] = f"## {q.title}"
            lines += block
        if section.key == "submission":
            contents_at = len(lines)  # the table of contents follows the run instructions
    lines[contents_at:contents_at] = _contents(lines)
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path


def build_pdfs() -> list[Path]:
    """report/report.pdf from report.md (built first when it is missing). Needs reportlab."""
    from report.pdf import markdown_to_pdf

    author = answers.load()["meta"].get("name", "")
    if not REPORT_FILE.exists():
        build_report()
    return [markdown_to_pdf(REPORT_FILE.read_text(), REPORT_PDF, title=TITLE, author=author)]
