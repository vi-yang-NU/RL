"""Discussion answers and submission details, saved by the dashboard to report/answers.json.

A question with a `source` (the run instructions) is answered by that Markdown file in report/ instead.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from report.questions import Question
from rl.outputs import read_json, write_json

REPORT_DIR = Path(__file__).resolve().parent
ANSWERS_FILE = REPORT_DIR / "answers.json"
META_FIELDS = {
    "name": "Name",
    "github_username": "GitHub username",
    "repo_name": "Repository name",
    "repo_url": "Link to the GitHub repository",
}


def load() -> dict[str, Any]:
    data = read_json(ANSWERS_FILE, {}) or {}
    data.setdefault("meta", {})
    data.setdefault("answers", {})
    return data


def save_answer(question_id: str, text: str) -> None:
    data = load()
    data["answers"][question_id] = text
    write_json(ANSWERS_FILE, data)


def save_meta(field: str, value: str) -> None:
    data = load()
    data["meta"][field] = value
    write_json(ANSWERS_FILE, data)


def text(question: Question, data: dict[str, Any]) -> str:
    """The answer to `question`: its source file (without HTML comments) or the answer saved in `data`."""
    if question.source:
        path = REPORT_DIR / question.source
        return re.sub(r"<!--.*?-->\s*", "", path.read_text(), flags=re.S) if path.exists() else ""
    return data["answers"].get(question.id, "")
