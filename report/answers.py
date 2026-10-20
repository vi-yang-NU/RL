"""Discussion answers and submission details, saved by the dashboard to report/answers.json."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rl.outputs import read_json, write_json

ANSWERS_FILE = Path(__file__).resolve().parent / "answers.json"
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
