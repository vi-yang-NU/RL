"""Unit tests for `python run.py save` (rl/save.py), on throwaway repositories with a local bare "GitHub"."""
from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from rl.save import auto_message, github_page, save


def sh(*args: str, cwd: Path) -> str:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@unittest.skipUnless(shutil.which("git"), "git is not installed")
class SaveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.remote = base / "remote.git"
        sh("git", "init", "-q", "--bare", str(self.remote), cwd=base)
        self.repo = base / "repo"
        self.repo.mkdir()
        self._init(self.repo)
        sh("git", "remote", "add", "origin", str(self.remote), cwd=self.repo)
        self.out = io.StringIO()

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _init(path: Path) -> None:
        sh("git", "init", "-q", cwd=path)
        sh("git", "checkout", "-q", "-b", "main", cwd=path)
        sh("git", "config", "user.name", "Test", cwd=path)
        sh("git", "config", "user.email", "test@example.com", cwd=path)
        sh("git", "config", "commit.gpgsign", "false", cwd=path)

    def _save(self, **kw) -> int:
        return save(self.repo, report=False, out=self.out, **kw)

    def _write(self, rel: str, text: str = "x\n") -> None:
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_first_save_commits_everything_and_pushes_it(self):
        self._write("rl/a.py")
        self._write("rl/b.py")
        self._write("ui/app.py")
        self._write("README.md")
        self.assertEqual(self._save(), 0, self.out.getvalue())
        head = sh("git", "rev-parse", "HEAD", cwd=self.repo)
        self.assertEqual(sh("git", "rev-parse", "main", cwd=self.remote), head)
        self.assertEqual(sh("git", "status", "--porcelain", cwd=self.repo), "")
        message = sh("git", "log", "-1", "--format=%s", cwd=self.repo)
        self.assertRegex(message, r"^Save \d{4}-\d{2}-\d{2} \d{2}:\d{2}: 4 files \(rl, ")
        self.assertIn("pushed to origin/main", self.out.getvalue())
        self.assertEqual(sh("git", "rev-parse", "--abbrev-ref", "main@{upstream}", cwd=self.repo), "origin/main")

    def test_nothing_new_is_not_an_error(self):
        self._write("a.txt")
        self.assertEqual(self._save(), 0)
        self.assertEqual(self._save(), 0)
        self.assertIn("nothing new to commit", self.out.getvalue())
        self.assertIn("already had everything", self.out.getvalue())
        self.assertEqual(sh("git", "rev-list", "--count", "HEAD", cwd=self.repo), "1")

    def test_custom_message_and_no_push(self):
        self._write("a.txt")
        self.assertEqual(self._save(message="hello there", push=False), 0)
        self.assertEqual(sh("git", "log", "-1", "--format=%s", cwd=self.repo), "hello there")
        self.assertEqual(sh("git", "branch", "--list", cwd=self.remote), "")

    def test_deleted_and_modified_files_are_saved_too(self):
        self._write("keep.txt", "1\n")
        self._write("gone.txt")
        self.assertEqual(self._save(), 0)
        self._write("keep.txt", "2\n")
        (self.repo / "gone.txt").unlink()
        self.assertEqual(self._save(), 0)
        files = sh("git", "ls-tree", "--name-only", "main", cwd=self.remote).splitlines()
        self.assertEqual(files, ["keep.txt"])
        self.assertEqual(sh("git", "show", "main:keep.txt", cwd=self.remote), "2")

    def test_refuses_files_github_would_reject(self):
        self._write("small.txt")
        self._write("results/huge.bin", "0" * 5000)
        self.assertEqual(self._save(limit_bytes=1000), 1)
        self.assertIn("huge.bin", self.out.getvalue())
        self.assertNotEqual(subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True).returncode, 0,
                            "nothing may be committed")
        status = sh("git", "status", "--porcelain", cwd=self.repo).splitlines()
        self.assertTrue(all(line.startswith("??") for line in status), status)

    def test_pulls_what_github_has_before_pushing(self):
        self._write("a.txt")
        self.assertEqual(self._save(), 0)
        other = Path(self.tmp.name) / "other"
        sh("git", "clone", "-q", str(self.remote), str(other), cwd=self.tmp.name)
        self._init_existing(other)
        (other / "from_github.txt").write_text("edited on github.com\n")
        sh("git", "add", "-A", cwd=other)
        sh("git", "commit", "-q", "-m", "edit on GitHub", cwd=other)
        sh("git", "push", "-q", "origin", "main", cwd=other)
        self._write("b.txt")
        self.assertEqual(self._save(), 0, self.out.getvalue())
        self.assertIn("pulling them first", self.out.getvalue())
        self.assertEqual(sh("git", "rev-parse", "main", cwd=self.remote), sh("git", "rev-parse", "HEAD", cwd=self.repo))
        self.assertEqual(set(sh("git", "ls-tree", "--name-only", "main", cwd=self.remote).splitlines()),
                         {"a.txt", "b.txt", "from_github.txt"})

    def test_push_failure_keeps_the_commit_here(self):
        sh("git", "remote", "set-url", "origin", str(Path(self.tmp.name) / "missing.git"), cwd=self.repo)
        self._write("a.txt")
        self.assertEqual(self._save(), 1)
        self.assertEqual(sh("git", "rev-list", "--count", "HEAD", cwd=self.repo), "1")
        self.assertIn("committed", self.out.getvalue())
        self.assertIn("run `python run.py save` again", self.out.getvalue())

    def test_no_remote_is_explained(self):
        sh("git", "remote", "remove", "origin", cwd=self.repo)
        self._write("a.txt")
        self.assertEqual(self._save(), 1)
        self.assertIn("git remote add origin", self.out.getvalue())

    @staticmethod
    def _init_existing(path: Path) -> None:
        sh("git", "config", "user.name", "Other", cwd=path)
        sh("git", "config", "user.email", "other@example.com", cwd=path)
        sh("git", "config", "commit.gpgsign", "false", cwd=path)


class HelperTests(unittest.TestCase):
    def test_github_page_from_ssh_and_https_urls(self):
        for url in ("git@github.com:vi-yang-NU/RL.git", "https://github.com/vi-yang-NU/RL.git",
                    "https://github.com/vi-yang-NU/RL", "ssh://git@github.com/vi-yang-NU/RL.git"):
            self.assertEqual(github_page(url), "https://github.com/vi-yang-NU/RL", url)
        self.assertIsNone(github_page("/tmp/remote.git"))
        self.assertIsNone(github_page("https://gitlab.com/a/b.git"))

    def test_auto_message_names_the_areas_touched(self):
        msg = auto_message(["rl/dp/a.py", "rl/pg/b.py", "ui/app.py", "README.md"])
        self.assertRegex(msg, r"^Save \d{4}-\d{2}-\d{2} \d{2}:\d{2}: 4 files \(rl, ui, README\.md\)$")
        self.assertTrue(auto_message(["x"]).endswith("1 file (x)"))
        many = auto_message([f"d{i}/f" for i in range(7)])
        self.assertIn("+2 more", many)


if __name__ == "__main__":
    unittest.main()
