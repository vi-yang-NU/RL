"""Unit tests for `python run.py dp` (rl/dp/run.py): tagged runs keep the handout's results untouched."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rl import outputs
from rl.dp import run as dp_run

OUTPUTS = ("params.json", "enumeration.json", "value_iteration.json", "policy_iteration.json",
           "enumeration_value.png", "value_iteration_value.png", "policy_iteration_policy.png")


class DpRunTagTests(unittest.TestCase):
    def _run(self, argv: list[str], results: Path) -> str:
        parser = argparse.ArgumentParser()
        dp_run.add_arguments(parser)
        out = io.StringIO()
        with mock.patch.object(outputs, "RESULTS", results), contextlib.redirect_stdout(out):
            self.assertEqual(dp_run.main(parser.parse_args(argv)), 0)
        return out.getvalue()

    def test_tagged_run_writes_everything_next_to_its_data_and_leaves_the_handout_folder_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            text = self._run(["--set", "max_customers=20", "--set", "capacity=5", "--set", "horizon=5", "--tag", "tiny station!"], results)
            out = results / "dp_tinystation"  # the tag is reduced to [A-Za-z0-9_-]
            for name in OUTPUTS:
                self.assertTrue((out / name).is_file(), name)
            self.assertFalse((results / "dp").exists(), "the handout's folder must not be touched")
            params = json.loads((out / "params.json").read_text())
            self.assertEqual((params["max_customers"], params["capacity"], params["horizon"]), (20, 5, 5))
            self.assertIn("never dispatches", text)  # the tiny-station edge case: the cap turns customers away for free

    def test_untagged_run_is_the_handout_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            self._run(["--method", "policy_iteration", "--set", "max_customers=20", "--set", "horizon=5"], results)
            self.assertTrue((results / "dp" / "policy_iteration.json").is_file())
            self.assertTrue((results / "dp" / "policy_iteration_policy.png").is_file())


if __name__ == "__main__":
    unittest.main()
