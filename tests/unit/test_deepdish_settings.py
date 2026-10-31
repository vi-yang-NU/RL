"""Unit tests for the deepdish connection settings (ui/deepdish.py): the .env file and the DEEPDISH_* variables
that overlay config/deepdish.json, where each value comes from, saving choices back to .env, and the ssh
options a server resolves to."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ui import deepdish as dd

BASE = {
    "servers": [1, 2, 3],
    "alias_template": "deepdish{n}",
    "host_template": "mlds-deepdish{n}.example.edu",
    "user": "",
    "identity_file": "",
    "default_server": 1,
    "remote_dir": "~/RL",
    "control_path": "~/.ssh/cm-%C",
    "control_persist": "4h",
    "connect_timeout": 10,
    "upload_excludes": [".git/", ".env"],
}


class SettingsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.config, self.local, self.env = root / "deepdish.json", root / "deepdish.local.json", root / ".env"
        self.config.write_text(json.dumps(BASE))
        for name, path in (("CONFIG_FILE", self.config), ("LOCAL_CONFIG_FILE", self.local), ("ENV_FILE", self.env)):
            patcher = mock.patch.object(dd, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_env_file_parsing(self):
        self.env.write_text(
            "# comment\n\nDEEPDISH_USER=alice\nexport DEEPDISH_REMOTE_DIR='~/work dir'\n"
            'DEEPDISH_DEFAULT_SERVER="2"\nDEEPDISH_CONTROL_PERSIST=2h # trailing comment\nnot a pair\n=novalue\n')
        self.assertEqual(dd.read_env_file(), {"DEEPDISH_USER": "alice", "DEEPDISH_REMOTE_DIR": "~/work dir",
                                              "DEEPDISH_DEFAULT_SERVER": "2", "DEEPDISH_CONTROL_PERSIST": "2h"})

    def test_without_env_file_the_base_config_applies(self):
        self.assertEqual(dd.read_env_file(), {})
        self.assertEqual(dd.load_config(environ={}), BASE)
        self.assertEqual(set(dd.setting_sources(environ={}).values()), {"config/deepdish.json"})

    def test_precedence_json_local_env_file_environment(self):
        self.local.write_text(json.dumps({"user": "local", "default_server": 2, "remote_dir": "~/local"}))
        self.env.write_text("DEEPDISH_USER=dotenv\nDEEPDISH_DEFAULT_SERVER=3\n")
        environ = {"DEEPDISH_USER": "environment", "HOME": "/x"}
        cfg = dd.load_config(environ)
        self.assertEqual((cfg["user"], cfg["default_server"], cfg["remote_dir"]), ("environment", 3, "~/local"))
        sources = dd.setting_sources(environ)
        self.assertEqual(sources["user"], "environment")
        self.assertEqual(sources["default_server"], ".env")
        self.assertEqual(sources["remote_dir"], "config/deepdish.local.json")
        self.assertEqual(sources["control_path"], "config/deepdish.json")

    def test_values_take_the_types_of_the_base_config(self):
        self.env.write_text("DEEPDISH_SERVERS=2, 4,5\nDEEPDISH_CONNECT_TIMEOUT=3\nDEEPDISH_UPLOAD_EXCLUDES=a/,b\nDEEPDISH_USER=\n")
        cfg = dd.load_config(environ={})
        self.assertEqual(cfg["servers"], [2, 4, 5])
        self.assertEqual(cfg["connect_timeout"], 3)
        self.assertEqual(cfg["upload_excludes"], ["a/", "b"])
        self.assertEqual(cfg["user"], "")  # an empty value leaves the setting alone

    def test_a_bad_value_names_the_variable(self):
        with self.assertRaisesRegex(ValueError, "DEEPDISH_DEFAULT_SERVER='two'"):
            dd.load_config(environ={"DEEPDISH_DEFAULT_SERVER": "two"})

    def test_unknown_variables_are_ignored(self):
        self.assertNotIn("something", dd.load_config(environ={"DEEPDISH_SOMETHING": "x"}))

    def test_save_creates_env_with_a_header(self):
        dd.save_settings(user="alice", default_server=2)
        text = self.env.read_text()
        self.assertTrue(text.startswith("#"))
        self.assertIn("DEEPDISH_USER=alice\n", text)
        self.assertIn("DEEPDISH_DEFAULT_SERVER=2\n", text)
        self.assertEqual(dd.load_config(environ={})["user"], "alice")
        self.assertEqual(dd.setting_sources(environ={})["default_server"], ".env")

    def test_save_replaces_in_place_and_keeps_other_lines(self):
        self.env.write_text("# mine\nOTHER=1\nexport DEEPDISH_USER=old # was\nDEEPDISH_REMOTE_DIR=~/RL\n")
        dd.save_settings(user="new", servers=[1, 2], remote_dir="~/my dir")
        self.assertEqual(self.env.read_text(),
                         '# mine\nOTHER=1\nDEEPDISH_USER=new\nDEEPDISH_REMOTE_DIR="~/my dir"\nDEEPDISH_SERVERS=1,2\n')
        cfg = dd.load_config(environ={})
        self.assertEqual((cfg["user"], cfg["servers"], cfg["remote_dir"]), ("new", [1, 2], "~/my dir"))

    def test_dashboard_saves_win_over_an_older_local_json(self):
        """A NetID typed in the sidebar (saved to .env) takes effect even next to an older deepdish.local.json."""
        self.local.write_text(json.dumps({"user": "old"}))
        dd.save_settings(user="typed")
        self.assertEqual(dd.load_config(environ={})["user"], "typed")


@unittest.skipUnless(shutil.which("ssh"), "ssh is not installed")
class ServerTests(unittest.TestCase):
    """resolve_server asks ssh -G (no connection is made); the alias is one no ~/.ssh/config defines."""

    def test_identity_file_and_user_reach_ssh(self):
        cfg = {**BASE, "user": "alice", "identity_file": "~/.ssh/mykey", "alias_template": "no-such-alias-{n}"}
        server = dd.resolve_server(2, cfg)
        self.assertEqual(server.target, "alice@mlds-deepdish2.example.edu")
        self.assertEqual((server.user, server.hostname), ("alice", "mlds-deepdish2.example.edu"))
        opts = server.ssh_options()
        self.assertIn("IdentityFile=~/.ssh/mykey", opts)
        self.assertIn("ControlMaster=no", opts)
        if not server.shared_control_path:  # unless ~/.ssh/config sets a ControlPath for every host
            self.assertIn(f"ControlPath={BASE['control_path']}", opts)
        master = server.ssh_options(master=True)
        self.assertIn("ControlMaster=auto", master)
        self.assertIn("ControlPersist=4h", master)

    def test_without_identity_file_ssh_picks_the_key(self):
        server = dd.resolve_server(1, {**BASE, "alias_template": "no-such-alias-{n}"})
        self.assertFalse(any(o.startswith("IdentityFile=") for o in server.ssh_options()))
