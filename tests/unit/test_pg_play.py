"""Unit tests for the live game window (rl/pg/play.py), run with SDL's dummy video driver and scripted keys."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rl.pg.envs import PONG_ACTIONS, make_env
from rl.pg.play import KEYS, Game, Window, load_policy, run

torch.set_num_threads(2)


def _game(env_name: str) -> tuple[Game, Window]:
    cfg, policy, label = load_policy(env_name, None, untrained=True, seed=1)
    env = make_env(env_name, render_mode="rgb_array")
    game = Game(cfg, env, policy, label)
    game.reset(1)
    return game, Window(game, headless=True)


class CartPolePlayTests(unittest.TestCase):
    def test_agent_plays_then_you_take_over_and_hand_back(self):
        game, window = _game("cartpole")
        try:
            script = {3: {"right"}, 4: {"right"}, 5: {"left"}, 7: {"a"}, 9: {"space"}, 10: {"space"}}
            with tempfile.TemporaryDirectory() as tmp:
                log = run(game, window, speed=1.0, seed=1, max_steps=12, script=script, screenshot=Path(tmp) / "shot.png", realtime=False)
                self.assertTrue(Path(log["screenshot"]).exists())
            self.assertEqual(log["steps"], 12)
            self.assertEqual(log["controllers"][:3], ["agent"] * 3)
            self.assertEqual(log["controllers"][3:7], ["you"] * 4)
            self.assertEqual(log["controllers"][7:], ["agent"] * 5)
            self.assertEqual(log["actions"][3:5], [KEYS["cartpole"]["right"]] * 2)
            self.assertEqual(log["actions"][5], KEYS["cartpole"]["left"])
            self.assertTrue(set(log["actions"]) <= {0, 1})
        finally:
            window.close()
            game.env.close()

    def test_reset_and_quit_keys(self):
        game, window = _game("cartpole")
        try:
            log = run(game, window, speed=1.0, seed=1, max_steps=50, script={4: {"r"}, 6: {"q"}}, realtime=False)
            self.assertEqual(log["steps"], 6)  # quit at step 6
            self.assertEqual(game.t, 2)  # the new episode started at step 4
        finally:
            window.close()
            game.env.close()


class PongPlayTests(unittest.TestCase):
    def test_pong_keys_send_right_and_left_and_idle_sends_noop(self):
        game, window = _game("pong")
        try:
            script = {0: {"up"}, 1: {"up"}, 2: {"down"}, 3: set(), 4: {"a"}}
            log = run(game, window, speed=1.0, seed=1, max_steps=6, script=script, realtime=False)
            self.assertEqual(log["actions"][:3], [2, 2, 3])
            self.assertEqual(log["actions"][3], 0)  # you hold no key: NOOP
            self.assertEqual(log["controllers"][3], "you")
            self.assertEqual(log["controllers"][4], "agent")
            self.assertIn(log["actions"][4], PONG_ACTIONS)  # the agent only ever sends RIGHT or LEFT
            self.assertEqual(game.frame.shape, (210, 160, 3))
            np.testing.assert_allclose(game.probs.sum(), 1.0, atol=1e-5)
        finally:
            window.close()
            game.env.close()


if __name__ == "__main__":
    unittest.main()


class ClockTests(unittest.TestCase):
    def test_clock_text(self):
        from rl.pg.play import clock_text

        self.assertEqual(clock_text("pong", 764), "51 s")
        self.assertEqual(clock_text("pong", 3000), "3:20 min")
        self.assertEqual(clock_text("cartpole", 500), "10 s")
        self.assertEqual(clock_text("cartpole", 40), "0.8 s")
