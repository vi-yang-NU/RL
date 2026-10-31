"""Play the actual game live in a window, with the learned policy at the controls, and take over whenever you like.

    python run.py play --env pong                      # the latest Part 1 Pong policy (results/pg/part1_pong) plays
    python run.py play --env cartpole --run part1_cartpole
    python run.py play --env pong --untrained --speed 2

A pygame window shows the real frames of CartPole-v1 or ALE/Pong-v5 as the game runs, at the game's own speed by
default. The policy (a run's policy.pt, or an untrained network) chooses the actions and the panel on the right
shows its probabilities, the action, the score and the controls. Press the arrow keys to take the controls
yourself (Pong: Up/Down move the paddle, sent as RIGHT (2) / LEFT (3); CartPole: Left/Right push the cart) and
A to hand them back; Space pauses, R starts a new game, G switches the policy between sampling and greedy, + and
- change the speed, Q quits.

Pong's opponent is the cartridge's built-in computer player: the environment has one joystick (ale-py exposes a
single `act`), so taking over means playing the agent's side against the same opponent the network faces, not
playing against the network. `--headless` runs with SDL's dummy video driver (no window) for the tests and for
`--screenshot`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rl.outputs import ENV_LABELS, RESULTS, ROOT
from rl.pg.envs import ENV_IDS, PONG_ACTIONS, SECONDS_PER_STEP, make_adapter, make_env

# per environment: what the arrow keys do (key name -> action sent to the environment), what no key means while you
# play, real-time steps per second, and the default speed factor (CartPole runs at 50 steps/s, too fast for a hand)
KEYS = {
    "pong": {"up": 2, "down": 3},
    "cartpole": {"left": 0, "right": 1},
}
IDLE_ACTION = {"pong": 0, "cartpole": None}  # None: CartPole has no "do nothing", the policy's choice is used
REAL_TIME_FPS = {"pong": 15, "cartpole": 50}
DEFAULT_SPEED = {"pong": 1.0, "cartpole": 0.4}
ACTION_NAMES = {
    "pong": {0: "NOOP", 1: "FIRE", 2: "RIGHT (up)", 3: "LEFT (down)", 4: "RIGHTFIRE", 5: "LEFTFIRE"},
    "cartpole": {0: "push left", 1: "push right"},
}
POLICY_SHORT = {"pong": ("RIGHT (up)", "LEFT (down)"), "cartpole": ("push left", "push right")}
HELP = {
    "pong": "Up/Down: take the paddle   A: the agent plays   Space: pause   R: new game   G: greedy/sampling   +/-: speed   Q: quit",
    "cartpole": "Left/Right: push the cart   A: the agent plays   Space: pause   R: new episode   G: greedy/sampling   +/-: speed   Q: quit",
}
SCALE = {"pong": 3, "cartpole": 1.25}
PANEL_WIDTH = 330


def clock_text(env_name: str, steps: float) -> str:
    """Game time of `steps` steps as m:ss (or seconds with a decimal below 10 s)."""
    seconds = float(steps) * SECONDS_PER_STEP[env_name]
    if seconds < 10:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes}:{rest:02d} min" if minutes else f"{rest} s"


@dataclass
class Game:
    """The environment, the adapter and the policy; `step` plays one step with the policy or with a given action."""

    cfg: Any
    env: Any
    policy: Any
    policy_label: str
    greedy: bool = False
    t: int = 0
    total: float = 0.0
    score: list = field(default_factory=lambda: [0, 0])  # Pong: agent's points, opponent's points
    done: bool = False
    frame: np.ndarray | None = None
    probs: np.ndarray | None = None
    last_action: int | None = None
    last_controller: str = "agent"
    finished: list = field(default_factory=list)  # (controller, total reward, steps) of every finished game

    def __post_init__(self) -> None:
        from rl.pg.options import options_for

        self.adapter = make_adapter(self.cfg, self.env, options_for(self.cfg))
        self.obs = None

    @property
    def env_name(self) -> str:
        return self.cfg.env

    def reset(self, seed: int | None = None) -> None:
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        self.obs = self.adapter.reset(self.env, seed)
        self.frame = self.env.render()
        self.t, self.total, self.score, self.done = 0, 0.0, [0, 0], False
        self.last_action, self.last_controller = None, "agent"
        self.probs = self.policy_probs()
        self.games_controller = "agent"  # who played any part of this game ("you" once you touched the controls)

    def policy_probs(self) -> np.ndarray:
        import torch

        with torch.no_grad():
            logits = self.policy(torch.as_tensor(self.obs).unsqueeze(0))
            return torch.softmax(logits, dim=-1)[0].cpu().numpy()

    def policy_action(self) -> int:
        """The environment action the policy takes in the current state (sampled, or the most probable)."""
        import torch

        p = self.probs if self.probs is not None else self.policy_probs()
        index = int(np.argmax(p)) if self.greedy else int(torch.multinomial(torch.as_tensor(p, dtype=torch.float32), 1).item())
        return int(self.adapter.action(index))

    def step(self, env_action: int | None, controller: str = "agent") -> float:
        """One step: `env_action` None lets the policy choose. Returns the reward."""
        if self.done:
            return 0.0
        if env_action is None:
            env_action = self.policy_action()
            controller = "agent"
        else:
            self.games_controller = "you"
        obs, reward, terminated, truncated, _ = self.env.step(int(env_action))
        self.obs = self.adapter.encode(obs)  # keeps the adapter's frame bookkeeping (Pong's frame difference)
        self.frame = self.env.render()
        self.t += 1
        self.total += float(reward)
        if reward > 0:
            self.score[0] += 1
        elif reward < 0:
            self.score[1] += 1
        self.last_action, self.last_controller = int(env_action), controller
        self.done = bool(terminated or truncated)
        self.probs = self.policy_probs() if not self.done else self.probs
        if self.done:
            self.finished.append((self.games_controller, self.total, self.t))
        return float(reward)


def load_policy(env_name: str, run: str | None, untrained: bool, seed: int) -> tuple[Any, Any, str]:
    """(cfg, policy, label): the run's checkpoint, or the latest Part 1 / Part 2 run for the environment when no
    run is named, or an untrained network."""
    from rl.pg.watch import load_run
    from rl.pg.watch import untrained as fresh

    if not untrained:
        candidates = [RESULTS / "pg" / run] if run else [RESULTS / "pg" / f"part{p}_{env_name}" for p in (1, 2)]
        if run and not candidates[0].exists() and Path(run).exists():
            candidates = [Path(run)]
        for run_dir in candidates:
            if (run_dir / "policy.pt").exists() and (run_dir / "config.json").exists():
                cfg, policy = load_run(run_dir)
                if cfg.env != env_name:
                    raise SystemExit(f"{run_dir.name} was trained on {cfg.env}, not {env_name}")
                rel = run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir
                return cfg, policy, f"{rel}/policy.pt"
        if run:
            raise SystemExit(f"no policy.pt in results/pg/{run} (train it first: python run.py pg --part 1 --env {env_name})")
        print(f"no trained policy for {env_name} in results/pg yet: an untrained network plays")
    cfg, policy = fresh(env_name, seed)
    return cfg, policy, "untrained network (random initial weights)"


class Window:
    """The pygame window: the frame on the left, the panel on the right, the keyboard."""

    def __init__(self, game: Game, headless: bool = False):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        self.pg = pygame
        pygame.init()
        pygame.display.init()
        self.game = game
        h, w = game.frame.shape[:2]
        scale = SCALE[game.env_name]
        self.frame_size = (int(round(w * scale)), int(round(h * scale)))
        height = max(self.frame_size[1], 520)
        self.screen = pygame.display.set_mode((self.frame_size[0] + PANEL_WIDTH, height))
        pygame.display.set_caption(f"{ENV_LABELS[game.env_name]} · {game.policy_label}")
        self.font = pygame.font.SysFont("menlo,monaco,consolas,dejavusansmono,couriernew", 15)
        self.font_big = pygame.font.SysFont("helveticaneue,helvetica,arial,dejavusans", 22, bold=True)
        self.font_small = pygame.font.SysFont("helveticaneue,helvetica,arial,dejavusans", 13)
        self.colors = {"bg": (24, 26, 31), "panel": (36, 39, 46), "ink": (236, 236, 236), "muted": (150, 150, 150),
                       "blue": (57, 135, 229), "orange": (235, 104, 52), "green": (30, 180, 120), "red": (220, 80, 80),
                       "track": (60, 64, 72)}

    def draw(self, controller: str, paused: bool, fps: float, speed: float, message: str | None = None) -> None:
        pg, g, c = self.pg, self.game, self.colors
        self.screen.fill(c["bg"])
        surface = pg.surfarray.make_surface(np.transpose(g.frame, (1, 0, 2)))
        self.screen.blit(pg.transform.scale(surface, self.frame_size), (0, 0))
        x0 = self.frame_size[0]
        pg.draw.rect(self.screen, c["panel"], (x0, 0, PANEL_WIDTH, self.screen.get_height()))
        x, y = x0 + 16, 14

        def line(text, font=None, color=None, dy=None):
            nonlocal y
            img = (font or self.font).render(text, True, color or c["ink"])
            self.screen.blit(img, (x, y))
            y += dy or (img.get_height() + 6)

        line(f"{ENV_LABELS[g.env_name]} · {ENV_IDS[g.env_name]}", self.font_big)
        line(f"policy: {g.policy_label}", self.font_small, c["muted"])
        line(f"{'greedy' if g.greedy else 'sampling'} actions · {fps:.0f} steps/s ({speed:g}× real time)", self.font_small, c["muted"], dy=24)
        you = controller == "you"
        line("CONTROLS: YOU (keyboard)" if you else "CONTROLS: the agent", self.font_big, c["orange"] if you else c["blue"])
        if g.env_name == "pong":
            line(f"score  agent {g.score[0]} : {g.score[1]} opponent")
            line(f"step {g.t} · game time {clock_text(g.env_name, g.t)}")
        else:
            line(f"step {g.t} · game time {clock_text(g.env_name, g.t)} · reward {g.total:g}")
            line("the episode ends at 500 steps (10 s) or when the pole falls", self.font_small, c["muted"])
        y += 8
        line("the policy's probabilities", self.font_small, c["muted"])
        names = POLICY_SHORT[g.env_name]
        for i, p in enumerate(g.probs if g.probs is not None else []):
            label = self.font.render(f"{names[i]:<12} {100 * p:5.1f}%", True, c["ink"])
            self.screen.blit(label, (x, y))
            bar_x = x + 200
            pg.draw.rect(self.screen, c["track"], (bar_x, y + 3, 100, 10), border_radius=3)
            pg.draw.rect(self.screen, c["blue"] if i == 0 else c["orange"], (bar_x, y + 3, int(100 * p), 10), border_radius=3)
            y += 22
        action = ACTION_NAMES[g.env_name].get(g.last_action, "–") if g.last_action is not None else "–"
        line(f"last action: {action}  ({'you' if g.last_controller == 'you' else 'agent'})", dy=30)
        if g.finished:
            if g.env_name == "pong":
                agent = [f for f in g.finished if f[0] == "agent"]
                yours = [f for f in g.finished if f[0] == "you"]
                won = lambda fs: sum(1 for f in fs if f[1] > 0)  # noqa: E731
                line(f"games: agent {won(agent)} won / {len(agent) - won(agent)} lost", self.font_small, c["muted"])
                line(f"       you   {won(yours)} won / {len(yours) - won(yours)} lost", self.font_small, c["muted"])
                mean_len = sum(f[2] for f in g.finished) / len(g.finished)
                line(f"       a game lasts {mean_len:.0f} steps ({clock_text('pong', mean_len)}) on average", self.font_small, c["muted"])
            else:
                best = max(f[1] for f in g.finished)
                line(f"episodes: {len(g.finished)} · best {best:g} · last {g.finished[-1][1]:g}", self.font_small, c["muted"])
        y = self.screen.get_height() - 92
        if message:
            line(message, self.font, c["green"] if "won" in message else c["ink"])
        if paused:
            line("PAUSED (Space)", self.font, c["orange"])
        words, row = HELP[g.env_name].split("   "), ""
        for word in words + [""]:
            if word and len(row) + len(word) < 44:
                row = f"{row}   {word}" if row else word
            else:
                if row:
                    line(row, self.font_small, c["muted"], dy=16)
                row = word
        pg.display.flip()

    def poll(self) -> tuple[set, set, bool]:
        """(keys pressed this frame, keys held now, quit requested) from pygame's event queue."""
        pg = self.pg
        pressed, quit_ = set(), False
        names = {pg.K_UP: "up", pg.K_DOWN: "down", pg.K_LEFT: "left", pg.K_RIGHT: "right", pg.K_a: "a", pg.K_SPACE: "space",
                 pg.K_r: "r", pg.K_g: "g", pg.K_q: "q", pg.K_ESCAPE: "q", pg.K_PLUS: "+", pg.K_EQUALS: "+", pg.K_KP_PLUS: "+",
                 pg.K_MINUS: "-", pg.K_KP_MINUS: "-"}
        for event in pg.event.get():
            if event.type == pg.QUIT:
                quit_ = True
            elif event.type == pg.KEYDOWN and event.key in names:
                pressed.add(names[event.key])
        held_keys = pg.key.get_pressed()
        held = {name for key, name in names.items() if name in ("up", "down", "left", "right") and held_keys[key]}
        return pressed, held, quit_

    def screenshot(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pg.image.save(self.screen, str(path))
        return path

    def close(self) -> None:
        self.pg.quit()


def run(game: Game, window: Window, speed: float, *, seed: int = 0, max_steps: int | None = None,
        script: dict[int, set] | None = None, screenshot: Path | None = None, realtime: bool = True) -> dict[str, Any]:
    """The game loop. `script` (for the tests) maps a step number to key names acted on at that step instead of
    the keyboard; `max_steps` stops after that many steps. Returns what happened (controllers used, actions, ...)."""
    env_name = game.env_name
    controller, paused, greedy = "agent", False, game.greedy
    base_fps = REAL_TIME_FPS[env_name]
    clock = window.pg.time.Clock()
    game.reset(seed)
    log: dict[str, Any] = {"controllers": [], "actions": [], "steps": 0, "games": 0, "speed": speed}
    game_over_at: float | None = None
    message = None
    steps = 0
    while True:
        if script is not None:
            keys = script.get(steps, set())
            pressed, held, quit_ = {k for k in keys if k not in ("up", "down", "left", "right")}, {k for k in keys if k in KEYS[env_name]}, "quit" in keys
        else:
            pressed, held, quit_ = window.poll()
        if quit_ or "q" in pressed:
            break
        if "space" in pressed:
            paused = not paused
        if "a" in pressed:
            controller = "agent"
        if "g" in pressed:
            game.greedy = greedy = not greedy
        if "+" in pressed:
            speed = min(8.0, speed * 2)
        if "-" in pressed:
            speed = max(0.125, speed / 2)
        if "r" in pressed or (game.done and game_over_at is not None and time.time() - game_over_at > 2.5 and not paused and script is None):
            seed += 1
            game.reset(seed)
            game_over_at, message = None, None
        if held & set(KEYS[env_name]):
            controller = "you"
        if not paused and not game.done:
            if controller == "you":
                key = next((k for k in ("up", "down", "left", "right") if k in held), None)
                action = KEYS[env_name][key] if key else IDLE_ACTION[env_name]
                game.step(action, "you") if action is not None else game.step(None)
            else:
                game.step(None)
            log["controllers"].append(controller)
            log["actions"].append(game.last_action)
            steps += 1
            if game.done:
                log["games"] += 1
                game_over_at = time.time()
                if env_name == "pong":
                    outcome = "won" if game.total > 0 else "lost"
                    message = f"Game over: {outcome} {game.score[0]} : {game.score[1]} after {game.t} steps ({clock_text(env_name, game.t)})"
                else:
                    message = f"Episode over after {game.t} steps ({clock_text(env_name, game.t)}), reward {game.total:g}"
        fps = base_fps * speed
        window.draw(controller, paused, fps, speed, message)
        if max_steps is not None and steps >= max_steps:
            break
        if realtime:
            clock.tick(fps)
    log["steps"] = steps
    if screenshot is not None:
        log["screenshot"] = str(window.screenshot(screenshot))
    return log


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", choices=list(ENV_IDS), required=True)
    parser.add_argument("--run", help="results/pg folder (or its name) whose policy.pt plays; default: part1_<env>, then part2_<env>")
    parser.add_argument("--untrained", action="store_true", help="let an untrained network play")
    parser.add_argument("--greedy", action="store_true", help="the policy takes its most probable action (G toggles this)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, help="speed as a multiple of real time (default: Pong 1, CartPole 0.4)")
    parser.add_argument("--steps", type=int, help="stop after this many steps (otherwise play until Q)")
    parser.add_argument("--headless", action="store_true", help="no window (SDL's dummy driver); for tests and --screenshot")
    parser.add_argument("--screenshot", help="save the window to this PNG when the loop ends")


def main(args: argparse.Namespace) -> int:
    if not args.headless and os.environ.get("SSH_CONNECTION") and not os.environ.get("DISPLAY"):
        print("No display here (over ssh): run this on your own computer, or use the dashboard's Watch the game panel.")
        return 1
    cfg, policy, label = load_policy(args.env, args.run, args.untrained, args.seed)
    cfg.extra = dict(cfg.extra or {})
    env = make_env(args.env, render_mode="rgb_array")
    try:
        game = Game(cfg, env, policy, label, greedy=args.greedy)
        game.reset(args.seed)
        window = Window(game, headless=args.headless)
        try:
            speed = args.speed if args.speed else DEFAULT_SPEED[args.env]
            print(f"{ENV_IDS[args.env]} · policy: {label} · {REAL_TIME_FPS[args.env] * speed:g} steps/s. {HELP[args.env]}")
            log = run(game, window, speed, seed=args.seed, max_steps=args.steps,
                      screenshot=Path(args.screenshot) if args.screenshot else None, realtime=not args.headless)
        finally:
            window.close()
    finally:
        env.close()
    you = sum(1 for c in log["controllers"] if c == "you")
    print(f"{log['steps']} steps played ({you} by you), {log['games']} game(s) finished"
          + (f"; screenshot {log['screenshot']}" if "screenshot" in log else ""))
    return 0
