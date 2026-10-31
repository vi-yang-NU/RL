"""The "Watch the game" panel of the Part 1 and Part 2 tabs: one episode of the real gym environment, frame by frame.

The widgets on the left choose the game (CartPole-v1 or Pong-v5), the policy (the episode `python run.py pg`
recorded with the trained policy, a checkpoint to play a new episode with, or an untrained policy) and the seed;
`rl.pg.watch` records the episode (cached per choice) and `render_html` embeds it in rl/pg/watch.html, shown in
an iframe on the right. `python run.py watch` writes the same page from the command line.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from rl.outputs import ENV_LABELS, ROOT
from rl.pg.watch import load_recording, policy_sources, record, render_html, viewer_data

HEIGHT = {"cartpole": 1120, "pong": 1180}  # initial iframe height; the page then sizes the iframe to its content
ENVS = ("cartpole", "pong")
PART = {"pg1": 1, "pg2": 2}


def _theme() -> str | None:
    try:
        kind = st.context.theme.type
    except Exception:  # noqa: BLE001 - older Streamlit without st.context.theme
        return None
    return kind if kind in ("light", "dark") else None


@st.cache_data(show_spinner="Playing one episode…", max_entries=24)
def _episode(env: str, kind: str, path: str, stamp: float, seed: int, max_steps: int, greedy: bool) -> dict:
    """The viewer's data for one choice of game, policy and seed. `stamp` (the file's modification time) is part
    of the key so a rerun's new episode or checkpoint replaces the cached one."""
    if kind == "episode":
        return viewer_data(load_recording(Path(path)), max_steps)
    run = Path(path) if kind == "checkpoint" else None
    return viewer_data(record(env, run, seed=seed, max_steps=max_steps, greedy=greedy), max_steps)


def _stamp(path: str) -> float:
    try:
        return Path(path).stat().st_mtime if path else 0.0
    except OSError:
        return 0.0


def _has_display() -> bool:
    """Whether a game window can open on the computer running the dashboard (not over ssh without a display)."""
    if os.environ.get("SSH_CONNECTION") and not os.environ.get("DISPLAY"):
        return False
    return sys.platform in ("darwin", "win32") or bool(os.environ.get("DISPLAY"))


def play_command(env: str, source: dict) -> list[str]:
    """`python run.py play ...` for the chosen game and policy source."""
    cmd = ["python", "run.py", "play", "--env", env]
    if source["kind"] == "untrained":
        cmd.append("--untrained")
    else:
        run_dir = Path(source["path"]).parent if source["kind"] == "episode" else Path(source["path"])
        cmd += ["--run", run_dir.name]
    return cmd


def play_panel(tab: str, env: str, source: dict) -> None:
    """Button that opens the live game window (a separate process) with this policy; shown only where a window can open."""
    cmd = play_command(env, source)
    st.markdown("**Play it live**")
    if _has_display():
        if st.button("Open the game window", icon=":material/stadia_controller:", key=f"watch_play::{tab}::{env}",
                     help="Opens a window on this computer with the real game running and this policy at the controls. "
                          "Arrow keys take over (Pong: the paddle; CartPole: the cart), A hands back, Q quits."):
            subprocess.Popen([sys.executable, *cmd[1:]], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            st.toast("Opening the game window (it appears behind the browser if it is not in front).")
        st.caption("Pong's opponent is the cartridge's own computer player (the environment has one joystick), so taking over "
                   "means playing the agent's side against it. From a terminal: `" + " ".join(cmd) + "`.")
    else:
        st.caption("A game window needs a screen: on your own computer run `" + " ".join(cmd) + "`.")


def watch_panel(tab: str) -> None:
    """The whole panel for tab "pg1" or "pg2"."""
    part = PART[tab]
    with st.expander("Watch the game", expanded=True, icon=":material/sports_esports:"):
        st.markdown(
            "One episode of the real environment, step by step: the frame the agent sees, what it does with it "
            "(for Pong-v5 the handout's `preprocess` output and the frame difference the network receives; for "
            "CartPole-v1 the four state variables), the action it takes with the policy's probabilities, the reward "
            "and the reward-to-go. Play it, step through it, or jump to any step from the charts and the table."
        )
        left, right = st.columns([1, 3], gap="medium")
        with left:
            env = st.radio("Game", ENVS, format_func=ENV_LABELS.__getitem__, horizontal=True, key=f"watch_env::{tab}")
            sources = policy_sources(env, part)
            labels = [s["label"] for s in sources]
            source = sources[st.selectbox("Policy", range(len(sources)), format_func=labels.__getitem__, key=f"watch_policy::{tab}::{env}",
                                          help="The episode recorded at the end of a run shows the trained agent even on a computer "
                                               "without its checkpoint; a checkpoint plays a new episode here; the untrained policy "
                                               "is what training starts from.")]
            replay = source["kind"] == "episode"
            seed = st.number_input("Seed", min_value=0, max_value=10_000_000, step=1, key=f"watch_seed::{tab}",
                                   disabled=replay, help="Seeds the environment and the action sampling of a new episode.")
            max_steps = st.number_input("Show at most this many steps (0 = the whole episode)", min_value=0, max_value=100_000, step=100,
                                        key=f"watch_steps::{tab}",
                                        help="A Pong game lasts 1000-3000 steps (about 1 MB of frames); a cap keeps the page small.")
            greedy = st.toggle("Most probable action instead of sampling", key=f"watch_greedy::{tab}", disabled=replay,
                               help="Training and evaluation sample from π(·|s); this shows the policy's deterministic choice.")
            if replay:
                st.caption("Recorded by `python run.py pg` at the end of the run with the trained policy, sampling its actions "
                           "(seed: the run's seed + 20000). The seed and sampling settings above apply to new episodes only.")
            st.caption("The same page from the command line: `python run.py watch --env " + env
                       + ("" if source["kind"] != "checkpoint" else f" --run {Path(source['path']).name}") + "`.")
            play_panel(tab, env, source)
        with right:
            try:
                data = _episode(env, source["kind"], source["path"], _stamp(source["path"]), int(seed), int(max_steps), bool(greedy))
            except Exception as e:  # noqa: BLE001 - a missing renderer (pygame) or an unreadable file
                st.error(f"Could not play the episode: {type(e).__name__}: {e}. On a fresh computer run "
                         "`pip install -r requirements.txt` (CartPole-v1 renders with pygame).")
                return
            html = render_html(data, theme=_theme())
            components.html(html, height=HEIGHT[env], scrolling=True)
            st.download_button("Download this page as standalone HTML", html, file_name=f"watch_{env}.html", mime="text/html",
                               key=f"watch_download::{tab}::{env}")
