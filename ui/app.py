"""Dashboard for Assignment 1. Graders start on the Overview tab and read each part's answers, plots and
validation results in that part's tab; each part's tab can also rerun it, here or on deepdish.

    python run.py ui        (or: streamlit run ui/app.py)
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from report import answers as answer_store  # noqa: E402
from report.build import INSTRUCTIONS_PDF, REPORT_FILE, REPORT_PDF, build_pdfs, build_report  # noqa: E402
from report.questions import QUESTIONS, SECTION_BY_KEY, SECTIONS, Question, questions_for  # noqa: E402
from rl.outputs import FIGURE_BY_KEY, RESULTS, read_json, tests_dir  # noqa: E402
from rl.plotting import BLUE_DARK, BLUE_LIGHT, moving_average  # noqa: E402
from ui import deepdish as dd  # noqa: E402
from ui.explore import explore_panel  # noqa: E402

# job name -> (label, command). Policy-gradient job names double as their results folder.
JOBS = {
    "dp": ("Enumeration, value iteration, policy iteration", ["python", "run.py", "dp"]),
    "tests_dp": ("Validation tests", ["python", "run.py", "tests", "--section", "dp"]),
    "part1_cartpole": ("(a) CartPole-v1", ["python", "run.py", "pg", "--part", "1", "--env", "cartpole"]),
    "part1_pong": ("(b) Pong-v5", ["python", "run.py", "pg", "--part", "1", "--env", "pong"]),
    "tests_pg1": ("Validation tests", ["python", "run.py", "tests", "--section", "pg1"]),
    "part2_cartpole": ("(a) CartPole-v1 with baseline", ["python", "run.py", "pg", "--part", "2", "--env", "cartpole"]),
    "part2_pong": ("(b) Pong-v5 with baseline", ["python", "run.py", "pg", "--part", "2", "--env", "pong"]),
    "tests_pg2": ("Validation tests", ["python", "run.py", "tests", "--section", "pg2"]),
    "setup": ("Install the Python environment (setup.sh --server)", ["bash", "setup.sh", "--server"]),
    "check": ("Environment check (packages, GPU, gym envs)", ["python", "run.py", "check"]),
}
# The jobs each part's tab launches and lists; the one-time setup jobs belong to the sidebar's deepdish panel.
TAB_JOBS = {
    "dp": ("dp", "tests_dp"),
    "pg1": ("part1_cartpole", "part1_pong", "tests_pg1"),
    "pg2": ("part2_cartpole", "part2_pong", "tests_pg2"),
}
SETUP_JOBS = ("setup", "check")
TABS = {
    "overview": "Overview",
    "dp": "2 · Value & Policy Iteration",
    "pg1": "3 · Policy Gradient Part 1",
    "pg2": "Policy Gradient Part 2",
}
CONTENTS_JS = (Path(__file__).resolve().parent / "contents.js").read_text()
TEST_FILES = {"dp": "tests/test_dp.py", "pg1": "tests/test_pg_part1.py", "pg2": "tests/test_pg_part2.py"}
STATUS_STYLE = {"passed": ":green[PASSED]", "failed": ":red[FAILED]", "error": ":orange[ERROR]", "recorded": ":blue[RECORDED]"}

st.set_page_config(page_title="RL Assignment 1", layout="wide", initial_sidebar_state="expanded")
# Streamlit shows the sidebar's collapse arrow only while the pointer is over the sidebar; keep it visible.
st.html("<style>[data-testid='stSidebarCollapseButton'] { visibility: visible !important; }</style>")
# Run and Explore are the interactive panels; a bar in the theme's primary colour on their left sets them apart from the
# written answers below.
ACCENT = st.get_option("theme.primaryColor") or "#2a78d6"
st.html(f"<style>[class*='st-key-interactive_'] {{ border-left: 4px solid {ACCENT}; padding-left: 0.75rem; }}</style>")


@st.cache_data(ttl=10, show_spinner=False)
def server_statuses(cfg_json: str) -> list[dd.ServerStatus]:
    return dd.status_all(json.loads(cfg_json))


@st.cache_data(ttl=3, show_spinner=False)
def _job_rows(target_id: str, _target) -> tuple[list[dict[str, str]], str]:
    return dd.job_status(_target)


def job_rows(target) -> tuple[list[dict[str, str]], str]:
    """Jobs on the target, read once for all tabs (each rerun draws every tab)."""
    return _job_rows(f"{target.label}:{getattr(target, 'remote_dir', '')}", target)


def flash(kind: str, message: str) -> None:
    st.session_state.setdefault("flash", []).append((kind, message))


def show_flash() -> None:
    for kind, message in st.session_state.pop("flash", []):
        getattr(st, kind)(message)


def show_output(key: str) -> None:
    """Output of the last upload/download/launch, kept across reruns."""
    result = st.session_state.get(key)
    if result:
        ok, text = result
        (st.success if ok else st.error)("Done." if ok else "Failed.")
        if text:
            st.code(text, language=None)


# --- sidebar: connection and where jobs run ----------------------------------------------------

def _connect(server: dd.Server) -> None:
    ok, message = dd.connect(server, st.session_state.get("password") or None)
    st.session_state.password = ""
    if ok:
        st.session_state.where = "server"
    flash("success" if ok else "error", message)
    server_statuses.clear()


def _disconnect(server: dd.Server) -> None:
    ok, message = dd.disconnect(server)
    flash("success" if ok else "error", message)
    server_statuses.clear()


def connection_panel(cfg: dict, statuses: list[dd.ServerStatus], here: int | None) -> dd.ServerStatus:
    by_n = {s.server.n: s for s in statuses}
    st.header("Deepdish")
    show_flash()
    for s in statuses:
        dot = ":green[●]" if s.connected else ":gray[○]"
        st.markdown(f"{dot} **{s.server.name}**: {s.summary}")
    st.button("Refresh status", on_click=server_statuses.clear)

    connected = [s.server.n for s in statuses if s.connected]
    if "server_n" not in st.session_state:
        st.session_state.server_n = here or (connected[0] if connected else cfg.get("default_server", cfg["servers"][0]))
    n = st.selectbox("Server", cfg["servers"], key="server_n", format_func=lambda i: f"deepdish{i}")
    status = by_n[n]
    server = status.server
    st.caption(f"ssh `{server.target}` → `{server.hostname}`")

    if here:
        st.success(f"This dashboard is running on deepdish{here}, so jobs run here directly.")
        return status

    user = st.text_input("NetID", value=server.user, help="Saved to config/deepdish.local.json")
    if user.strip() and user.strip() != server.user:
        dd.save_local_config(user=user.strip())
        server_statuses.clear()
        st.rerun()
    st.text_input("Password (only if you have no SSH key)", type="password", key="password",
                  help="Used once to open the shared connection; never saved.")
    left, right = st.columns(2)
    left.button("Connect", type="primary", on_click=_connect, args=(server,))
    right.button("Disconnect", on_click=_disconnect, args=(server,), disabled=not status.ready)
    if n != cfg.get("default_server"):
        st.button(f"Make deepdish{n} the default", on_click=dd.save_local_config, kwargs={"default_server": n})

    with st.expander("How connection status works"):
        st.markdown(
            "A server shows as connected when any of these is true:\n"
            "- **shared SSH connection open**: opened by *Connect*, or by a terminal / VS Code session when "
            "`~/.ssh/config` enables connection sharing (`./setup.sh` offers to add it).\n"
            "- **terminal ssh / VS Code Remote-SSH**: an `ssh` process to that deepdish is running on this computer.\n"
            "- **dashboard is running on this server**: you started the dashboard on deepdish itself.\n\n"
            "Off campus, connect to the Northwestern VPN (GlobalProtect) first."
        )
    return status


def run_settings(cfg: dict, status: dd.ServerStatus, here: int | None):
    """Where jobs run and on which GPU. Every tab's Launch button uses these."""
    st.header("Jobs run on")
    if here:
        target = dd.LocalTarget(f"deepdish{here} (this machine)")
    else:
        if "where" not in st.session_state:
            # Graders, who have no deepdish connection or NetID set up, run on their own computer.
            st.session_state.where = "server" if status.connected or cfg.get("user") else "local"
        where = st.radio("Jobs run on", ["server", "local"], horizontal=True, key="where", label_visibility="collapsed",
                         format_func=lambda w: status.server.name if w == "server" else "this computer")
        if where == "local":
            target = dd.LocalTarget()
        else:
            if "remote_dir" not in st.session_state:
                st.session_state.remote_dir = cfg["remote_dir"]
            target = dd.RemoteTarget(status.server, st.session_state.remote_dir)
    return target, gpu_select(target)


def gpu_select(target) -> str:
    key = f"gpus::{target.label}"
    if st.button("Refresh GPU usage" if key in st.session_state else "Show GPU usage"):
        st.session_state[key] = dd.gpus(target)
    rows, error = st.session_state.get(key, ([], ""))
    if error:
        st.caption(f"No GPU information: {error}")
    usage = {r["index"]: f"GPU {r['index']} · {r['memory used (MiB)']}/{r['memory total (MiB)']} MiB · "
                         f"{r['utilization (%)']}% busy" for r in rows}
    options = ["cpu", *usage]
    if rows:
        default = min(rows, key=lambda r: float(r["memory used (MiB)"] or 0))["index"]
    else:
        default = "0" if target.remote else "cpu"
        if default not in options:
            options.append(default)
    gpu = st.selectbox("GPU (exactly one card)", options, index=options.index(default),
                       format_func=lambda g: "CPU only" if g == "cpu" else usage.get(g, f"GPU {g}"),
                       key=f"gpu_choice::{target.label}")
    st.caption("On deepdish the handout requires a single GPU card: each job sees only this card (CUDA_VISIBLE_DEVICES).")
    return gpu


# --- run panels (one per tab) --------------------------------------------------------------------

def _in_tab(name: str, tab: str) -> bool:
    return any(name == p or name.startswith(f"{p}_") for p in TAB_JOBS[tab])


def _launch(target, name: str, gpu: str, argv: list[str], upload: bool, excludes: list[str]) -> tuple[bool, str]:
    if upload:
        ok, text = dd.upload(target.server, target.remote_dir, excludes)
        if not ok:
            return False, f"Upload failed, so the job was not launched.\n{text}"
    proc = dd.job_start(target, name, gpu, argv)
    text = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, ("Uploaded the code.\n" if upload else "") + text


def run_panel(tab: str, cfg: dict, status: dd.ServerStatus, target, gpu: str) -> None:
    """Launch this tab's jobs and follow them: state, log and, for training runs, the live reward curve."""
    ready = not target.remote or status.connected
    folder = f" in `{target.remote_dir}`" if target.remote else ""
    device = "CPU only" if gpu == "cpu" else f"GPU {gpu}"
    st.caption(f"Runs on **{target.label}**{folder} · {device}. Change both in the sidebar.")
    if not ready:
        st.warning(f"{target.label} is not connected. Press **Connect** in the sidebar, or open a terminal / VS Code session to it.")

    preset = st.radio("Job", TAB_JOBS[tab], format_func=lambda k: JOBS[k][0], horizontal=True, key=f"job_preset::{tab}")
    argv = list(JOBS[preset][1])
    name = preset
    left, right = st.columns([1, 2])
    if preset.startswith("part"):
        tag = left.text_input("Run tag (optional)", key=f"job_tag::{tab}",
                              help="Keeps several runs side by side: results/pg/<run>_<tag>/")
        tag = re.sub(r"[^A-Za-z0-9_-]", "", tag)
        if tag:
            argv += ["--tag", tag]
            name = f"{preset}_{tag}"
    extra = right.text_input("Extra arguments", key=f"job_extra::{tab}", placeholder="e.g. --episodes 3000 --lr 0.001")
    try:
        argv += shlex.split(extra)
    except ValueError as e:
        st.error(f"Extra arguments: {e}")
        return
    st.code(f"CUDA_VISIBLE_DEVICES={'' if gpu == 'cpu' else gpu} {shlex.join(argv)}    # on {target.label}", language="bash")

    upload = target.remote and st.checkbox(
        "Upload code first", value=True, key=f"upload_first::{tab}",
        help="Copies this project to the server, so the job runs your latest code. The server's results/, logs/ "
             "and .venv/ are never overwritten.",
    )
    launch, download, _ = st.columns([1, 1, 2])
    output_key = f"run_output::{tab}"
    if launch.button(f"Launch on {target.label}", type="primary", key=f"launch::{tab}", disabled=not ready):
        with st.spinner(f"{'Uploading and launching' if upload else 'Launching'} on {target.label}…"):
            st.session_state[output_key] = _launch(target, name, gpu, argv, upload, cfg["upload_excludes"])
        st.session_state[f"show_job::{tab}"] = name
        _job_rows.clear()
    if target.remote and download.button("Download results", key=f"download::{tab}", disabled=not ready,
                                         help="Copies results/ back here, so the plots below update."):
        with st.spinner(f"Downloading results from {target.label}…"):
            st.session_state[output_key] = dd.download(target.server, target.remote_dir)
    show_output(output_key)

    st.markdown("**Jobs**")
    if not ready:
        st.caption(f"Connect to {target.label} to see its jobs.")
        return
    auto = st.toggle("Auto-refresh every 5 seconds", key=f"auto_refresh::{tab}")

    @st.fragment(run_every=5 if auto else None)
    def jobs() -> None:
        jobs_panel(tab, target)

    jobs()


def _job_action(action, target, name: str) -> None:
    """Stop or clear a job; as a button callback it runs before the job list is redrawn."""
    action(target, name)
    _job_rows.clear()


def jobs_panel(tab: str, target) -> None:
    rows, error = job_rows(target)
    if error:
        st.error(f"Could not read jobs on {target.label}: {error}")
        return
    rows = [r for r in rows if _in_tab(r["name"], tab)]
    if not rows:
        st.caption(f"No jobs for this tab on {target.label} yet.")
        return
    st.dataframe(pd.DataFrame(rows)[["name", "state", "exit", "gpu", "started", "host", "command"]], hide_index=True)
    names = [r["name"] for r in rows]
    key = f"job_selected::{tab}::{target.label}"
    launched = st.session_state.pop(f"show_job::{tab}", None)
    if launched in names:
        st.session_state[key] = launched
    name = st.selectbox("Show job", names, key=key)
    row = next(r for r in rows if r["name"] == name)
    stop, clear, _ = st.columns([1, 1, 4])
    stop.button("Stop job", key=f"stop::{tab}", disabled=row["state"] != "running",
                on_click=_job_action, args=(dd.job_stop, target, name))
    clear.button("Clear job", key=f"clear::{tab}", disabled=row["state"] == "running",
                 on_click=_job_action, args=(dd.job_clear, target, name))

    if name.startswith("part"):
        records = dd.read_progress(target, name)
        if records:
            df = pd.DataFrame(records)
            df["100-episode moving average"] = moving_average(df["reward"], 100)
            df = df.rename(columns={"reward": "episode reward"})
            st.line_chart(df, x="episode", y=["episode reward", "100-episode moving average"],
                          color=[BLUE_LIGHT, BLUE_DARK], x_label="Training episode", y_label="Episode reward")
            st.caption(f"{len(df)} episodes logged · last reward {df['episode reward'].iloc[-1]:.1f}")
    st.code(dd.job_tail(target, name, 200) or "(log is empty)", language=None)


# --- deepdish setup (sidebar) -------------------------------------------------------------------

def _save_remote_dir() -> None:
    dd.save_local_config(remote_dir=st.session_state.remote_dir.strip() or "~/RL")


def _server_info(server: dd.Server, remote_dir: str) -> tuple[bool, str]:
    info = dd.remote_info(server, remote_dir)
    if "error" in info:
        return False, info["error"]
    if info.get("code") != "yes":
        flash("warning", "The project is not on the server yet: press Upload code.")
    elif info.get("venv", "no") == "no":
        flash("warning", "No Python environment on the server yet: press Install.")
    return True, "\n".join(f"{k:<8} {v}" for k, v in info.items())


def setup_panel(cfg: dict, status: dd.ServerStatus, target, gpu: str, here: int | None) -> None:
    """One-time steps for the machine that runs the jobs, under the connection settings in the sidebar: copy
    the code to deepdish, install its Python environment and check it. Afterwards each part runs from its tab.
    Graders, whose jobs run on their own computer, never see this panel."""
    with st.expander("Set up deepdish (once per server)", icon=":material/build:"):
        ready = not target.remote or status.connected
        if here:
            st.caption(f"The dashboard runs on deepdish{here}, so jobs start in this folder: nothing to upload.")
        else:
            st.caption("Copy this project to the server, install its Python environment, then check it. The "
                       "server's results/, logs/ and .venv/ are never overwritten.")
            if not status.connected:
                st.warning(f"{status.server.name} is not connected: press **Connect** above.")
            elif not status.ready:
                st.caption(f"Connected via {status.summary}. Commands use your SSH key; if you log in with a "
                           "password, press Connect.")
            st.text_input("Project folder on deepdish", key="remote_dir", on_change=_save_remote_dir)
            mirror = st.checkbox("Also delete files there that no longer exist here",
                                 help="results/, logs/ and .venv/ on the server are kept.")
            upload, check = st.columns(2)
            if upload.button("Upload code", type="primary", disabled=not ready):
                with st.spinner(f"Uploading to {status.server.name}:{target.remote_dir}…"):
                    st.session_state.setup_output = dd.upload(status.server, target.remote_dir,
                                                              cfg["upload_excludes"], delete=mirror)
            if check.button("Check server", disabled=not ready):
                with st.spinner(f"Checking {status.server.name}…"):
                    st.session_state.setup_output = _server_info(status.server, target.remote_dir)
        st.markdown("**Python environment**")
        st.caption("*Install* runs `setup.sh --server` there (uploading the code first). When its log ends, "
                   "*Check* should end with `ready` and `test computation on the GPU: ok`.")
        install, check_env = st.columns(2)
        if install.button("Install", disabled=not ready, help=JOBS["setup"][0]):
            with st.spinner(f"Launching on {target.label}…"):
                st.session_state.setup_output = _launch(target, "setup", "cpu", list(JOBS["setup"][1]),
                                                        target.remote, cfg["upload_excludes"])
            _job_rows.clear()
        if check_env.button("Check", disabled=not ready, help=JOBS["check"][0]):
            st.session_state.setup_output = _launch(target, "check", gpu, list(JOBS["check"][1]), False, [])
            _job_rows.clear()
        show_flash()
        show_output("setup_output")
        if ready:
            setup_jobs(target)


def setup_jobs(target) -> None:
    """The setup jobs on the target, and the log of one of them."""
    rows, error = job_rows(target)
    if error:
        st.caption(f"Could not read jobs on {target.label}: {error}")
        return
    rows = [r for r in rows if r["name"] in SETUP_JOBS]
    if not rows:
        return
    st.markdown("  \n".join(f"**{r['name']}**: {r['state']}" + (f" (exit {r['exit']})" if r["exit"] else "")
                            + f" · {r['started']}" for r in rows))
    name = st.selectbox("Log", [r["name"] for r in rows], key="setup_log")
    st.code(dd.job_tail(target, name, 30) or "(log is empty)", language=None)
    st.button("Refresh", key="setup_refresh", on_click=_job_rows.clear)


# --- section tabs ----------------------------------------------------------------------------

def _save_answer(question_id: str, key: str) -> None:
    answer_store.save_answer(question_id, st.session_state[key])


def answer_box(q: Question, saved: dict) -> None:
    """The answer as formatted text for reading, with the box for writing it below."""
    key = f"answer::{q.id}"
    if key not in st.session_state:
        st.session_state[key] = saved.get(q.id, "")
    text = st.session_state[key].strip()
    st.markdown("**Answer**")
    if text:
        st.markdown(text)
    else:
        st.caption("Not answered yet.")
    with st.expander("Edit answer" if text else "Write answer", expanded=not text):
        st.text_area("Answer (Markdown)", key=key, height=180, on_change=_save_answer, args=(q.id, key),
                     label_visibility="collapsed",
                     placeholder="Markdown. Saved to report/answers.json when you click outside the box.")


def show_figures(keys: tuple[str, ...]) -> None:
    columns = st.columns(len(keys)) if len(keys) > 1 else st.columns([2, 1])[:1]
    for column, key in zip(columns, keys):
        fig = FIGURE_BY_KEY[key]
        with column:
            if fig.file.exists():
                st.image(str(fig.file), caption=fig.caption)
            else:
                st.info(f"**Plot not generated yet.** {fig.caption}\n\nRun `{fig.command}` from the Run panel above, "
                        "then *Download results*.")


def show_eval(run: str) -> None:
    stats = read_json(RESULTS / "pg" / run / "eval.json")
    if not stats:
        st.caption("Mean and standard deviation over the 500 evaluation episodes appear here after the run.")
        return
    mean, std, episodes = st.columns(3)
    mean.metric("Mean episode reward", f"{stats['mean']:.2f}")
    std.metric("Standard deviation", f"{stats['std']:.2f}")
    episodes.metric("Evaluation episodes", stats["episodes"])


def show_tests(section: str) -> None:
    data = read_json(tests_dir() / f"{section}.json")
    if not data or not data.get("tests"):
        st.info(f"No validation results yet. Add experiments to `{TEST_FILES[section]}` and run "
                f"`python run.py tests --section {section}` (or *Validation tests* in the Run panel above).")
        return
    counts = {s: sum(t["status"] == s for t in data["tests"]) for s in STATUS_STYLE}
    st.markdown(" · ".join(f"{STATUS_STYLE[s]} {n}" for s, n in counts.items() if n)
                + f"  \nLast run {data['updated']} on {data['host']}")
    for t in data["tests"]:
        with st.expander(f"{t['title']}: {t['status']}"):
            st.markdown(STATUS_STYLE[t["status"]] + f" in {t['duration_s']} s")
            if t["what"]:
                st.markdown(f"**What it does.** {t['what']}")
            if t["how"]:
                st.markdown(f"**How it validates.** {t['how']}")
            for c in t["checks"]:
                mark = ":green[✓]" if c["passed"] else ":red[✗]"
                st.markdown(f"{mark} {c['name']}" + (f": `{c['detail']}`" if c["detail"] else ""))
            if t["metrics"]:
                st.dataframe(pd.DataFrame([{"metric": k, "value": str(v)} for k, v in t["metrics"].items()]), hide_index=True)
            for name, rows in t["tables"].items():
                st.markdown(f"**{name}**")
                st.dataframe(pd.DataFrame(rows), hide_index=True)
            for path in t["figures"]:
                if (ROOT / path).exists():
                    st.image(str(ROOT / path))
            for note in t["notes"]:
                st.markdown(f"- {note}")
            if t["error"]:
                st.code(t["error"], language=None)
            if t["stdout"].strip():
                st.caption("Output (last 40 lines)")
                st.code("\n".join(t["stdout"].splitlines()[-40:]), language=None)


def section_tab(key: str, cfg: dict, status: dd.ServerStatus, target, gpu: str) -> None:
    section = SECTION_BY_KEY[key]
    st.header(section.title, anchor=key)
    st.caption(f"{section.points} points · {section.breakdown}")
    with st.container(key=f"interactive_run_{key}"):
        with st.expander("Run", expanded=True, icon=":material/play_arrow:"):
            run_panel(key, cfg, status, target, gpu)
    if key == "dp":
        with st.container(key="interactive_explore"):
            explore_panel()
    saved = answer_store.load()["answers"]
    for q in questions_for(key):
        with st.container(border=True):
            st.subheader(q.title, anchor=q.id)
            st.markdown("\n".join(f"> {line}" for line in q.prompt.splitlines()))
            if q.figures:
                show_figures(q.figures)
            for run in q.evals:
                show_eval(run)
            if q.tests:
                show_tests(q.tests)
            answer_box(q, saved)


# --- overview tab ----------------------------------------------------------------------------

def _git_remote() -> dict[str, str]:
    proc = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"], capture_output=True, text=True)
    match = re.search(r"github\.com[:/]([^/]+)/(.+?)(?:\.git)?$", proc.stdout.strip())
    if not match:
        return {}
    user, repo = match.groups()
    return {"github_username": user, "repo_name": repo, "repo_url": f"https://github.com/{user}/{repo}"}


def _save_meta(field: str) -> None:
    answer_store.save_meta(field, st.session_state[f"meta::{field}"].strip())


def _fill_from_git(values: dict[str, str]) -> None:
    for field, value in values.items():
        st.session_state[f"meta::{field}"] = value
        answer_store.save_meta(field, value)


GUIDE = """
- **Tabs** above: this overview and one tab per part of the handout.
- **Contents**, at the right edge of the page, lists the sections of the open tab; the entry for the section on
  screen is highlighted, and the highlight glides to the next entry as you scroll. Click an entry to jump to it
  (the list needs a window about 1250 px wide).
- **A part's tab** lists the handout's questions in order, each in a box: the question, the required plots, the
  evaluation mean and standard deviation (policy gradient), the validation tests and the **Answer**. Click a
  validation test to see what it does, how it validates the algorithm, its checks, metrics and figures.
- **The Run panel** at the top of a part's tab reruns that part: choose a job, press **Launch**, and follow its log
  and reward curve under **Jobs**. When it has finished, press R to reload the plots. Click the panel's header to
  collapse it.
- **The sidebar** sets where jobs run (this computer or deepdish) and on which GPU; its **Set up deepdish** panel
  is only needed to run jobs on the deepdish GPU servers. Hide it with « at its top and bring it back with » at
  the top left.
- **If the dashboard stops loading** (a blank page or a connection error): reload it once the terminal shows
  `Dashboard: http://localhost:8501`. *If the dashboard does not load*, in the instructions below (section 3),
  lists the other steps and the report as the fallback.

Nothing has to be run to grade: the plots, statistics and test results shown are the submitted ones. Rerunning a
part overwrites them in your copy (`git checkout results/` restores them); a **Run tag** keeps a policy-gradient
rerun in a separate folder.
"""


def overview_tab() -> None:
    data = answer_store.load()
    meta = data["meta"]
    st.header("Overview", anchor="overview")
    url = meta.get("repo_url", "").strip()
    details = {**meta, "repo_url": f"[{url}]({url})" if url else ""}
    st.markdown("  \n".join(f"**{label}:** {details.get(field) or '*not filled in*'}"
                            for field, label in answer_store.META_FIELDS.items()))
    with st.expander("Edit submission details"):
        for field, label in answer_store.META_FIELDS.items():
            key = f"meta::{field}"
            if key not in st.session_state:
                st.session_state[key] = meta.get(field, "")
            st.text_input(label, key=key, on_change=_save_meta, args=(field,))
        remote = _git_remote()
        if remote:
            st.button(f"Fill from git remote ({remote['repo_url']})", on_click=_fill_from_git, args=(remote,))

    st.subheader("Where to find each part")
    rows = [("Submission details, instructions for running the code", "", TABS["overview"],
             "this tab; the instructions are also a PDF (below)")]
    rows += [(s.title, f"{s.points} ({s.breakdown})", TABS[s.key], " · ".join(q.title for q in questions_for(s.key)))
             for s in SECTIONS if s.key in TABS]
    st.markdown("\n".join(["| Part of the handout | Points | Tab | Sections, in order |", "|---|---|---|---|",
                           *(f"| {' | '.join(row)} |" for row in rows)]))

    st.subheader("Using this dashboard")
    st.markdown(GUIDE)

    for q in questions_for("submission"):
        with st.expander(q.title):
            st.caption(f"The handout asks for \"{q.prompt.rstrip('.').lower()}\". Written in `report/{q.source}`; "
                       f"*Build* turns it into this section of the report and into `report/{INSTRUCTIONS_PDF.name}`.")
            st.markdown(answer_store.text(q, data))

    st.subheader("Progress")
    answered = sum(bool(answer_store.text(q, data).strip()) for q in QUESTIONS)
    figures = list(FIGURE_BY_KEY.values())
    tests = [t for s in TEST_FILES for t in (read_json(tests_dir() / f"{s}.json", {}) or {}).get("tests", [])]
    a, f, t = st.columns(3)
    a.metric("Questions answered", f"{answered} / {len(QUESTIONS)}")
    f.metric("Required plots generated", f"{sum(fig.file.exists() for fig in figures)} / {len(figures)}")
    t.metric("Validation tests passing", f"{sum(x['status'] == 'passed' for x in tests)} / {len(tests)}")

    st.subheader("Report")
    columns = st.columns(3)
    for column, path, mime in zip(columns, (REPORT_PDF, INSTRUCTIONS_PDF, REPORT_FILE),
                                  ("application/pdf", "application/pdf", "text/markdown")):
        if path.exists():
            column.download_button(f"Download {path.name}", path.read_bytes(), file_name=path.name, mime=mime)
    st.caption("*Build* writes report.md, then report.pdf (the submitted report) and instructions.pdf (the "
               "instructions for running the code on their own), all in report/. The same as `python run.py report`.")
    st.button("Build report and PDFs", on_click=_build)
    show_output("build_output")
    if REPORT_FILE.exists():
        with st.expander("Markdown source"):
            st.code(REPORT_FILE.read_text(), language="markdown")


def _build() -> None:
    build_report()
    try:
        written = [REPORT_FILE, *build_pdfs()]
    except ImportError as e:
        st.session_state.build_output = (False, f"Wrote {REPORT_FILE.relative_to(ROOT)}, but not the PDFs: {e}.\n"
                                                "Install reportlab with: pip install -r requirements-ui.txt")
        return
    st.session_state.build_output = (True, "Wrote " + ", ".join(str(p.relative_to(ROOT)) for p in written))


# --- contents (right edge of the page) --------------------------------------------------------

def contents_panel() -> None:
    """The headings of the open tab, listed at the right edge of the page with the one on screen highlighted
    (ui/contents.js). Streamlit draws only inside its own elements, so a zero-height component installs the
    script into the page itself, once per page load; from then on it follows reruns and tab switches itself."""
    accent = st.get_option("theme.primaryColor") or "#2a78d6"
    script = json.dumps(CONTENTS_JS.replace("__ACCENT__", accent))
    components.html(
        "<script>(function () {\n"
        "  var page = window.parent;\n"
        "  if (!page || page === window) return;\n"
        "  if (page.__rlContents) { page.__rlContents.rebuild(); return; }\n"
        "  var s = page.document.createElement('script');\n"
        f"  s.textContent = {script};\n"
        "  page.document.body.appendChild(s);\n"
        "})();</script>",
        height=0,
    )


# --- page ------------------------------------------------------------------------------------

cfg = dd.load_config()
here = dd.local_server()
statuses = server_statuses(json.dumps(cfg, sort_keys=True))
with st.sidebar:
    status = connection_panel(cfg, statuses, here)
    target, gpu = run_settings(cfg, status, here)
    if target.remote or here:
        setup_panel(cfg, status, target, gpu, here)

st.title("Assignment 1: Reinforcement Learning")
dot = ":green[●]" if status.connected else ":gray[○]"
st.markdown(f"{dot} **{status.server.name}**: {status.summary} · due Oct 26, 2026, 11:59 pm")

overview, dp, pg1, pg2 = st.tabs(list(TABS.values()))
with overview:
    overview_tab()
with dp:
    section_tab("dp", cfg, status, target, gpu)
with pg1:
    section_tab("pg1", cfg, status, target, gpu)
with pg2:
    section_tab("pg2", cfg, status, target, gpu)
contents_panel()
