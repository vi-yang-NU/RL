"""Dashboard for Assignment 1: connect to deepdish, upload and run the code, and review the
plots, validation results and discussion answers for each part.

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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from report import answers as answer_store  # noqa: E402
from report.build import REPORT_FILE, build_report  # noqa: E402
from report.questions import QUESTIONS, SECTION_BY_KEY, Question, questions_for  # noqa: E402
from rl.outputs import FIGURE_BY_KEY, RESULTS, read_json, tests_dir  # noqa: E402
from rl.plotting import BLUE_DARK, BLUE_LIGHT, moving_average  # noqa: E402
from ui import deepdish as dd  # noqa: E402

# job name -> (label, command). Policy-gradient job names double as their results folder.
JOBS = {
    "dp": ("Section 2: enumeration, value iteration, policy iteration", ["python", "run.py", "dp"]),
    "part1_cartpole": ("Part 1 (a): CartPole-v1", ["python", "run.py", "pg", "--part", "1", "--env", "cartpole"]),
    "part1_pong": ("Part 1 (b): Pong-v5", ["python", "run.py", "pg", "--part", "1", "--env", "pong"]),
    "part2_cartpole": ("Part 2 (a): CartPole-v1 with baseline", ["python", "run.py", "pg", "--part", "2", "--env", "cartpole"]),
    "part2_pong": ("Part 2 (b): Pong-v5 with baseline", ["python", "run.py", "pg", "--part", "2", "--env", "pong"]),
    "tests_dp": ("Validation tests: Section 2", ["python", "run.py", "tests", "--section", "dp"]),
    "tests_pg1": ("Validation tests: Part 1", ["python", "run.py", "tests", "--section", "pg1"]),
    "tests_pg2": ("Validation tests: Part 2", ["python", "run.py", "tests", "--section", "pg2"]),
    "check": ("Environment check (packages, GPU, gym envs)", ["python", "run.py", "check"]),
    "setup": ("Install the Python environment (setup.sh --server)", ["bash", "setup.sh", "--server"]),
}
TEST_FILES = {"dp": "tests/test_dp.py", "pg1": "tests/test_pg_part1.py", "pg2": "tests/test_pg_part2.py"}
STATUS_STYLE = {"passed": ":green[PASSED]", "failed": ":red[FAILED]", "error": ":orange[ERROR]", "recorded": ":blue[RECORDED]"}

st.set_page_config(page_title="RL Assignment 1", layout="wide")


@st.cache_data(ttl=10, show_spinner=False)
def server_statuses(cfg_json: str) -> list[dd.ServerStatus]:
    return dd.status_all(json.loads(cfg_json))


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


# --- sidebar: connection -----------------------------------------------------------------------

def _connect(server: dd.Server) -> None:
    ok, message = dd.connect(server, st.session_state.get("password") or None)
    st.session_state.password = ""
    flash("success" if ok else "error", message)
    server_statuses.clear()


def _disconnect(server: dd.Server) -> None:
    ok, message = dd.disconnect(server)
    flash("success" if ok else "error", message)
    server_statuses.clear()


def sidebar(cfg: dict, statuses: list[dd.ServerStatus], here: int | None) -> dd.ServerStatus:
    by_n = {s.server.n: s for s in statuses}
    with st.sidebar:
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


# --- run tab ---------------------------------------------------------------------------------

def _save_remote_dir() -> None:
    dd.save_local_config(remote_dir=st.session_state.remote_dir.strip() or "~/RL")


def run_tab(cfg: dict, status: dd.ServerStatus, here: int | None) -> None:
    server = status.server
    if here:
        target = dd.LocalTarget(f"deepdish{here} (this machine)")
        st.info(f"The dashboard runs on deepdish{here}: no upload needed, jobs start in this folder.")
    else:
        where = st.radio("Run on", [server.name, "this computer"], horizontal=True, key="where")
        if where == "this computer":
            target = dd.LocalTarget()
        else:
            if "remote_dir" not in st.session_state:
                st.session_state.remote_dir = cfg["remote_dir"]
            target = dd.RemoteTarget(server, st.session_state.remote_dir)
            code_panel(cfg, status, target)

    st.subheader("GPU")
    gpu = gpu_panel(target)

    st.subheader("Launch a job")
    launch_panel(target, gpu)

    st.subheader("Jobs")
    if target.remote and not status.connected:
        st.caption(f"Connect to {server.name} to see its jobs.")
        return
    auto = st.toggle("Auto-refresh every 5 seconds", key="auto_refresh")

    @st.fragment(run_every=5 if auto else None)
    def jobs() -> None:
        jobs_panel(target)

    jobs()


def code_panel(cfg: dict, status: dd.ServerStatus, target: dd.RemoteTarget) -> None:
    server = status.server
    st.subheader(f"Code on {server.name}")
    if not status.connected:
        st.warning(f"{server.name} is not connected. Press **Connect** in the sidebar, or open a terminal / VS Code session to it.")
    elif not status.ready:
        st.caption(f"Connected via {status.summary}. Commands use your SSH key; if you log in with a password, press Connect.")
    st.text_input("Project folder on deepdish", key="remote_dir", on_change=_save_remote_dir)
    mirror = st.checkbox("Also delete files on deepdish that no longer exist here (results/, logs/ and .venv/ are kept)")
    upload, download, check = st.columns(3)
    if upload.button("Upload code", type="primary"):
        with st.spinner(f"Uploading to {server.name}:{target.remote_dir}…"):
            st.session_state.code_output = dd.upload(server, target.remote_dir, cfg["upload_excludes"], delete=mirror)
    if download.button("Download results"):
        with st.spinner(f"Downloading results from {server.name}…"):
            st.session_state.code_output = dd.download(server, target.remote_dir)
    if check.button("Check server"):
        with st.spinner(f"Checking {server.name}…"):
            info = dd.remote_info(server, target.remote_dir)
        if "error" in info:
            st.session_state.code_output = (False, info["error"])
        else:
            st.session_state.code_output = (True, "\n".join(f"{k:<8} {v}" for k, v in info.items()))
            if info.get("code") != "yes":
                flash("warning", "The project is not on the server yet: press Upload code.")
            elif info.get("venv", "no") == "no":
                flash("warning", "No Python environment on the server yet: launch the job 'Install the Python environment'.")
    show_flash()
    show_output("code_output")


def gpu_panel(target) -> str:
    key = f"gpus::{target.label}"
    if st.button("Show GPUs"):
        st.session_state[key] = dd.gpus(target)
    rows, error = st.session_state.get(key, ([], ""))
    if error:
        st.caption(f"No GPU information: {error}")
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    options = ["cpu"] + [r["index"] for r in rows]
    if rows:
        default = min(rows, key=lambda r: float(r["memory used (MiB)"] or 0))["index"]
    else:
        default = "0" if target.remote else "cpu"
        if default not in options:
            options.append(default)
    gpu = st.selectbox("GPU for the next job (exactly one card)", options, index=options.index(default),
                       format_func=lambda g: "CPU only" if g == "cpu" else f"GPU {g}", key=f"gpu_choice::{target.label}")
    st.caption("On deepdish the handout requires a single GPU card: the job sees only this card (CUDA_VISIBLE_DEVICES).")
    return gpu


def launch_panel(target, gpu: str) -> None:
    preset = st.selectbox("Job", list(JOBS), format_func=lambda k: JOBS[k][0], key="job_preset")
    label, argv = JOBS[preset]
    argv = list(argv)
    name = preset
    left, right = st.columns([1, 2])
    if preset.startswith("part"):
        tag = left.text_input("Run tag (optional)", key="job_tag", help="Keeps several runs side by side: results/pg/<run>_<tag>/")
        tag = re.sub(r"[^A-Za-z0-9_-]", "", tag)
        if tag:
            argv += ["--tag", tag]
            name = f"{preset}_{tag}"
    extra = right.text_input("Extra arguments", key="job_extra", placeholder="e.g. --episodes 3000 --lr 0.001")
    try:
        argv += shlex.split(extra)
    except ValueError as e:
        st.error(f"Extra arguments: {e}")
        return
    job_gpu = "cpu" if preset == "setup" else gpu
    st.code(f"CUDA_VISIBLE_DEVICES={'' if job_gpu == 'cpu' else job_gpu} {shlex.join(argv)}    # on {target.label}", language="bash")
    if st.button(f"Launch on {target.label}", type="primary"):
        proc = dd.job_start(target, name, job_gpu, argv)
        st.session_state.launch_output = (proc.returncode == 0, (proc.stdout + proc.stderr).strip())
    show_output("launch_output")


def jobs_panel(target) -> None:
    rows, error = dd.job_status(target)
    if error:
        st.error(f"Could not read jobs on {target.label}: {error}")
        return
    if not rows:
        st.caption(f"No jobs on {target.label} yet.")
        return
    st.dataframe(pd.DataFrame(rows)[["name", "state", "exit", "gpu", "started", "host", "command"]], hide_index=True)
    names = [r["name"] for r in rows]
    name = st.selectbox("Show job", names, key=f"job_selected::{target.label}")
    row = next(r for r in rows if r["name"] == name)
    stop, clear, _ = st.columns([1, 1, 4])
    if stop.button("Stop job", disabled=row["state"] != "running"):
        dd.job_stop(target, name)
        st.rerun(scope="fragment")
    if clear.button("Clear job", disabled=row["state"] == "running"):
        dd.job_clear(target, name)
        st.rerun(scope="fragment")

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


# --- section tabs ----------------------------------------------------------------------------

def _save_answer(question_id: str, key: str) -> None:
    answer_store.save_answer(question_id, st.session_state[key])


def answer_box(q: Question, saved: dict) -> None:
    key = f"answer::{q.id}"
    if key not in st.session_state:
        st.session_state[key] = saved.get(q.id, "")
    st.text_area("Your answer (Markdown)", key=key, height=150, on_change=_save_answer, args=(q.id, key),
                 placeholder="Not answered yet. Saved to report/answers.json when you click outside the box.")


def show_figures(keys: tuple[str, ...]) -> None:
    columns = st.columns(len(keys)) if len(keys) > 1 else st.columns([2, 1])[:1]
    for column, key in zip(columns, keys):
        fig = FIGURE_BY_KEY[key]
        with column:
            if fig.file.exists():
                st.image(str(fig.file), caption=fig.caption)
            else:
                st.info(f"**Plot not generated yet.** {fig.caption}\n\nRun `{fig.command}` (Run tab), then *Download results*.")


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
                f"`python run.py tests --section {section}` (or the *Validation tests* job).")
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


def section_tab(key: str) -> None:
    section = SECTION_BY_KEY[key]
    st.header(section.title)
    st.caption(f"{section.points} points · {section.breakdown}")
    saved = answer_store.load()["answers"]
    for q in questions_for(key):
        with st.container(border=True):
            st.subheader(q.title)
            st.markdown("\n".join(f"> {line}" for line in q.prompt.splitlines()))
            if q.figures:
                show_figures(q.figures)
            for run in q.evals:
                show_eval(run)
            if q.tests:
                show_tests(q.tests)
            answer_box(q, saved)


# --- report tab ------------------------------------------------------------------------------

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


def report_tab() -> None:
    data = answer_store.load()
    st.header("Report")
    st.subheader("Submission details")
    for field, label in answer_store.META_FIELDS.items():
        key = f"meta::{field}"
        if key not in st.session_state:
            st.session_state[key] = data["meta"].get(field, "")
        st.text_input(label, key=key, on_change=_save_meta, args=(field,))
    remote = _git_remote()
    if remote:
        st.button(f"Fill from git remote ({remote['repo_url']})", on_click=_fill_from_git, args=(remote,))
    for q in questions_for("submission"):
        with st.container(border=True):
            st.subheader(q.title)
            st.markdown(f"> {q.prompt}  \nThe README's run instructions are a good starting point.")
            answer_box(q, data["answers"])

    st.subheader("Progress")
    answered = sum(bool(data["answers"].get(q.id, "").strip()) for q in QUESTIONS)
    figures = list(FIGURE_BY_KEY.values())
    tests = [t for s in TEST_FILES for t in (read_json(tests_dir() / f"{s}.json", {}) or {}).get("tests", [])]
    a, f, t = st.columns(3)
    a.metric("Questions answered", f"{answered} / {len(QUESTIONS)}")
    f.metric("Required plots generated", f"{sum(fig.file.exists() for fig in figures)} / {len(figures)}")
    t.metric("Validation tests passing", f"{sum(x['status'] == 'passed' for x in tests)} / {len(tests)}")

    st.subheader("Build")
    if st.button("Build report/report.md", type="primary"):
        build_report()
        st.success(f"Wrote {REPORT_FILE.relative_to(ROOT)}")
    if REPORT_FILE.exists():
        text = REPORT_FILE.read_text()
        st.download_button("Download report.md", text, file_name="report.md")
        with st.expander("Markdown source"):
            st.code(text, language="markdown")
    st.caption("For the Canvas PDF: `cd report && pandoc report.md -o report.pdf`, "
               "or open report.md in VS Code and export it with a Markdown-to-PDF extension.")


# --- page ------------------------------------------------------------------------------------

cfg = dd.load_config()
here = dd.local_server()
statuses = server_statuses(json.dumps(cfg, sort_keys=True))
status = sidebar(cfg, statuses, here)

st.title("Assignment 1: Reinforcement Learning")
dot = ":green[●]" if status.connected else ":gray[○]"
st.markdown(f"{dot} **{status.server.name}**: {status.summary} · due Oct 26, 2026, 11:59 pm")

run, dp, pg1, pg2, report = st.tabs(
    ["Run", "2 · Value & Policy Iteration", "3 · Policy Gradient Part 1", "Policy Gradient Part 2", "Report"]
)
with run:
    run_tab(cfg, status, here)
with dp:
    section_tab("dp")
with pg1:
    section_tab("pg1")
with pg2:
    section_tab("pg2")
with report:
    report_tab()
