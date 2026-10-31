"""Connecting to deepdish, moving code and results, and running jobs there.

Everything goes through the system `ssh` and `rsync`, so the dashboard uses the same keys,
~/.ssh/config aliases and shared connections as your terminal and VS Code.

Settings come from config/deepdish.json, overlaid in turn with config/deepdish.local.json, .env
(DEEPDISH_USER=..., see .env.example) and DEEPDISH_* variables in the process environment, so the
same connection works from the dashboard, from `python run.py deepdish` and from any tool or platform
that sets environment variables. The dashboard and setup.sh save your choices to .env.

A server counts as connected when any of these hold:
  * this process runs on that deepdish machine (e.g. the dashboard was started over VS Code Remote-SSH),
  * an SSH ControlMaster socket for it is open (created by the dashboard, a terminal or VS Code),
  * an `ssh` process to it is running on this computer (a terminal session or VS Code Remote-SSH).
"""
from __future__ import annotations

import getpass
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config" / "deepdish.json"
LOCAL_CONFIG_FILE = ROOT / "config" / "deepdish.local.json"
ENV_FILE = ROOT / ".env"
ENV_PREFIX = "DEEPDISH_"
ENV_HEADER = ("# Your deepdish settings: ignored by git and never uploaded. .env.example lists every variable;\n"
              "# a DEEPDISH_* variable set in the environment takes precedence over this file.\n")
ASKPASS = Path(__file__).resolve().parent / "askpass.sh"
JOB_FIELDS = ("name", "state", "exit", "pid", "gpu", "started", "host", "command")

DEEPDISH_RE = re.compile(r"deepdish(\d+)", re.IGNORECASE)
EDITOR_RE = re.compile(r"visual studio code|code helper|vscode|\bcode\b|cursor|windsurf", re.IGNORECASE)


# --- configuration -------------------------------------------------------------------------

def env_var(key: str) -> str:
    """The environment variable that sets a config/deepdish.json key: user -> DEEPDISH_USER."""
    return ENV_PREFIX + key.upper()


def read_env_file(path: Path | None = None) -> dict[str, str]:
    """The KEY=value lines of a .env file. Blank lines and comments are skipped, `export ` is allowed,
    quotes around a value are removed and an unquoted value ends at a ` #` comment."""
    path = ENV_FILE if path is None else path
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def _typed(base: dict, text: Mapping[str, str]) -> dict:
    """The DEEPDISH_* entries of `text`, as settings with the types config/deepdish.json gives them
    (lists are comma-separated). An empty value counts as unset."""
    settings = {}
    for key, like in base.items():
        var = env_var(key)
        value = text.get(var, "").strip()
        if not value:
            continue
        try:
            if isinstance(like, bool):
                settings[key] = value.lower() in ("1", "true", "yes", "on")
            elif isinstance(like, int):
                settings[key] = int(value)
            elif isinstance(like, list):
                items = [item.strip() for item in value.split(",") if item.strip()]
                settings[key] = [type(like[0])(item) for item in items] if like else items
            else:
                settings[key] = value
        except ValueError as e:
            raise ValueError(f"{var}={value!r} is not a valid {key} ({type(like).__name__} expected)") from e
    return settings


def config_layers(environ: Mapping[str, str] | None = None) -> list[tuple[str, dict]]:
    """The settings layers, lowest priority first; each later layer overrides the keys it sets."""
    environ = os.environ if environ is None else environ
    base = json.loads(CONFIG_FILE.read_text())
    layers = [("config/deepdish.json", base)]
    if LOCAL_CONFIG_FILE.exists():
        layers.append(("config/deepdish.local.json", json.loads(LOCAL_CONFIG_FILE.read_text())))
    layers.append((".env", _typed(base, read_env_file())))
    layers.append(("environment", _typed(base, {k: v for k, v in environ.items() if k.startswith(ENV_PREFIX)})))
    return layers


def load_config(environ: Mapping[str, str] | None = None) -> dict:
    """config/deepdish.json overlaid with config/deepdish.local.json, .env and DEEPDISH_* environment variables."""
    cfg: dict = {}
    for _, layer in config_layers(environ):
        cfg.update(layer)
    return cfg


def setting_sources(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """For each setting, the layer its effective value comes from (a name from config_layers)."""
    sources: dict[str, str] = {}
    for name, layer in config_layers(environ):
        for key in layer:
            sources[key] = name
    return sources


def _env_line_key(line: str) -> str | None:
    text = line.strip()
    if not text or text.startswith("#") or "=" not in text:
        return None
    if text.startswith("export "):
        text = text[len("export "):].lstrip()
    return text.partition("=")[0].strip() or None


def _env_text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = ",".join(map(str, value)) if isinstance(value, list) else str(value)
    return f'"{text}"' if any(c in text for c in " \t#\"\'") else text


def save_settings(**updates) -> None:
    """Remember settings in .env (DEEPDISH_USER=..., DEEPDISH_DEFAULT_SERVER=...). Other lines, comments
    included, are kept; a variable already there is replaced in place."""
    pending = {env_var(key): _env_text(value) for key, value in updates.items()}
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else ENV_HEADER.rstrip("\n").splitlines()
    out = []
    for line in lines:
        key = _env_line_key(line)
        out.append(f"{key}={pending.pop(key)}" if key in pending else line)
    out.extend(f"{key}={value}" for key, value in pending.items())
    ENV_FILE.write_text("\n".join(out) + "\n")


@dataclass(frozen=True)
class Server:
    n: int
    target: str  # what ssh is given: an ~/.ssh/config alias or user@host
    hostname: str
    user: str
    shared_control_path: bool  # ~/.ssh/config sets ControlPath, so terminal/VS Code sessions share sockets
    control_path: str
    control_persist: str
    connect_timeout: int
    identity_file: str = ""  # an SSH key to offer besides ssh's own choices (from the identity_file setting)

    @property
    def name(self) -> str:
        return f"deepdish{self.n}"

    def ssh_options(self, master: bool = False) -> list[str]:
        opts = [
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-o", "ServerAliveInterval=30",
            "-o", "StrictHostKeyChecking=accept-new",
        ]
        if self.identity_file:
            opts += ["-o", f"IdentityFile={self.identity_file}"]
        if not self.shared_control_path:
            opts += ["-o", f"ControlPath={self.control_path}"]
        # Only connect() opens a master. A master started from any other call would keep
        # our stdout/stderr pipes open in the background and hang the call.
        if master:
            opts += ["-o", "ControlMaster=auto", "-o", f"ControlPersist={self.control_persist}"]
        else:
            opts += ["-o", "ControlMaster=no"]
        return opts


_RESOLVED: dict[tuple, Server] = {}


def _ssh_config(target: str) -> dict[str, str]:
    """Settings ssh would use for `target` (`ssh -G`, no connection made)."""
    proc = _run(["ssh", "-G", target], timeout=5)
    settings: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, _, value = line.partition(" ")
        settings.setdefault(key.lower(), value)
    return settings


def resolve_server(n: int, cfg: dict) -> Server:
    """Use your ~/.ssh/config alias (e.g. deepdish3) when it exists, else user@host from the config."""
    key = (n, json.dumps(cfg, sort_keys=True))
    if key in _RESOLVED:
        return _RESOLVED[key]
    alias = cfg["alias_template"].format(n=n)
    user = cfg.get("user") or ""
    settings = _ssh_config(alias)
    if settings.get("hostname", alias) != alias:
        target = f"{user}@{alias}" if user else alias
    else:
        target = f"{user or getpass.getuser()}@{cfg['host_template'].format(n=n)}"
    settings = _ssh_config(target) if target != alias else settings
    server = Server(
        n=n,
        target=target,
        hostname=settings.get("hostname", target.split("@")[-1]),
        user=settings.get("user", user),
        shared_control_path="controlpath" in settings and settings["controlpath"] != "none",
        control_path=settings.get("controlpath", cfg["control_path"]),
        control_persist=cfg.get("control_persist", "4h"),
        connect_timeout=int(cfg.get("connect_timeout", 10)),
        identity_file=cfg.get("identity_file") or "",
    )
    _RESOLVED[key] = server
    return server


# --- connection status -----------------------------------------------------------------------

@dataclass(frozen=True)
class Link:
    kind: str  # "local", "socket" or "process"
    label: str


@dataclass(frozen=True)
class ServerStatus:
    server: Server
    links: tuple[Link, ...]

    @property
    def connected(self) -> bool:
        return bool(self.links)

    @property
    def ready(self) -> bool:
        """The dashboard can run commands without asking for a password."""
        return any(link.kind in ("local", "socket") for link in self.links)

    @property
    def summary(self) -> str:
        return "; ".join(link.label for link in self.links) or "not connected"


@dataclass(frozen=True)
class SshProcess:
    pid: int
    n: int
    source: str


def local_server() -> int | None:
    """The deepdish number of this machine, if the dashboard is running on deepdish."""
    match = DEEPDISH_RE.search(socket.gethostname())
    return int(match.group(1)) if match else None


def master_alive(server: Server) -> bool:
    return _run(["ssh", *server.ssh_options(), "-O", "check", server.target], timeout=5).returncode == 0


def ssh_processes() -> list[SshProcess]:
    """ssh sessions to deepdish started on this computer from a terminal or VS Code."""
    proc = _run(["ps", "-axo", "pid=,ppid=,tty=,command="], timeout=5)
    table: dict[int, tuple[int, str, str]] = {}
    for line in proc.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[0].isdigit() and parts[1].isdigit():
            table[int(parts[0])] = (int(parts[1]), parts[2], parts[3])

    me = os.getpid()
    found = []
    for pid, (_, tty, command) in table.items():
        argv = command.split()
        if os.path.basename(argv[0]) != "ssh" or "-O" in argv or "-G" in argv:
            continue  # not an ssh client, or a control/config query (ours or anyone's)
        match = next((m for m in map(DEEPDISH_RE.search, argv[1:]) if m), None)
        if not match:
            continue
        ancestors = []
        parent = table[pid][0]
        while parent in table and parent not in ancestors and len(ancestors) < 25:
            ancestors.append(parent)
            parent = table[parent][0]
        if me in ancestors:
            continue  # our own short-lived ssh/rsync calls
        in_editor = any(EDITOR_RE.search(table[a][2]) for a in ancestors)
        if tty not in ("?", "??", "-"):
            source = "VS Code terminal ssh" if in_editor else "terminal ssh"
        else:
            source = "VS Code Remote-SSH" if in_editor else "background ssh"
        found.append(SshProcess(pid, int(match.group(1)), source))
    return found


def status_all(cfg: dict) -> list[ServerStatus]:
    here = local_server()
    processes = ssh_processes()
    servers = [resolve_server(n, cfg) for n in cfg["servers"]]
    with ThreadPoolExecutor(max_workers=len(servers) or 1) as pool:
        alive = list(pool.map(master_alive, servers))
    statuses = []
    for server, has_socket in zip(servers, alive):
        links = []
        if here == server.n:
            links.append(Link("local", "dashboard is running on this server"))
        if has_socket:
            links.append(Link("socket", "shared SSH connection open"))
        for source in sorted({p.source for p in processes if p.n == server.n}):
            links.append(Link("process", source))
        statuses.append(ServerStatus(server, tuple(links)))
    return statuses


# --- connect / disconnect --------------------------------------------------------------------

def connect(server: Server, password: str | None = None, timeout: int = 45) -> tuple[bool, str]:
    """Open a shared SSH connection (ControlMaster) that later commands reuse without re-authenticating."""
    if master_alive(server):
        return True, f"Already connected to {server.name} (shared SSH connection)."
    env = os.environ.copy()
    opts = server.ssh_options(master=True)
    secret = None
    if password:
        secret = tempfile.NamedTemporaryFile("w", prefix="deepdish-", delete=False)
        os.chmod(secret.name, 0o600)
        secret.write(password + "\n")
        secret.close()
        env.update(
            SSH_ASKPASS=str(ASKPASS),
            SSH_ASKPASS_REQUIRE="force",
            DISPLAY=env.get("DISPLAY", ":0"),
            DEEPDISH_ASKPASS_FILE=secret.name,
        )
        opts += ["-o", "BatchMode=no", "-o", "NumberOfPasswordPrompts=1"]
    else:
        opts += ["-o", "BatchMode=yes"]
    try:
        # stdout/stderr go to a file, not a pipe: the backgrounded master inherits them.
        with tempfile.TemporaryFile("w+") as err:
            try:
                proc = subprocess.run(
                    ["ssh", *opts, server.target, "true"],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err, env=env, timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return False, f"Timed out reaching {server.hostname}. {_NETWORK_HINT}"
            err.seek(0)
            stderr = err.read()
    finally:
        if secret:
            os.unlink(secret.name)
    if proc.returncode != 0:
        return False, explain_ssh_error(server, stderr, password_given=bool(password))
    if master_alive(server):
        return True, f"Connected to {server.name}. The connection stays open for {server.control_persist}."
    return True, f"Connected to {server.name} (SSH key works; each command opens its own connection)."


def disconnect(server: Server) -> tuple[bool, str]:
    if not master_alive(server):
        return True, f"No shared connection to {server.name} is open."
    # If terminal sessions ride on the shared connection, only stop new sessions joining it.
    sessions = any(p.n == server.n for p in ssh_processes())
    proc = _run(["ssh", *server.ssh_options(), "-O", "stop" if sessions else "exit", server.target], timeout=10)
    if proc.returncode != 0:
        return False, proc.stderr.strip() or "ssh -O failed"
    note = " Terminal sessions using it keep running until you exit them." if sessions else ""
    return True, f"Closed the shared connection to {server.name}.{note}"


_NETWORK_HINT = "Off campus? Connect to the Northwestern VPN (GlobalProtect) first."


def explain_ssh_error(server: Server, stderr: str, password_given: bool = False) -> str:
    text = stderr.strip()
    if "Permission denied" in text:
        if password_given:
            return f"Authentication failed for {server.user}@{server.hostname}. Check your NetID and password."
        return (f"{server.hostname} did not accept an SSH key for {server.user}. Enter your NetID password and "
                "press Connect again, or install a key (see README, 'SSH keys').")
    if "Could not resolve hostname" in text:
        return f"Cannot resolve {server.hostname}. Check host_template in config/deepdish.json. {_NETWORK_HINT}"
    if any(s in text for s in ("timed out", "No route to host", "Connection refused", "Network is unreachable")):
        return f"Cannot reach {server.hostname}. {_NETWORK_HINT}"
    if "Host key verification failed" in text or "REMOTE HOST IDENTIFICATION HAS CHANGED" in text:
        return f"The host key of {server.hostname} changed. Check with IT, then remove the old key with ssh-keygen -R {server.hostname}."
    return text.splitlines()[-1] if text else "ssh failed without an error message."


# --- running commands ------------------------------------------------------------------------

def _run(argv: list[str], timeout: float, **kwargs) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, 124, "", f"timed out after {timeout:.0f}s")
    except OSError as e:
        return subprocess.CompletedProcess(argv, 127, "", str(e))


def remote_shell_path(path: str) -> str:
    """Quote a remote path for the shell, keeping a leading ~ expandable."""
    if path in ("~", "~/"):
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:].rstrip("/"))
    return shlex.quote(path.rstrip("/"))


def remote_rsync_path(path: str) -> str:
    """rsync resolves relative remote paths against the home directory."""
    return path[2:].rstrip("/") if path.startswith("~/") else path.rstrip("/")


def ssh_run(server: Server, command: str, timeout: float = 60) -> subprocess.CompletedProcess:
    return _run(["ssh", *server.ssh_options(), "-o", "BatchMode=yes", server.target, command], timeout)


class LocalTarget:
    """Run on this machine (your laptop, or deepdish itself when the dashboard runs there)."""

    remote = False

    def __init__(self, label: str = "this computer"):
        self.label = label

    def sh(self, command: str, timeout: float = 60) -> subprocess.CompletedProcess:
        return _run(["bash", "-c", command], timeout, cwd=ROOT)


class RemoteTarget:
    """Run inside the project folder on a deepdish server over ssh."""

    remote = True

    def __init__(self, server: Server, remote_dir: str):
        self.server = server
        self.remote_dir = remote_dir
        self.label = server.name

    def sh(self, command: str, timeout: float = 60) -> subprocess.CompletedProcess:
        return ssh_run(self.server, f"cd {remote_shell_path(self.remote_dir)} && {command}", timeout)


def remote_info(server: Server, remote_dir: str) -> dict[str, str]:
    """What is already set up on the server: project folder, code, .venv, Python and GPUs."""
    script = f"""
echo "host=$(hostname)"
echo "python3=$(python3 -V 2>&1)"
echo "gpus=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
if cd {remote_shell_path(remote_dir)} 2>/dev/null; then
  echo "folder=$(pwd)"
  [ -f run.py ] && echo code=yes || echo code=no
  [ -x .venv/bin/python ] && echo "venv=$(.venv/bin/python -V 2>&1)" || echo venv=no
else
  echo folder=missing
fi"""
    proc = ssh_run(server, script, timeout=30)
    if proc.returncode != 0:
        return {"error": explain_ssh_error(server, proc.stderr)}
    return dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)


def _rsync_shell(server: Server) -> str:
    return shlex.join(["ssh", *server.ssh_options(), "-o", "BatchMode=yes"])


def upload(server: Server, remote_dir: str, excludes: list[str], delete: bool = False) -> tuple[bool, str]:
    """Copy this project to the server. results/, logs/ and .venv/ there are never touched."""
    made = ssh_run(server, f"mkdir -p {remote_shell_path(remote_dir)}", timeout=30)
    if made.returncode != 0:
        return False, explain_ssh_error(server, made.stderr)
    argv = ["rsync", "-az", "--stats", "-e", _rsync_shell(server), *(f"--exclude={e}" for e in excludes)]
    if delete:
        argv.append("--delete")  # excluded paths are still protected on the server
    argv += [f"{ROOT}/", f"{server.target}:{remote_rsync_path(remote_dir)}/"]
    proc = _run(argv, timeout=900)
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    if "command not found" in proc.stderr or "rsync: not found" in proc.stderr:
        return _upload_tar(server, remote_dir, excludes)
    return False, (proc.stderr or proc.stdout).strip()


def _upload_tar(server: Server, remote_dir: str, excludes: list[str]) -> tuple[bool, str]:
    """Fallback when the server has no rsync: stream a tarball over ssh."""
    tar = ["tar", "-czf", "-", *(f"--exclude={e.rstrip('/')}" for e in excludes), "-C", str(ROOT), "."]
    ssh = ["ssh", *server.ssh_options(), "-o", "BatchMode=yes", server.target,
           f"mkdir -p {remote_shell_path(remote_dir)} && tar -xzf - -C {remote_shell_path(remote_dir)}"]
    env = {**os.environ, "COPYFILE_DISABLE": "1"}  # no macOS ._ files
    producer = subprocess.Popen(tar, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    proc = subprocess.run(ssh, stdin=producer.stdout, capture_output=True, text=True, timeout=900)
    producer.stdout.close()
    producer.wait()
    if proc.returncode != 0 or producer.returncode != 0:
        return False, (proc.stderr or producer.stderr.read().decode()).strip()
    return True, "Uploaded with tar (rsync is not installed on the server)."


def download(server: Server, remote_dir: str) -> tuple[bool, str]:
    """Copy results/ (and the job logs) from the server into this project."""
    base = f"{server.target}:{remote_rsync_path(remote_dir)}"
    shell = _rsync_shell(server)
    results = ROOT / "results"
    logs = ROOT / "logs" / server.name
    results.mkdir(exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    proc = _run(["rsync", "-az", "--stats", "-e", shell, f"{base}/results/", f"{results}/"], timeout=900)
    _run(["rsync", "-az", "-e", shell, f"{base}/logs/jobs/", f"{logs}/"], timeout=300)
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    if "No such file" in proc.stderr:
        return False, f"No results folder on {server.name} yet: run something first."
    return False, (proc.stderr or proc.stdout).strip()


# --- jobs (scripts/jobs.sh on the target) -----------------------------------------------------

def job_start(target, name: str, gpu: str, argv: list[str]) -> subprocess.CompletedProcess:
    return target.sh(f"bash scripts/jobs.sh start {shlex.quote(name)} {shlex.quote(gpu)} {shlex.join(argv)}")


def job_status(target) -> tuple[list[dict[str, str]], str]:
    proc = target.sh("bash scripts/jobs.sh status", timeout=30)
    if proc.returncode != 0:
        return [], (proc.stderr or proc.stdout).strip()
    rows = [dict(zip(JOB_FIELDS, line.split("\t"))) for line in proc.stdout.splitlines() if line.strip()]
    return rows, ""


def job_tail(target, name: str, lines: int = 200) -> str:
    proc = target.sh(f"bash scripts/jobs.sh tail {shlex.quote(name)} {int(lines)}", timeout=30)
    return proc.stdout if proc.returncode == 0 else (proc.stderr or proc.stdout)


def job_stop(target, name: str) -> subprocess.CompletedProcess:
    return target.sh(f"bash scripts/jobs.sh stop {shlex.quote(name)}", timeout=30)


def job_clear(target, name: str) -> subprocess.CompletedProcess:
    return target.sh(f"bash scripts/jobs.sh clear {shlex.quote(name)}", timeout=30)


def read_progress(target, run_name: str, max_lines: int = 200_000) -> list[dict]:
    """Per-episode records a training run has logged so far (results/pg/<run>/progress.jsonl)."""
    path = shlex.quote(f"results/pg/{run_name}/progress.jsonl")
    proc = target.sh(f"tail -n {int(max_lines)} {path} 2>/dev/null", timeout=60)
    records = []
    for line in proc.stdout.splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a line being written right now
    return records


def gpus(target) -> tuple[list[dict[str, str]], str]:
    """GPUs on the target with their current memory use and utilization."""
    proc = target.sh(
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits",
        timeout=30,
    )
    if proc.returncode != 0:
        return [], (proc.stderr or proc.stdout).strip() or "nvidia-smi is not available here"
    rows = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5:
            rows.append({"index": parts[0], "name": parts[1], "memory used (MiB)": parts[2],
                         "memory total (MiB)": parts[3], "utilization (%)": parts[4]})
    return rows, ""


# --- command line: python run.py deepdish ----------------------------------------------------

def add_arguments(parser) -> None:
    parser.add_argument("-n", "--server", type=int, metavar="N", help="server number (default: the default_server setting)")
    parser.add_argument("--connect", action="store_true",
                        help="open the shared SSH connection with your key, or with the NetID password typed at the prompt")
    parser.add_argument("--disconnect", action="store_true", help="close the shared SSH connection")


def main(args) -> int:
    """Show every setting with the layer it comes from, how ssh reaches the chosen server and each server's
    status; --connect/--disconnect manage the shared connection the dashboard and `ssh deepdishN` reuse."""
    cfg = load_config()
    sources = setting_sources()
    print("settings         (config/deepdish.json < config/deepdish.local.json < .env < environment)")
    for key, value in cfg.items():
        if key == "upload_excludes":
            continue
        text = ",".join(map(str, value)) if isinstance(value, list) else str(value)
        print(f"  {key:<16} {text or '(empty)':<42} {sources[key]}")
    n = args.server or cfg.get("default_server") or cfg["servers"][0]
    if n not in cfg["servers"]:
        print(f"deepdish{n} is not one of the servers: {cfg['servers']}", file=sys.stderr)
        return 2
    server = resolve_server(n, cfg)
    sharing = "on (~/.ssh/config)" if server.shared_control_path else f"dashboard only (ControlPath {server.control_path})"
    print(f"ssh              {server.target} -> {server.user}@{server.hostname}; key "
          f"{server.identity_file or '(ssh default)'}; connection sharing {sharing}")
    if args.disconnect:
        ok, message = disconnect(server)
        print(message)
        return 0 if ok else 1
    if args.connect:
        ok, message = connect(server)
        if not ok and "did not accept an SSH key" in message and sys.stdin.isatty():
            ok, message = connect(server, getpass.getpass(f"NetID password for {server.user}@{server.hostname}: "))
        print(message)
        return 0 if ok else 1
    print("status")
    for status in status_all(cfg):
        print(f"  {status.server.name:<14} {status.summary}")
    return 0
