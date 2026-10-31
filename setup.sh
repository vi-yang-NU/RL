#!/usr/bin/env bash
# One-time setup after cloning.
#
#   ./setup.sh            on your computer: Python environment + dashboard, then SSH setup for deepdish
#   ./setup.sh --server   on deepdish: Python environment for running the experiments (no dashboard)
#
# Options:
#   --python PATH   interpreter for the virtual environment (default: newest python3.x >= 3.9 found)
#   --venv-dir DIR  where to create the virtual environment; .venv becomes a symlink to it. Default: .venv,
#                   except on a server whose home is on NFS (deepdish), where /tmp/$USER/<project>-venv on the
#                   local disk is used: importing PyTorch from the NFS home takes minutes per process there.
#   --no-ssh        skip the SSH steps
#   -y, --yes       accept the default answer to every question
#
# Environment:
#   TORCH_INDEX_URL   install torch from this index first, e.g. https://download.pytorch.org/whl/cu121
#                     (use it if the default torch wheel does not match the server's CUDA driver)
set -euo pipefail
cd "$(dirname "$0")"

MODE=local
PYTHON=""
VENV_DIR=""
DO_SSH=1
ASSUME_YES=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --server) MODE=server ;;
    --python) PYTHON=$2; shift ;;
    --venv-dir) VENV_DIR=$2; shift ;;
    --no-ssh) DO_SSH=0 ;;
    -y|--yes) ASSUME_YES=1 ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see ./setup.sh --help)" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }

# ask "question" y|n -> success for yes. Without a terminal (or with --yes) the default is taken.
ask() {
  local default=$2 reply
  if (( ASSUME_YES )) || [[ ! -t 0 ]]; then [[ $default == y ]]; return; fi
  read -r -p "    $1 [$([[ $default == y ]] && echo Y/n || echo y/N)] " reply
  [[ ${reply:-$default} =~ ^[Yy] ]]
}

find_python() {
  local candidate
  for candidate in python3.12 python3.11 python3.10 python3.13 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
      command -v "$candidate"
      return
    fi
  done
  return 1
}

# --- Python environment ------------------------------------------------------------------------
say "Python environment (.venv)"
if [[ -z $PYTHON ]]; then
  PYTHON=$(find_python) || {
    echo "No Python >= 3.9 found. Install one (or on deepdish try 'module load python' / conda)," >&2
    echo "then rerun with --python /path/to/python3." >&2
    exit 1
  }
fi
note "using $("$PYTHON" -V 2>&1) at $PYTHON"
fs_type=$(stat -f -c %T . 2>/dev/null || df -P -T . 2>/dev/null | awk 'NR == 2 { print $2 }')
if [[ -z $VENV_DIR && $MODE == server && $fs_type == nfs* && -d /tmp ]]; then
  VENV_DIR=/tmp/$USER/$(basename "$PWD")-venv
  note "the project is on NFS ($fs_type): creating the virtual environment on the local disk at $VENV_DIR"
  note "(importing PyTorch from an NFS home takes minutes per process; .venv will be a symlink)"
fi
VENV_DIR=${VENV_DIR:-.venv}
if [[ $VENV_DIR != .venv ]]; then
  if [[ -d .venv && ! -L .venv ]]; then
    if ask "Replace the existing .venv directory with a symlink to $VENV_DIR?" y; then
      rm -rf .venv
    else
      VENV_DIR=.venv
    fi
  fi
  if [[ $VENV_DIR != .venv ]]; then
    mkdir -p "$VENV_DIR"
    ln -sfn "$VENV_DIR" .venv
  fi
fi
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON" -m venv "$VENV_DIR"
fi
VENV_PY=.venv/bin/python
"$VENV_PY" -m pip install --quiet --upgrade pip
TORCH_SPEC=torch
if [[ -z ${TORCH_INDEX_URL:-} ]] && command -v nvidia-smi >/dev/null 2>&1; then
  # deepdish's GTX 1080s are compute capability 6.1; current default CUDA wheels need >= 7.0.
  # The CUDA 12.6 builds (PyTorch <= 2.14) are the newest that still run on them.
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -n | head -n 1)
  if [[ -n $cc ]] && awk -v c="$cc" 'BEGIN { exit !(c < 7.0) }'; then
    TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
    TORCH_SPEC="torch<2.15"
    note "GPU compute capability $cc: installing a CUDA 12.6 build of torch (the newest that supports it)"
  fi
fi
if [[ -n ${TORCH_INDEX_URL:-} ]]; then
  note "installing $TORCH_SPEC from $TORCH_INDEX_URL"
  "$VENV_PY" -m pip install --quiet "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL"
fi
note "installing requirements.txt (torch, gymnasium, ale-py, ...; the first time takes a few minutes)"
"$VENV_PY" -m pip install --quiet -r requirements.txt
if [[ $MODE == local ]]; then
  note "installing requirements-ui.txt (dashboard)"
  "$VENV_PY" -m pip install --quiet -r requirements-ui.txt
fi
chmod +x setup.sh run.py scripts/*.sh ui/askpass.sh
mkdir -p results logs

# --- SSH to deepdish (your computer only) ---------------------------------------------------------

# save_setting key value: remember a deepdish setting in .env (DEEPDISH_KEY=value; git ignores .env)
save_setting() {
  "$VENV_PY" -c 'import sys; from ui.deepdish import save_settings; save_settings(**{sys.argv[1]: sys.argv[2]})' "$1" "$2"
  note "saved $1 '$2' to .env (DEEPDISH_$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]'))"
}

# ssh_block aliases <netid> | ssh_block sharing: ~/.ssh/config text built from the deepdish settings
ssh_block() {
  "$VENV_PY" - "$@" <<'EOF'
import sys
from ui.deepdish import load_config
cfg = load_config()
alias, host = cfg["alias_template"], cfg["host_template"]
if sys.argv[1] == "aliases":
    print("\n# >>> deepdish (added by RL assignment setup.sh) >>>")
    for n in cfg["servers"]:
        print(f"Host {alias.format(n=n)}\n    HostName {host.format(n=n)}\n    User {sys.argv[2]}")
    print("# <<< deepdish <<<")
else:
    print("\n# >>> deepdish connection sharing (added by RL assignment setup.sh) >>>")
    print(f"Host {alias.format(n='?')} {host.format(n='?')}")
    print(f"    ControlMaster auto\n    ControlPath {cfg['control_path']}\n    ControlPersist {cfg['control_persist']}")
    print("    ServerAliveInterval 30\n# <<< deepdish connection sharing <<<")
EOF
}

setup_ssh() {
  local alias_host netid cfg=~/.ssh/config
  alias_host=$(ssh -G deepdish1 2>/dev/null | awk '/^hostname /{print $2}')
  mkdir -p ~/.ssh && chmod 700 ~/.ssh

  netid=$("$VENV_PY" -c 'from ui.deepdish import load_config; print(load_config().get("user", ""))')
  if [[ $alias_host != "deepdish1" && -n $alias_host ]]; then
    note "found the deepdish1-5 aliases in ~/.ssh/config (deepdish1 -> $alias_host)"
    if [[ -z $netid ]]; then  # keep the NetID the aliases use in .env, so every tool sees the same one
      netid=$(ssh -G deepdish1 2>/dev/null | awk '/^user /{print $2}')
      [[ -n $netid ]] && save_setting user "$netid"
    fi
  else
    if [[ -z $netid && -t 0 && $ASSUME_YES == 0 ]]; then
      read -r -p "    Your Northwestern NetID (for ssh): " netid
    fi
    netid=${netid:-$USER}
    save_setting user "$netid"
    if ask "Add 'deepdish1'...'deepdish5' shortcuts to ~/.ssh/config (so 'ssh deepdish3' works)?" y; then
      [[ -f $cfg ]] && cp "$cfg" "$cfg.bak.$(date +%Y%m%d%H%M%S)"
      ssh_block aliases "$netid" >> "$cfg"
      chmod 600 "$cfg"
      note "added (the previous ~/.ssh/config was backed up next to it)"
    fi
  fi

  if [[ -z $(ssh -G deepdish1 2>/dev/null | awk '/^controlpath /{print $2}') ]]; then
    note "connection sharing lets the dashboard reuse SSH sessions you open from a terminal or VS Code"
    if ask "Enable SSH connection sharing for deepdish in ~/.ssh/config?" y; then
      [[ -f $cfg ]] && cp "$cfg" "$cfg.bak.$(date +%Y%m%d%H%M%S)"
      ssh_block sharing >> "$cfg"
      chmod 600 "$cfg"
      if [[ -n $(ssh -G deepdish1 2>/dev/null | awk '/^controlpath /{print $2}') ]]; then
        note "enabled"
      else
        warn "an earlier 'Host' block in ~/.ssh/config overrides ControlPath; move the new block above it"
      fi
    fi
  else
    note "SSH connection sharing is already enabled"
  fi

  if [[ ! -f ~/.ssh/id_ed25519 && ! -f ~/.ssh/id_rsa ]]; then
    if ask "Create an SSH key (~/.ssh/id_ed25519) so you don't type your password each time?" y; then
      ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "$USER@$(hostname -s) RL assignment"
    fi
  fi
  if ask "Test the connection to deepdish1 now (needs campus network or the VPN)?" y; then
    if ssh -o BatchMode=yes -o ConnectTimeout=8 deepdish1 true 2>/dev/null; then
      note "deepdish1 accepts your SSH key"
    elif ask "Key login failed. Copy your public key to deepdish (asks for your NetID password once)?" n; then
      ssh-copy-id deepdish1 || warn "ssh-copy-id failed; you can still log in with your password from the dashboard"
    else
      note "skipped; the dashboard can log in with your NetID password instead"
    fi
  fi
}

if [[ $MODE == local && $DO_SSH == 1 ]]; then
  say "SSH access to deepdish"
  setup_ssh
fi

# --- check ----------------------------------------------------------------------------------
say "Checking the installation"
"$VENV_PY" run.py check || warn "some checks failed (see above)"

say "Done"
if [[ $MODE == local ]]; then
  note "source .venv/bin/activate"
  note "python run.py ui        # dashboard at http://localhost:8501"
else
  note "source .venv/bin/activate"
  note "python run.py -h        # or launch jobs from the dashboard"
fi
