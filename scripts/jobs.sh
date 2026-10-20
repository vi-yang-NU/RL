#!/usr/bin/env bash
# Background job runner shared by the dashboard and the terminal; works locally and on deepdish.
#
#   scripts/jobs.sh start <name> <gpu-index|cpu> <command...>   launch a command in the background
#   scripts/jobs.sh status                                      one tab-separated line per job
#   scripts/jobs.sh tail <name> [lines]                         last lines of a job's log
#   scripts/jobs.sh stop <name>                                 terminate a job and its children
#   scripts/jobs.sh clear <name>                                forget a finished job
#
# Jobs survive the ssh session ending. Each one gets logs/jobs/<name>.{log,pid,meta,exit}.
# A GPU job sees exactly one card through CUDA_VISIBLE_DEVICES (the handout requires a single
# GPU on deepdish). The project's .venv is used when it exists.
set -euo pipefail

cd "$(dirname "$0")/.."
JOBS_DIR="logs/jobs"
mkdir -p "$JOBS_DIR"

die() { echo "jobs.sh: $*" >&2; exit 1; }
usage() { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

is_running() {
  local pid
  pid=$(cat "$JOBS_DIR/$1.pid" 2>/dev/null) || return 1
  [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null
}

meta() { sed -n "s/^$2=//p" "$JOBS_DIR/$1.meta" 2>/dev/null | head -n 1; }

# Terminate a process and all of its descendants, children first.
kill_tree() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$child"; done
  kill -TERM "$1" 2>/dev/null || true
}

cmd_start() {
  [[ $# -ge 3 ]] || usage
  local name=$1 gpu=$2
  shift 2
  [[ $name =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid job name '$name' (letters, digits, . _ - only)"
  is_running "$name" && die "job '$name' is already running (pid $(cat "$JOBS_DIR/$name.pid"))"

  if [[ $gpu == cpu ]]; then
    export CUDA_VISIBLE_DEVICES=""
  elif [[ $gpu =~ ^[0-9]+$ ]]; then
    export CUDA_VISIBLE_DEVICES=$gpu
  else
    die "gpu must be one card index (e.g. 0) or 'cpu', got '$gpu'"
  fi

  if [[ -x .venv/bin/python ]]; then
    export VIRTUAL_ENV="$PWD/.venv" PATH="$PWD/.venv/bin:$PATH"
  elif [[ $1 == python ]] && ! command -v python >/dev/null; then
    set -- python3 "${@:2}"
  fi

  local log="$JOBS_DIR/$name.log"
  rm -f "$JOBS_DIR/$name.exit" "$JOBS_DIR/$name.stopped"
  {
    echo "started=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "host=$(hostname)"
    echo "gpu=$gpu"
    echo "command=$*"
  } > "$JOBS_DIR/$name.meta"

  local launcher=(nohup)
  command -v setsid >/dev/null && launcher=(setsid nohup)
  JOB_EXIT_FILE="$JOBS_DIR/$name.exit" PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
    "${launcher[@]}" bash -c '"$@"; echo $? > "$JOB_EXIT_FILE"' job "$@" > "$log" 2>&1 < /dev/null &
  echo $! > "$JOBS_DIR/$name.pid"
  echo "started $name (pid $!, gpu $gpu): $*"
}

cmd_status() {
  local meta_file name state code
  shopt -s nullglob
  for meta_file in "$JOBS_DIR"/*.meta; do
    name=$(basename "$meta_file" .meta)
    code=""
    if is_running "$name"; then
      state=running
    elif [[ -f $JOBS_DIR/$name.stopped ]]; then
      state=stopped
    elif [[ -f $JOBS_DIR/$name.exit ]]; then
      code=$(cat "$JOBS_DIR/$name.exit")
      [[ $code == 0 ]] && state=finished || state=failed
    else
      state=lost  # killed from outside or the machine restarted
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$state" "$code" "$(cat "$JOBS_DIR/$name.pid" 2>/dev/null)" \
      "$(meta "$name" gpu)" "$(meta "$name" started)" "$(meta "$name" host)" "$(meta "$name" command)"
  done
}

cmd_tail() {
  [[ $# -ge 1 ]] || usage
  [[ -f $JOBS_DIR/$1.log ]] || die "no log for job '$1'"
  tail -n "${2:-100}" "$JOBS_DIR/$1.log"
}

cmd_stop() {
  [[ $# -ge 1 ]] || usage
  is_running "$1" || die "job '$1' is not running"
  local pid
  pid=$(cat "$JOBS_DIR/$1.pid")
  touch "$JOBS_DIR/$1.stopped"
  # With setsid (Linux) the job leads its own process group; otherwise walk its process tree.
  kill -TERM -- "-$pid" 2>/dev/null || kill_tree "$pid"
  echo "stopped $1 (pid $pid)"
}

cmd_clear() {
  [[ $# -ge 1 ]] || usage
  is_running "$1" && die "job '$1' is still running; stop it first"
  rm -f "$JOBS_DIR/$1".{log,pid,meta,exit,stopped}
  echo "cleared $1"
}

[[ $# -ge 1 ]] || usage
command=$1
shift
case $command in
  start) cmd_start "$@" ;;
  status) cmd_status ;;
  tail) cmd_tail "$@" ;;
  stop) cmd_stop "$@" ;;
  clear) cmd_clear "$@" ;;
  *) usage ;;
esac
