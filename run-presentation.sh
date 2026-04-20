#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PORT="${CHAIN_SENTRY_BACKEND_PORT:-8000}"
FRONTEND_PORT="${CHAIN_SENTRY_FRONTEND_PORT:-4173}"
BACKEND_HOST="${CHAIN_SENTRY_BACKEND_HOST:-127.0.0.1}"
FRONTEND_HOST="${CHAIN_SENTRY_FRONTEND_HOST:-127.0.0.1}"
PYTHON_BIN=""
LOG_DIR="$ROOT_DIR/.presentation"
VENV_DIR="$ROOT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
PRESENTATION_REQUIREMENTS="$ROOT_DIR/backend/requirements-presentation.txt"
INSTALL_STAMP="$VENV_DIR/.presentation-requirements.stamp"

cleanup() {
  local exit_code=$?
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]] && kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
  exit "$exit_code"
}

trap cleanup INT TERM EXIT

run_with_sudo() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
    return
  fi
  "$@"
}

try_install_python() {
  echo "Python 3 was not found. Attempting to install it automatically..."

  if command -v brew >/dev/null 2>&1; then
    brew install python
    return
  fi

  if command -v apt-get >/dev/null 2>&1; then
    run_with_sudo apt-get update
    run_with_sudo apt-get install -y python3 python3-venv python3-pip
    return
  fi

  if command -v dnf >/dev/null 2>&1; then
    run_with_sudo dnf install -y python3 python3-pip
    return
  fi

  if command -v yum >/dev/null 2>&1; then
    run_with_sudo yum install -y python3 python3-pip
    return
  fi

  echo "Automatic Python installation is not supported on this machine."
  echo "Install Python 3 manually, then rerun: bash run-presentation.sh"
  exit 1
}

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
    return
  fi

  try_install_python

  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
    return
  fi

  echo "Python 3 is still unavailable after the automatic installation attempt."
  exit 1
}

ensure_frontend_bundle() {
  if [[ ! -f "$ROOT_DIR/frontend/dist/index.html" ]]; then
    echo "Prebuilt frontend bundle is missing at frontend/dist/index.html."
    echo "Rebuild it on the source machine with: cd frontend && npm run build:presentation"
    exit 1
  fi
}

ensure_venv() {
  if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Creating local Python environment..."
    "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1 || true
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
}

backend_modules_ready() {
  "$VENV_PYTHON" - <<'PY' >/dev/null 2>&1
import importlib

required = ("fastapi", "uvicorn", "numpy", "networkx", "sklearn", "torch", "yaml")
for module_name in required:
    importlib.import_module(module_name)
PY
}

ensure_backend_requirements() {
  if [[ ! -f "$INSTALL_STAMP" || "$PRESENTATION_REQUIREMENTS" -nt "$INSTALL_STAMP" ]] || ! backend_modules_ready; then
    echo "Installing backend presentation dependencies..."
    "$VENV_PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 || true
    "$VENV_PYTHON" -m pip install --upgrade pip wheel
    "$VENV_PYTHON" -m pip install -r "$PRESENTATION_REQUIREMENTS"
    touch "$INSTALL_STAMP"
  fi
}

wait_for_backend() {
  local attempts=40
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if "$VENV_PYTHON" - <<PY
from urllib.request import urlopen

try:
    with urlopen("http://${BACKEND_HOST}:${BACKEND_PORT}/health", timeout=1) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 1
  done

  echo "Backend did not become ready in time. Check $LOG_DIR/backend.log"
  return 1
}

mkdir -p "$LOG_DIR"

find_python
ensure_frontend_bundle
ensure_venv
ensure_backend_requirements

echo "Starting backend on http://${BACKEND_HOST}:${BACKEND_PORT} ..."
(
  cd "$ROOT_DIR"
  PYTHONPATH=backend "$VENV_PYTHON" -m uvicorn app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT"
) >"$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!

wait_for_backend

echo "Starting prebuilt frontend on http://${FRONTEND_HOST}:${FRONTEND_PORT} ..."
(
  cd "$ROOT_DIR/frontend"
  "$PYTHON_BIN" -m http.server "$FRONTEND_PORT" -b "$FRONTEND_HOST" -d dist
) >"$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

cat <<EOF

ChainSentry presentation mode is running.

Frontend: http://${FRONTEND_HOST}:${FRONTEND_PORT}
Backend:  http://${BACKEND_HOST}:${BACKEND_PORT}

Logs:
  $LOG_DIR/frontend.log
  $LOG_DIR/backend.log

Press Ctrl+C to stop both servers.
EOF

wait "$BACKEND_PID" "$FRONTEND_PID"
