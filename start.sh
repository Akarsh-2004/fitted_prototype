#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
VENV_DIR="$BACKEND_DIR/.venv"
BACKEND_LOG="${TMPDIR:-/tmp}/vestir-backend.log"
FRONTEND_LOG="${TMPDIR:-/tmp}/vestir-frontend.log"

echo "=========================================================="
echo "Vestir AI - Pipeline Startup Orchestrator"
echo "=========================================================="

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

is_windows_bash() {
  case "$(uname -s 2>/dev/null || echo unknown)" in
    MINGW*|MSYS*|CYGWIN*) return 0 ;;
    *) return 1 ;;
  esac
}

python_cmd() {
  if command_exists py; then
    echo "py -3"
  elif command_exists python3; then
    echo "python3"
  elif command_exists python; then
    echo "python"
  else
    echo ""
  fi
}

# --- Python Virtual Environment Setup ---
echo "[1/3] Preparing Python FastAPI backend..."
PYTHON_CMD="$(python_cmd)"
if [ -z "$PYTHON_CMD" ]; then
  echo "Python was not found. Install Python 3.11+ and make sure it is on PATH."
  exit 1
fi

if [ -f "$VENV_DIR/Scripts/python.exe" ]; then
  VENV_PYTHON="$VENV_DIR/Scripts/python.exe"
elif ! is_windows_bash && [ -f "$VENV_DIR/bin/python" ]; then
  VENV_PYTHON="$VENV_DIR/bin/python"
else
  echo "Virtual environment missing, incomplete, or from a different OS. Recreating backend/.venv..."
  $PYTHON_CMD -m venv --clear "$VENV_DIR"

  if [ -f "$VENV_DIR/Scripts/python.exe" ]; then
    VENV_PYTHON="$VENV_DIR/Scripts/python.exe"
  elif ! is_windows_bash && [ -f "$VENV_DIR/bin/python" ]; then
    VENV_PYTHON="$VENV_DIR/bin/python"
  else
    echo "Could not create a usable Python virtual environment under backend/.venv."
    exit 1
  fi
fi

echo "Installing Python backend requirements..."
"$VENV_PYTHON" -m pip install -r "$BACKEND_DIR/requirements.txt"

# --- Frontend Node modules check ---
echo "[2/3] Preparing React frontend..."
if ! command_exists npm; then
  echo "npm was not found. Install Node.js 20+ and make sure npm is on PATH."
  exit 1
fi

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "Installing frontend node packages..."
  (cd "$FRONTEND_DIR" && npm install)
else
  echo "Node packages are already installed."
fi

# --- Cleanup Handler for Ctrl+C ---
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  echo "Shutting down services..."
  if [ -n "${BACKEND_PID:-}" ]; then
    echo "-> Stopping FastAPI backend (PID $BACKEND_PID)..."
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "${FRONTEND_PID:-}" ]; then
    echo "-> Stopping Vite frontend (PID $FRONTEND_PID)..."
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
  echo "All services stopped."
  exit 0
}
trap cleanup EXIT INT TERM

# --- Launch Services ---
echo "[3/3] Launching services concurrently..."

# Port check and cleanup for Unix-like shells. On native Windows Git Bash,
# close existing terminals using ports 8011/5188 if the script reports a bind error.
if command -v lsof >/dev/null 2>&1; then
  for pid in $(lsof -ti :8011 2>/dev/null || true); do
    echo "Port 8011 is occupied by PID $pid. Releasing port..."
    kill -9 "$pid" >/dev/null 2>&1 || true
  done
  for pid in $(lsof -ti :5188 2>/dev/null || true); do
    echo "Port 5188 is occupied by PID $pid. Releasing port..."
    kill -9 "$pid" >/dev/null 2>&1 || true
  done
  sleep 0.5
fi

echo "Starting FastAPI backend at http://127.0.0.1:8011..."
(cd "$SCRIPT_DIR" && "$VENV_PYTHON" -m uvicorn pipeline.main:app --host 127.0.0.1 --port 8011 --reload) >"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

echo "Starting Vite frontend at http://127.0.0.1:5188..."
(cd "$FRONTEND_DIR" && npm run dev -- --host 127.0.0.1 --port 5188) >"$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

echo ""
echo "=========================================================="
echo "Vestir AI is launching."
echo "Frontend Client URL:  http://127.0.0.1:5188"
echo "Backend API URL:      http://127.0.0.1:8011"
echo "Backend logs:         $BACKEND_LOG"
echo "Frontend logs:        $FRONTEND_LOG"
echo "=========================================================="
echo "Press Ctrl+C to terminate both servers."
echo ""

wait "$BACKEND_PID" "$FRONTEND_PID"
