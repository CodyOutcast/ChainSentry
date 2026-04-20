#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d dist ]]; then
  echo "Presentation bundle not found at frontend/dist. Run 'npm run build:presentation' first."
  exit 1
fi

port="${1:-4173}"

if command -v python3 >/dev/null 2>&1; then
  python_cmd="python3"
elif command -v python >/dev/null 2>&1; then
  python_cmd="python"
else
  echo "Python is required to serve the prebuilt frontend bundle."
  exit 1
fi

exec "$python_cmd" -m http.server "$port" -d dist