#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
  source venv/bin/activate
fi

export DIET_ASSISTANT_SETUP_MODE="${DIET_ASSISTANT_SETUP_MODE:-cli}"

python3 launch_assistant.py "$@"
