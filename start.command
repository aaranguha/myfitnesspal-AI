#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  Diet Assistant"
echo "========================================"
echo ""

# ── Auto-create venv if missing ──
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
  echo "  Done."
fi

source venv/bin/activate

# ── Auto-install dependencies ──
if [ -f "requirements.txt" ]; then
  echo "Checking dependencies..."
  pip install -q -r requirements.txt
  echo "  Dependencies ready."
fi

# ── First-time setup if needed ──
export DIET_ASSISTANT_SETUP_MODE="${DIET_ASSISTANT_SETUP_MODE:-cli}"
python3 -c "
from setup_wizard import ensure_user_config
import sys
if not ensure_user_config():
    print('Setup not completed. Run start.command again when ready.')
    sys.exit(1)
"
if [ $? -ne 0 ]; then
  echo ""
  echo "Press Enter to close..."
  read
  exit 1
fi

# ── Mode choice ──
echo ""
echo "How would you like to run Diet Assistant?"
echo ""
echo "  1) Terminal mode  — type commands in this window"
echo "  2) Hotkey mode    — hold Option key to speak (runs in background)"
echo ""
read -p "Choose [1 or 2]: " mode

case "$mode" in
  2)
    echo ""
    echo "Starting hotkey mode (hold Option key to speak)..."
    echo "  Tip: Add DietAssistant.app to Login Items for auto-start on boot."
    echo ""
    python3 hotkey_assistant.py "$@"
    ;;
  *)
    echo ""
    echo "Starting terminal mode..."
    echo ""
    python3 launch_assistant.py "$@"
    ;;
esac
