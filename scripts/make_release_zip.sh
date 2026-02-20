#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/release"
APP_DIR="$OUT_DIR/diet-assistant"

rm -rf "$APP_DIR"
mkdir -p "$APP_DIR"

cp "$ROOT_DIR/assistant.py" "$APP_DIR/"
cp "$ROOT_DIR/myfitnesspal.py" "$APP_DIR/"
cp "$ROOT_DIR/voice_io.py" "$APP_DIR/"
cp "$ROOT_DIR/setup_wizard.py" "$APP_DIR/"
cp "$ROOT_DIR/launch_assistant.py" "$APP_DIR/"
cp "$ROOT_DIR/start.command" "$APP_DIR/"

# Optional starter files
if [ -f "$ROOT_DIR/meals.json" ]; then
  cp "$ROOT_DIR/meals.json" "$APP_DIR/"
fi

cat > "$APP_DIR/README_FIRST.txt" <<'TXT'
Diet Assistant Quick Start

1) Double-click start.command
2) On first run, complete setup (name + OpenAI key)
3) Choose guided MyFitnessPal cookie setup and follow the steps
4) The app saves your config locally to ~/.diet_assistant/config.json
5) Future runs: just double-click start.command

Notes:
- Python 3.11+ is required
- If you use voice input, install audio dependencies first
TXT

cd "$OUT_DIR"
ZIP_NAME="diet-assistant-$(date +%Y%m%d-%H%M%S).zip"
rm -f "$ZIP_NAME"
zip -r "$ZIP_NAME" "diet-assistant" >/dev/null

echo "Created: $OUT_DIR/$ZIP_NAME"
