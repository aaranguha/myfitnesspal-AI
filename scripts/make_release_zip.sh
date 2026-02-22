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
cp "$ROOT_DIR/hotkey_assistant.py" "$APP_DIR/"
cp "$ROOT_DIR/start.command" "$APP_DIR/"
cp "$ROOT_DIR/requirements.txt" "$APP_DIR/"

# Copy the .app bundle (preserving structure)
if [ -d "$ROOT_DIR/DietAssistant.app" ]; then
  cp -R "$ROOT_DIR/DietAssistant.app" "$APP_DIR/"
fi

# Optional starter files
if [ -f "$ROOT_DIR/meals.json" ]; then
  cp "$ROOT_DIR/meals.json" "$APP_DIR/"
fi

# Make start.command executable
chmod +x "$APP_DIR/start.command"

cat > "$APP_DIR/README_FIRST.txt" <<'TXT'
Diet Assistant Quick Start

1) Double-click start.command
2) On first run it will:
   - Create a virtual environment
   - Install dependencies
   - Run the setup wizard (name, OpenAI key, MFP cookie, phone numbers)
3) Choose your mode:
   - Terminal mode: type commands in the terminal
   - Hotkey mode: hold Option key to speak (push-to-talk)

Phone Numbers:
- YOUR number: receives on-demand "text me my macros" summaries
- Trainer numbers: receive the nightly diet summary along with you

Auto-Start on Boot (Hotkey Mode):
- Open System Settings → General → Login Items
- Add DietAssistant.app from this folder

Notes:
- Python 3.10+ is required (3.11+ recommended)
- macOS only (uses iMessage for texting, say for TTS)
- Grant Accessibility permission to DietAssistant.app for hotkey mode
TXT

cd "$OUT_DIR"
ZIP_NAME="diet-assistant-$(date +%Y%m%d-%H%M%S).zip"
rm -f "$ZIP_NAME"
zip -r "$ZIP_NAME" "diet-assistant" >/dev/null

echo "Created: $OUT_DIR/$ZIP_NAME"
