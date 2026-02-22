#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────────────
#  Diet Assistant — One-liner installer
#
#  curl -fsSL https://raw.githubusercontent.com/aaranguha/myfitnesspal-AI/main/install.sh | bash
# ─────────────────────────────────────────────────────

REPO="https://github.com/aaranguha/myfitnesspal-AI.git"
BRANCH="main"
INSTALL_DIR="$HOME/diet-assistant"
APP_NAME="DietAssistant"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║       Diet Assistant — Installer         ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Check Python ──
if ! command -v python3 &>/dev/null; then
  echo "❌  Python 3 is required but not found."
  echo "   Install it from https://www.python.org/downloads/"
  exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]); then
  echo "❌  Python 3.9+ is required (found $PY_VERSION)"
  exit 1
fi
echo "✓  Python $PY_VERSION"

# ── Check git ──
if ! command -v git &>/dev/null; then
  echo "❌  git is required but not found."
  echo "   Run: xcode-select --install"
  exit 1
fi
echo "✓  git"

# ── Clone or update repo ──
if [ -d "$INSTALL_DIR/.git" ]; then
  echo ""
  echo "Updating existing install at $INSTALL_DIR..."
  cd "$INSTALL_DIR"
  git pull --ff-only origin "$BRANCH" 2>/dev/null || true
else
  echo ""
  echo "Downloading Diet Assistant..."
  git clone -b "$BRANCH" "$REPO" "$INSTALL_DIR"
  cd "$INSTALL_DIR"
fi
echo "✓  Source code ready"

# ── Create venv ──
if [ ! -d "venv" ]; then
  echo ""
  echo "Creating virtual environment..."
  python3 -m venv venv
fi
source venv/bin/activate
echo "✓  Virtual environment"

# ── Install dependencies ──
echo ""
echo "Installing dependencies (this may take a minute)..."
pip install -q --upgrade pip 2>/dev/null
pip install -q -r requirements.txt
echo "✓  Dependencies installed"

# ── Build DietAssistant.app with correct paths ──
echo ""
echo "Building DietAssistant.app..."
APP_DIR="$INSTALL_DIR/$APP_NAME.app"
MACOS_DIR="$APP_DIR/Contents/MacOS"
mkdir -p "$MACOS_DIR"

cat > "$APP_DIR/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>DietAssistant</string>
    <key>CFBundleIdentifier</key>
    <string>com.dietassistant.hotkey</string>
    <key>CFBundleName</key>
    <string>DietAssistant</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>LSBackgroundOnly</key>
    <true/>
</dict>
</plist>
PLIST

cat > "$MACOS_DIR/DietAssistant" <<LAUNCHER
#!/bin/bash
cd "$INSTALL_DIR"
source venv/bin/activate
exec python hotkey_assistant.py
LAUNCHER
chmod +x "$MACOS_DIR/DietAssistant"
echo "✓  DietAssistant.app built"

# ── Copy to ~/Applications for Spotlight ──
USER_APPS="$HOME/Applications"
mkdir -p "$USER_APPS"
cp -R "$APP_DIR" "$USER_APPS/$APP_NAME.app"
echo "✓  Copied to ~/Applications (searchable via Spotlight)"

# ── Run setup wizard ──
echo ""
echo "════════════════════════════════════════════"
echo "  First-time setup"
echo "════════════════════════════════════════════"
echo ""

export DIET_ASSISTANT_SETUP_MODE=cli
python3 setup_wizard.py </dev/tty

# ── Done! ──
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║         Installation complete!           ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  How to use:"
echo ""
echo "  Terminal mode:"
echo "    cd $INSTALL_DIR && ./start.command"
echo ""
echo "  Background hotkey mode:"
echo "    Open Spotlight (Cmd+Space) → type 'DietAssistant' → Enter"
echo "    Hold Option to speak, Option+Space to type"
echo ""
echo "  Auto-start on login:"
echo "    System Settings → General → Login Items"
echo "    Add: ~/Applications/DietAssistant.app"
echo ""
echo "  Grant Accessibility permission (required for hotkey):"
echo "    System Settings → Privacy & Security → Accessibility"
echo "    Add: ~/Applications/DietAssistant.app"
echo ""
