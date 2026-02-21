#!/bin/bash
# Install the Diet Assistant hotkey service as a macOS Launch Agent.
# This makes it start automatically on login and restart if it crashes.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_NAME="com.dietassistant.hotkey.plist"
PLIST_SRC="$SCRIPT_DIR/$PLIST_NAME"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"

echo "=== Diet Assistant Hotkey — Install ==="
echo ""

# Check plist exists
if [ ! -f "$PLIST_SRC" ]; then
    echo "ERROR: $PLIST_SRC not found."
    exit 1
fi

# Unload existing service if loaded
if launchctl list | grep -q "com.dietassistant.hotkey"; then
    echo "  Stopping existing service..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

# Copy plist to LaunchAgents
echo "  Copying plist to ~/Library/LaunchAgents/..."
mkdir -p "$HOME/Library/LaunchAgents"
cp "$PLIST_SRC" "$PLIST_DST"

# Load the service
echo "  Loading service..."
launchctl load "$PLIST_DST"

# Verify
sleep 1
if launchctl list | grep -q "com.dietassistant.hotkey"; then
    PID=$(launchctl list | grep "com.dietassistant.hotkey" | awk '{print $1}')
    echo ""
    echo "=== Installed! ==="
    echo "  PID: $PID"
    echo "  Logs: ~/Library/Logs/diet-assistant-hotkey.log"
    echo ""
    echo "  Hold Option key and speak to test."
    echo ""
    echo "  Manage with:"
    echo "    Stop:   launchctl unload ~/Library/LaunchAgents/$PLIST_NAME"
    echo "    Start:  launchctl load ~/Library/LaunchAgents/$PLIST_NAME"
    echo "    Logs:   tail -f ~/Library/Logs/diet-assistant-hotkey.log"
else
    echo ""
    echo "  WARNING: Service loaded but not running yet."
    echo "  Check logs: tail ~/Library/Logs/diet-assistant-hotkey.log"
fi
