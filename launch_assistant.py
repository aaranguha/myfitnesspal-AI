#!/usr/bin/env python3
"""Launcher for Diet Assistant local distribution.

- Runs first-time setup wizard when needed.
- Applies per-user config from ~/.diet_assistant/config.json to env.
- Starts assistant.py unchanged.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setup_wizard import apply_user_config, ensure_user_config


def main() -> int:
    if not ensure_user_config():
        print("Setup was not completed. Exiting.")
        return 1

    script_dir = Path(__file__).resolve().parent
    assistant_path = script_dir / "assistant.py"
    if not assistant_path.exists():
        print(f"Could not find assistant.py at {assistant_path}")
        return 1

    env = apply_user_config()
    cmd = [sys.executable, str(assistant_path), *sys.argv[1:]]
    return subprocess.call(cmd, cwd=str(script_dir), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
