# Local Distribution (No Backend)

This keeps your existing assistant code as-is and adds a first-run setup flow.

## What was added

- `setup_wizard.py`: first-run setup (GUI with CLI fallback)
- `launch_assistant.py`: applies per-user config and starts `assistant.py`
- `start.command`: double-click start for macOS
- `scripts/make_release_zip.sh`: creates a shareable zip

Your core files remain unchanged:

- `assistant.py`
- `myfitnesspal.py`
- `voice_io.py`

## Friend Setup Flow

1. Download and unzip your release folder.
2. Double-click `start.command`.
3. First run asks for:
- Name
- OpenAI API key
- MyFitnessPal guided cookie setup (recommended)
- Optional reminder/SMS settings
4. Setup saves local config to `~/.diet_assistant/config.json`.
5. Assistant starts. Future launches are one click.

## Build A Shareable Zip

From repo root:

```bash
./scripts/make_release_zip.sh
```

Output zip appears in `release/`.

## Notes

- This is local-only per user. No centralized backend needed.
- Each user keeps their own credentials on their own machine.
- `~/.diet_assistant/config.json` is set to user-read/write permissions where possible.
