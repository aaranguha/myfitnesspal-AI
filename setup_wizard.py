#!/usr/bin/env python3
"""First-run setup wizard for Diet Assistant local installs.

Stores user-specific secrets/config in ~/.diet_assistant/config.json so the
project's shared .env file can stay untouched.
"""

from __future__ import annotations

import json
import os
import subprocess
import stat
import sys
import threading
import time
from pathlib import Path

import requests

APP_DIR = Path.home() / ".diet_assistant"
CONFIG_FILE = APP_DIR / "config.json"
MFP_DIARY_URL = "https://www.myfitnesspal.com/food/diary"
OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
MFP_LOGIN_URL = "https://www.myfitnesspal.com/account/login"

DEFAULTS = {
    "USER_NAME": "Friend",
    "REMINDER_TIMES": "09:30,13:30,19:30",
    "SUMMARY_TIME": "23:00",
    "MY_PHONE_NUMBER": "",
    "SMS_PHONE_NUMBERS": "",
    "OPENAI_API_KEY": "",
    "MFP_COOKIE": "",
}


def load_user_config() -> dict:
    if not CONFIG_FILE.exists():
        return dict(DEFAULTS)
    try:
        raw = json.loads(CONFIG_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULTS)

    merged = dict(DEFAULTS)
    for key in DEFAULTS:
        value = raw.get(key)
        if value is not None:
            merged[key] = str(value)
    return merged


def _secure_file(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def save_user_config(config: dict) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))
    _secure_file(CONFIG_FILE)


def validate_mfp_cookie(cookie: str) -> tuple[bool, str]:
    cookie = (cookie or "").strip()
    if not cookie:
        return False, "MFP cookie is required."

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Cookie": cookie,
    }
    try:
        resp = requests.get(MFP_DIARY_URL, headers=headers, allow_redirects=False, timeout=12)
    except Exception as exc:
        return False, f"Could not reach MyFitnessPal: {exc}"

    if resp.status_code in (301, 302, 303):
        location = resp.headers.get("Location", "")
        if "login" in location or "account" in location:
            return False, "MFP session looks expired/invalid (redirected to login)."

    if resp.status_code != 200:
        return False, f"Unexpected MFP response: {resp.status_code}"

    return True, "MFP session works."


def validate_openai_key(api_key: str) -> tuple[bool, str]:
    api_key = (api_key or "").strip()
    if not api_key:
        return False, "OpenAI API key is required."
    if not api_key.startswith("sk-"):
        return False, "OpenAI API key should start with 'sk-'."

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(OPENAI_MODELS_URL, headers=headers, timeout=12)
    except Exception as exc:
        return False, f"Could not reach OpenAI: {exc}"

    if resp.status_code == 200:
        return True, "OpenAI API key works."
    if resp.status_code in (401, 403):
        return False, "OpenAI API key was rejected."
    return False, f"Unexpected OpenAI response: {resp.status_code}"


def _build_config_from_values(values: dict) -> tuple[dict, str | None]:
    config = dict(DEFAULTS)
    for key in config:
        config[key] = (values.get(key, "") or "").strip()

    if not config["USER_NAME"]:
        return config, "Name is required."
    if not config["OPENAI_API_KEY"]:
        return config, "OpenAI API key is required."
    if not config["MFP_COOKIE"]:
        return config, "MyFitnessPal connection is required."
    return config, None


def capture_mfp_cookie_manually() -> tuple[str | None, str]:
    print("\nMyFitnessPal cookie setup:")
    print(f"1) Open: {MFP_LOGIN_URL}")
    print("2) Log in to your account")
    print(f"3) Go to: {MFP_DIARY_URL}")
    print("4) Open DevTools: Cmd+Option+I (Mac) or Ctrl+Shift+I (Windows)")
    print("5) Go to Network tab and reload the page")
    print("6) In the Filter box, type: diary")
    print("7) Scroll up in the request list and click the diary row")
    print("8) On the right, go to Headers")
    print("9) Scroll to Request Headers and find cookie")
    print("10) Copy the cookie value (long string on the right)")
    print("\nCopy the cookie value to your clipboard first.")
    input("Press Enter when ready. I will read it from clipboard...\n")

    cookie = ""
    if sys.platform == "darwin":
        try:
            cookie = subprocess.run(
                ["pbpaste"], capture_output=True, text=True, check=False
            ).stdout.strip()
        except Exception:
            cookie = ""

    # Fallback manual entry if clipboard read fails/empty
    if not cookie:
        cookie = input("Clipboard empty. Paste cookie here manually, then Enter:\n> ").strip()
    if not cookie:
        return None, "No cookie entered."

    print(f"Received cookie ({len(cookie)} chars).")

    ok, msg = validate_mfp_cookie_with_ui(cookie, timeout_seconds=30)
    if not ok:
        return None, msg
    return cookie, "Connected to MyFitnessPal (manual cookie)."


def _clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def validate_mfp_cookie_with_ui(cookie: str, timeout_seconds: int = 30) -> tuple[bool, str]:
    """Validate cookie with a simple terminal UI and hard timeout."""
    result = {"done": False, "ok": False, "msg": ""}

    def worker():
        ok, msg = validate_mfp_cookie(cookie)
        result["done"] = True
        result["ok"] = ok
        result["msg"] = msg

    _clear_terminal()
    print("+--------------------------------------+")
    print("|          Diet Assistant              |")
    print("|         Validating Cookie...         |")
    print("+--------------------------------------+\n")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    spinner = ["|", "/", "-", "\\"]
    start = time.time()
    i = 0
    while not result["done"]:
        elapsed = int(time.time() - start)
        if elapsed >= timeout_seconds:
            print("\r" + " " * 60 + "\r", end="", flush=True)
            return False, "something went wrong talk to aaran"
        print(f"\rValidating... {spinner[i % len(spinner)]}  ({elapsed}s)", end="", flush=True)
        time.sleep(0.12)
        i += 1

    print("\r" + " " * 60 + "\r", end="", flush=True)
    return result["ok"], result["msg"]


def run_setup_wizard() -> bool:
    # Some macOS + Python builds crash in Tk before exceptions can be caught.
    # Use CLI mode by default in that environment.
    if sys.platform == "darwin" and sys.version_info < (3, 10):
        return run_setup_cli()

    try:
        import tkinter as tk
        from tkinter import messagebox, simpledialog
    except Exception:
        return run_setup_cli()

    existing = load_user_config()

    root = tk.Tk()
    root.title("Diet Assistant Setup")
    root.geometry("640x480")

    frame = tk.Frame(root, padx=16, pady=16)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text="Diet Assistant First-Time Setup", font=("Helvetica", 16, "bold")).pack(anchor="w")
    tk.Label(
        frame,
        text="Enter your credentials once. They are saved locally on this computer.",
        fg="#333333",
    ).pack(anchor="w", pady=(4, 12))

    fields = [
        ("USER_NAME", "Your name", False),
        ("OPENAI_API_KEY", "OpenAI API key", True),
        ("MFP_COOKIE", "MyFitnessPal cookie (paste from browser DevTools)", True),
        ("REMINDER_TIMES", "Reminder times (HH:MM, comma-separated)", False),
        ("SUMMARY_TIME", "Nightly summary time (HH:MM)", False),
        ("MY_PHONE_NUMBER", "Your phone number (+1XXXXXXXXXX)", False),
        ("SMS_PHONE_NUMBERS", "Trainer/accountability partner numbers (comma-separated, optional)", False),
    ]

    widgets = {}
    for key, label, hidden in fields:
        tk.Label(frame, text=label).pack(anchor="w", pady=(8, 2))
        entry = tk.Entry(frame, show="*" if hidden else "")
        entry.insert(0, existing.get(key, ""))
        entry.pack(fill="x")
        widgets[key] = entry

    status_var = tk.StringVar(value="")
    tk.Label(frame, textvariable=status_var, fg="#005a9c", justify="left", anchor="w").pack(fill="x", pady=(12, 8))

    def collect_values() -> dict:
        return {k: w.get() for k, w in widgets.items()}

    def connect_mfp_browser() -> None:
        status_var.set("Connecting to MyFitnessPal...")
        root.update_idletasks()
        messagebox.showinfo(
            "MyFitnessPal Cookie Steps",
            "1) Log in at myfitnesspal.com/account/login\n"
            "2) Open your diary page\n"
            "3) DevTools -> Network -> reload page\n"
            "4) In the Filter box type: diary\n"
            "5) Open a diary row -> Headers (right panel)\n"
            "6) Scroll to Request Headers -> cookie\n"
            "7) Copy the long cookie value and paste in next prompt",
        )
        cookie = simpledialog.askstring(
            "MyFitnessPal Cookie",
            "Paste full Cookie header value:",
            show="*",
            parent=root,
        )
        cookie = (cookie or "").strip()
        if not cookie:
            status_var.set("No cookie entered.")
            messagebox.showerror("MFP Connect Failed", "No cookie entered.")
            return
        ok, message = validate_mfp_cookie(cookie)
        if not ok:
            status_var.set(message)
            messagebox.showerror("MFP Connect Failed", message)
            return

        widgets["MFP_COOKIE"].delete(0, tk.END)
        widgets["MFP_COOKIE"].insert(0, cookie)
        status_var.set("MyFitnessPal connected.")
        messagebox.showinfo("Connected", message)

    def save_only() -> None:
        cfg, err = _build_config_from_values(collect_values())
        if err:
            messagebox.showerror("Setup Error", err)
            return
        save_user_config(cfg)
        status_var.set(f"Saved config to {CONFIG_FILE}")
        messagebox.showinfo("Saved", "Saved successfully.")

    def test_connections() -> None:
        cfg, err = _build_config_from_values(collect_values())
        if err:
            messagebox.showerror("Setup Error", err)
            return

        status_var.set("Testing OpenAI key...")
        root.update_idletasks()
        ok_openai, msg_openai = validate_openai_key(cfg["OPENAI_API_KEY"])
        if not ok_openai:
            status_var.set(msg_openai)
            messagebox.showerror("OpenAI Test Failed", msg_openai)
            return

        status_var.set("Testing MyFitnessPal connection...")
        root.update_idletasks()
        ok_mfp, msg_mfp = validate_mfp_cookie(cfg["MFP_COOKIE"])
        if not ok_mfp:
            status_var.set(msg_mfp)
            messagebox.showerror("MFP Test Failed", msg_mfp)
            return

        status_var.set("Both checks passed.")
        messagebox.showinfo("Success", "OpenAI + MyFitnessPal checks passed.")

    def save_and_close() -> None:
        cfg, err = _build_config_from_values(collect_values())
        if err:
            messagebox.showerror("Setup Error", err)
            return
        save_user_config(cfg)
        root.destroy()

    btn_row = tk.Frame(frame)
    btn_row.pack(fill="x", pady=(10, 0))

    tk.Button(btn_row, text="Connect MyFitnessPal", command=connect_mfp_browser).pack(side="left")
    tk.Button(btn_row, text="Test Credentials", command=test_connections).pack(side="left")
    tk.Button(btn_row, text="Save", command=save_only).pack(side="left", padx=(8, 0))
    tk.Button(btn_row, text="Save and Start", command=save_and_close).pack(side="right")

    root.mainloop()
    return CONFIG_FILE.exists()


def run_setup_cli() -> bool:
    existing = load_user_config()
    print("\nDiet Assistant Setup")
    print("This saves config locally in ~/.diet_assistant/config.json\n")

    def ask(key: str, label: str, secret: bool = False) -> str:
        current = existing.get(key, "")
        prompt = label
        if current:
            prompt += " [press Enter to keep current]"
        prompt += ": "

        if secret:
            import getpass

            val = getpass.getpass(prompt)
        else:
            val = input(prompt)

        val = val.strip()
        return val if val else current

    def choose_mfp_cookie() -> str:
        current = existing.get("MFP_COOKIE", "").strip()
        while True:
            print("\nMyFitnessPal setup:")
            print("1) Show step-by-step cookie instructions (recommended)")
            print("2) Paste cookie directly")
            if current:
                print("3) Keep current saved cookie")

            raw = input("Choose option [1]: ").strip()
            choice = raw or "1"

             # If user pastes a full cookie string at the option prompt, accept it.
            if ("=" in raw and ";" in raw) and len(raw) > 80:
                print("\nDetected cookie pasted at option prompt. Using it directly...")
                ok, msg = validate_mfp_cookie_with_ui(raw, timeout_seconds=30)
                if ok:
                    return raw
                print(msg)
                print("Let's try again.")
                continue

            if choice == "3" and current:
                return current

            cookie, message = capture_mfp_cookie_manually()
            print(message)
            if cookie:
                return cookie

            print("Let's try again.")

    values = {
        "USER_NAME": ask("USER_NAME", "Your name"),
        "OPENAI_API_KEY": ask("OPENAI_API_KEY", "Paste OpenAI API key here (visible input)"),
        "MFP_COOKIE": choose_mfp_cookie(),
        "REMINDER_TIMES": ask("REMINDER_TIMES", "Reminder times (HH:MM, comma-separated)"),
        "SUMMARY_TIME": ask("SUMMARY_TIME", "Nightly summary time (HH:MM)"),
        "MY_PHONE_NUMBER": ask("MY_PHONE_NUMBER", "Your phone number (+1XXXXXXXXXX)"),
        "SMS_PHONE_NUMBERS": ask("SMS_PHONE_NUMBERS",
            "Trainer/accountability partner numbers (comma-separated, optional)\n"
            "  Nightly summary will be sent to you AND these numbers"),
    }

    cfg, err = _build_config_from_values(values)
    if err:
        print(f"\nSetup error: {err}\n")
        return False

    print("\nTesting credentials...")
    ok_openai, msg_openai = validate_openai_key(cfg["OPENAI_API_KEY"])
    print(f"OpenAI: {msg_openai}")
    if not ok_openai:
        return False

    ok_mfp, msg_mfp = validate_mfp_cookie(cfg["MFP_COOKIE"])
    print(f"MFP: {msg_mfp}")
    if not ok_mfp:
        return False

    save_user_config(cfg)
    print(f"\nSaved: {CONFIG_FILE}\n")
    return True


def apply_user_config(env: dict | None = None) -> dict:
    merged = dict(os.environ if env is None else env)
    cfg = load_user_config()
    for key, value in cfg.items():
        if value:
            merged[key] = value
    return merged


def ensure_user_config() -> bool:
    cfg = load_user_config()
    if cfg.get("OPENAI_API_KEY") and cfg.get("MFP_COOKIE") and cfg.get("USER_NAME"):
        return True
    return run_setup_wizard()


def main() -> int:
    mode = (os.getenv("DIET_ASSISTANT_SETUP_MODE", "auto") or "auto").strip().lower()
    if mode == "cli":
        ok = run_setup_cli()
    elif mode == "gui":
        ok = run_setup_wizard()
    else:
        ok = run_setup_wizard()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
