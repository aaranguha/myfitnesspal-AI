#!/usr/bin/env python3
"""
Push-to-talk Diet Assistant — hold Option key, speak, release.

Runs as a background process. Hold the Option (Alt) key to record,
release to process. Uses the same GPT parsing + MFP logging as the
terminal assistant.

Usage:
    python hotkey_assistant.py
    python hotkey_assistant.py --whisper-model small
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from datetime import date, datetime

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from pynput import keyboard

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

# Load per-user config from ~/.diet_assistant/config.json (for friend installs)
try:
    from setup_wizard import apply_user_config, ensure_user_config
    _user_env = apply_user_config()
    for _k, _v in _user_env.items():
        if _v and _k not in os.environ:
            os.environ[_k] = _v
except ImportError:
    pass

# Import voice helpers
from voice_io import FOOD_CONTEXT_PROMPT, transcribe_audio

# Import assistant logic
from assistant import (
    MEALS,
    MEAL_NAMES_TO_IDX,
    build_diary_context,
    create_session,
    get_diary_details,
    get_recent_foods_cached,
    get_unlogged_meals,
    guess_meal_from_time,
    has_reasonable_recent_overlap,
    invalidate_recents_cache,
    log_foods,
    log_searched_food,
    match_foods_with_gpt,
    match_search_results_with_gpt,
    normalize_food_query,
    parse_user_message,
    pick_meal_bundle_results,
    pick_reasonable_search_result,
    quick_add_food,
    remove_food,
    resolve_meal_idx,
    search_foods,
    smart_reply,
    split_compound_food_for_search,
    split_food_descriptions,
)

from myfitnesspal import check_meals_logged, send_sms, send_sms_to, build_daily_summary, get_preset

USER_NAME = os.getenv("USER_NAME", "Aaran")

# Default reminder/summary config (can override via env or CLI args)
DEFAULT_REMINDER_TIMES = os.getenv("REMINDER_TIMES", "09:30,13:30,19:30")
DEFAULT_SUMMARY_TIME = os.getenv("SUMMARY_TIME", "23:00")
SUMMARY_SENT_FILE = os.path.join(SCRIPT_DIR, ".summary_sent")

# ── Audio config ─────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
BLOCKSIZE = 1024  # frames per read


# ── TTS ──────────────────────────────────────────────────────────────────

def speak(text):
    """Speak text using macOS say. Non-blocking, returns Popen process."""
    if not text:
        return None
    try:
        return subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None


def speak_synced(text, overlay_cmd, cancel_event=None):
    """Speak text and show overlay result in sync with actual audio playback.

    Uses say -o to pre-render audio, then plays with afplay so we know
    exactly when sound starts and stops. Stops immediately if cancel_event is set.
    """
    if not text:
        return
    import tempfile
    tmp = None
    proc = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".aiff", delete=False)
        tmp_path = tmp.name
        tmp.close()

        # Pre-render speech to file (fast, no audio output)
        subprocess.run(
            ["say", "-o", tmp_path, text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10,
        )

        if cancel_event and cancel_event.is_set():
            return

        # Play audio — show overlay the instant playback begins
        proc = subprocess.Popen(
            ["afplay", tmp_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        overlay_cmd(f"result:{text}")

        # Wait for audio, but bail immediately if cancelled
        while proc.poll() is None:
            if cancel_event and cancel_event.is_set():
                proc.terminate()
                return
            time.sleep(0.05)

        time.sleep(1.0)  # linger after voice ends
        if not (cancel_event and cancel_event.is_set()):
            overlay_cmd("hide")
    except Exception:
        if proc:
            proc.terminate()
        overlay_cmd("hide")
    finally:
        if tmp:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── Single-turn food logging ─────────────────────────────────────────────

def process_preset_log(session, preset_name, meal_idx):
    """Log all foods in a named preset. Returns list of logged food names."""
    matched_name, preset = get_preset(preset_name)
    if not matched_name or not preset:
        return []

    logged_names = []
    foods = preset.get("foods", [])
    print(f"  Logging preset \"{matched_name}\" ({len(foods)} foods)...")

    # Get search metadata once
    _, search_metadata = search_foods(session, foods[0]["search_name"] if foods else "", meal_idx)

    for item in foods:
        name = item["search_name"]
        # Try recents first
        all_foods, metadata = get_recent_foods_cached(session, meal_idx)
        if all_foods:
            match_results = match_foods_with_gpt(name, all_foods)
            if match_results and match_results[0].get("matches"):
                candidate = all_foods[match_results[0]["matches"][0]]
                if has_reasonable_recent_overlap(name, [candidate["name"]]):
                    if log_foods(session, [candidate], all_foods, meal_idx, metadata):
                        logged_names.append(candidate["name"])
                        invalidate_recents_cache(meal_idx)
                        continue

        # Fall back to MFP search
        results, s_meta = search_foods(session, name, meal_idx)
        if s_meta and not search_metadata:
            search_metadata = s_meta
        if results:
            best, confidence = match_search_results_with_gpt(name, results)
            if best and confidence in ("high", "medium"):
                if log_searched_food(session, best, meal_idx, search_metadata or s_meta):
                    logged_names.append(best["name"])
                    invalidate_recents_cache(meal_idx)

    return logged_names


def process_log(session, food_desc, meal_idx, prefer_custom=False):
    """Log foods in a single turn — auto-resolve disambiguation."""
    meal_name = MEALS[meal_idx]["name"]

    all_foods, metadata = get_recent_foods_cached(session, meal_idx)

    logged_names = []
    pending_foods = []        # from recents
    pending_searched = []     # from MFP search
    search_metadata = None

    # Step 1: Match against recents (skipped when user says "my X")
    not_found_descs = []
    if all_foods and not prefer_custom:
        print(f"  Checking {meal_name} recents...")
        match_results = match_foods_with_gpt(food_desc, all_foods)

        for item in match_results:
            matches = item.get("matches", [])
            desc = item.get("description", "")

            if len(matches) >= 1:
                # Auto-pick first match (no disambiguation in push-to-talk)
                candidate = all_foods[matches[0]]
                if has_reasonable_recent_overlap(desc, [candidate["name"]]):
                    pending_foods.append(candidate)
                else:
                    not_found_descs.append(desc)
            else:
                not_found_descs.append(desc)

        if not match_results:
            not_found_descs = split_food_descriptions(food_desc)
    else:
        not_found_descs = split_food_descriptions(food_desc)

    # Step 2: Search MFP for anything not in recents
    for desc in not_found_descs:
        parts = split_compound_food_for_search(desc)
        for part in parts:
            query = normalize_food_query(part)
            print(f"  Searching MFP for \"{query}\"...")
            results, s_meta = search_foods(session, query, meal_idx)
            if s_meta and not search_metadata:
                search_metadata = s_meta
            if results:
                # Always sort custom (unverified) foods to the top
                results = sorted(results, key=lambda r: (0 if not r.get("verified") else 1))
                print(f"  [debug] Top results: " + " | ".join(
                    f"{'[custom]' if not r.get('verified') else '[mfp]'} {r['name']}"
                    for r in results[:5]
                ))
                bundle = pick_meal_bundle_results(query, results)
                if bundle:
                    for m in bundle:
                        pending_searched.append(m)
                    continue
                best, confidence = match_search_results_with_gpt(query, results, prefer_custom=prefer_custom)
                if best and confidence in ("high", "medium"):
                    pending_searched.append(best)
                else:
                    fallback = pick_reasonable_search_result(query, results)
                    if fallback:
                        pending_searched.append(fallback)

    # Step 3: Log everything
    if pending_foods:
        if log_foods(session, pending_foods, all_foods, meal_idx, metadata):
            logged_names.extend(f["name"] for f in pending_foods)

    for food in pending_searched:
        if log_searched_food(session, food, meal_idx, search_metadata):
            logged_names.append(food["name"])

    if logged_names:
        invalidate_recents_cache(meal_idx)

    return logged_names


def process_remove(session, food_desc, mentioned_meal):
    """Remove food(s) from diary in a single turn."""
    from datetime import date
    import json
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    diary, _ = get_diary_details(session, date.today())

    candidates = []
    if mentioned_meal and mentioned_meal in MEAL_NAMES_TO_IDX:
        target_idx = MEAL_NAMES_TO_IDX[mentioned_meal]
        for f in diary.get(target_idx, []):
            f["meal_idx"] = target_idx
            candidates.append(f)
    else:
        for midx, foods in diary.items():
            for f in foods:
                f["meal_idx"] = midx
                candidates.append(f)

    if not candidates:
        return None, "Nothing logged to remove."

    if not food_desc or food_desc.lower() in ("everything", "all"):
        removed = 0
        for f in candidates:
            if f.get("entry_id") and remove_food(session, f["entry_id"]):
                removed += 1
        if removed:
            invalidate_recents_cache()
        return removed, f"Removed {removed} item(s)."

    # GPT match
    food_names_list = [f"{i}: {f['name']}" for i, f in enumerate(candidates)]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "The user wants to REMOVE food from their diary. "
                    "Match what they said to the numbered list of logged foods. "
                    "Return ONLY a JSON array of matching index numbers, e.g. [0, 2]. "
                    "Match loosely. Return [] if no match."
                ),
            },
            {
                "role": "user",
                "content": f"Remove: {food_desc}\n\nLogged foods:\n" + "\n".join(food_names_list),
            },
        ],
    )
    raw = response.choices[0].message.content.strip()
    try:
        indices = json.loads(raw)
        to_remove = [candidates[i] for i in indices if isinstance(i, int) and 0 <= i < len(candidates)]
    except Exception:
        to_remove = []

    if not to_remove:
        return 0, f"Couldn't find \"{food_desc}\" in your logged foods."

    removed = 0
    names = []
    for f in to_remove:
        if f.get("entry_id") and remove_food(session, f["entry_id"]):
            removed += 1
            names.append(f["name"])
    if removed:
        invalidate_recents_cache()
    return removed, f"Removed {', '.join(names)}."


def process_command(session, text):
    """Process a single voice command. Returns response string."""
    print(f"\n  You: {text}")

    parsed = parse_user_message(text)
    intent = parsed.get("intent", "chat")
    food_desc = parsed.get("foods")
    mentioned_meal = parsed.get("meal")

    # ── Quick add ──
    if intent == "quick_add":
        calories = parsed.get("calories", 0) or 0
        protein = parsed.get("protein", 0) or 0
        fat = parsed.get("fat", 0) or 0
        carbs = parsed.get("carbs", 0) or 0
        description = food_desc or "Quick Add"

        unlogged = get_unlogged_meals(session)
        meal_idx = resolve_meal_idx(mentioned_meal, unlogged)
        meal_name = MEALS[meal_idx]["name"]

        if quick_add_food(session, meal_idx, description, calories, protein, carbs, fat):
            return f"Quick added {description} to {meal_name}. {calories} calories."
        return "Something went wrong with the quick add."

    # ── Remove ──
    if intent == "remove":
        _, msg = process_remove(session, food_desc, mentioned_meal)
        return msg

    # ── Log meal preset ──
    preset_name = parsed.get("preset_name")
    if intent == "log_meal" and preset_name and not food_desc:
        unlogged = get_unlogged_meals(session)
        meal_idx = resolve_meal_idx(mentioned_meal, unlogged)
        meal_name = MEALS[meal_idx]["name"]
        logged = process_preset_log(session, preset_name, meal_idx)
        if logged:
            return f"Got it, logged your {preset_name} to {meal_name}."
        return f"Couldn't find a preset called \"{preset_name}\"."

    # ── Log ──
    if intent in ("log", "log_meal") and food_desc:
        unlogged = get_unlogged_meals(session)
        meal_idx = resolve_meal_idx(mentioned_meal, unlogged)
        meal_name = MEALS[meal_idx]["name"]

        # Detect "my X" in the raw text before GPT strips it
        prefer_custom = bool(re.search(r"\bmy\b", text, re.IGNORECASE))
        logged = process_log(session, food_desc, meal_idx, prefer_custom=prefer_custom)
        if logged:
            names = ", ".join(logged)
            return f"Got it, logged {names} to {meal_name}."
        return f"Couldn't find a match for {food_desc}."

    # ── Text summary on demand (only to user's number) ──
    text_lower = text.lower()
    if any(kw in text_lower for kw in ("text me", "send me", "text my")) and \
       any(kw in text_lower for kw in ("summary", "macros", "calories", "protein", "progress")):
        msg = build_daily_summary(session)
        if msg:
            my_number = os.getenv("MY_PHONE_NUMBER", "+15109469095")
            ok = send_sms_to(msg, my_number)
            if ok:
                return "Done, just texted you your summary."
            return "Couldn't send the text. Check your config."
        return "No diary data to summarize yet today."

    # ── Chat / questions ──
    if intent == "chat":
        context = build_diary_context(session)
        return smart_reply(text, context)

    # ── Skip ──
    if intent == "skip":
        return "OK, noted."

    # ── Defer ──
    if intent == "defer":
        return "Got it, I'll leave it for now."

    return "I didn't catch that. Try again?"


# ── Reminders ────────────────────────────────────────────────────────────

def parse_reminder_times(raw):
    """Parse comma-separated HH:MM values."""
    if not raw:
        return []
    parsed = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        try:
            datetime.strptime(t, "%H:%M")
            parsed.append(t)
        except ValueError:
            print(f"Skipping invalid reminder time: {t} (expected HH:MM)")
    return sorted(set(parsed))


def start_reminder_loop(session, reminder_times, stop_event):
    """Background reminders — speaks when meals are unlogged at scheduled times.

    reminder_times[0] → Breakfast, [1] → Lunch, [2] → Dinner.
    """
    if not reminder_times:
        return

    MEAL_TIME_MAP = {}
    for i, t in enumerate(reminder_times):
        if i < 3:
            MEAL_TIME_MAP[t] = i

    fired_today = set()

    def _check_meal_logged(meal_idx):
        try:
            logged = check_meals_logged(session, date.today())
            return logged.get(meal_idx, False)
        except Exception:
            return False

    def _get_remaining():
        try:
            _, summary = get_diary_details(session, date.today())
            remaining = summary.get("remaining", {})
            cal = remaining.get("calories", "?")
            protein = remaining.get("protein", "?")
            return cal, protein
        except Exception:
            return "?", "?"

    def _loop():
        while not stop_event.is_set():
            now = datetime.now()
            day_key = now.strftime("%Y-%m-%d")
            hhmm = now.strftime("%H:%M")

            # Reset fired set on new day
            if fired_today and not any(k.startswith(day_key) for k in fired_today):
                fired_today.clear()

            slot_key = f"{day_key} {hhmm}"
            if hhmm in MEAL_TIME_MAP and slot_key not in fired_today:
                meal_idx = MEAL_TIME_MAP[hhmm]
                meal_name = MEALS[meal_idx]["name"]
                if not _check_meal_logged(meal_idx):
                    fired_today.add(slot_key)
                    cal, protein = _get_remaining()
                    msg = f"Hey, wanna log {meal_name.lower()}? You still need {cal} cal and {protein}g protein today."
                    print(f"\n  Reminder: {msg}\n")
                    speak(msg)

            stop_event.wait(20)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


# ── Nightly summary ─────────────────────────────────────────────────────

def _was_summary_sent_today():
    try:
        with open(SUMMARY_SENT_FILE, "r") as f:
            last_date = f.read().strip()
        return last_date == date.today().isoformat()
    except (FileNotFoundError, ValueError):
        return False


def _mark_summary_sent():
    with open(SUMMARY_SENT_FILE, "w") as f:
        f.write(date.today().isoformat())


def start_nightly_summary_loop(session, summary_time, stop_event):
    """Background thread that texts a daily diet summary at the configured time."""
    if not summary_time:
        return

    try:
        h, m = summary_time.split(":")
        target_h, target_m = int(h), int(m)
        summary_time = f"{target_h:02d}:{target_m:02d}"
    except (ValueError, AttributeError):
        print(f"  [summary] Invalid time format: {summary_time}. Use HH:MM.")
        return

    def _send_summary():
        try:
            msg = build_daily_summary(session)
            if msg:
                ok = send_sms(msg)
                if ok:
                    _mark_summary_sent()
                    print(f"\n  Sent nightly diet summary!\n")
                    speak("Sent your nightly diet summary.")
                    return True
                else:
                    print(f"\n  Couldn't send summary. Check config.\n")
            else:
                print(f"\n  No diary data to summarize today.\n")
        except Exception as e:
            print(f"\n  Error building summary: {e}\n")
        return False

    def _loop():
        print(f"  [summary] Daily summary scheduled at {summary_time}")
        while not stop_event.is_set():
            now = datetime.now()
            past_target = (now.hour > target_h or
                           (now.hour == target_h and now.minute >= target_m))
            if past_target and not _was_summary_sent_today():
                _send_summary()
            stop_event.wait(30)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


# ── Push-to-talk engine ──────────────────────────────────────────────────

class PushToTalk:
    """Hold Option key to record, release to process."""

    def __init__(self, whisper_model="base", reminder_times=None, summary_time=None):
        print(f"\n{'═' * 50}")
        print(f"  Diet Assistant — Push-to-Talk")
        print(f"  Hold Option (Alt) key and speak")
        print(f"{'═' * 50}\n")

        # Launch floating overlay bar
        self._overlay = None
        try:
            overlay_path = os.path.join(SCRIPT_DIR, "overlay.py")
            if os.path.exists(overlay_path):
                self._overlay = subprocess.Popen(
                    [sys.executable, overlay_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                print("  Overlay bar ready.")
        except Exception as e:
            print(f"  (overlay unavailable: {e})")

        print("  Loading Whisper model...")
        from faster_whisper import WhisperModel
        self.model = WhisperModel(whisper_model, device="cpu", compute_type="int8")

        print("  Connecting to MyFitnessPal...")
        self.session = create_session()

        self._recording = False
        self._audio_buffer = []
        self._stream = None
        self._lock = threading.Lock()
        self._processing = False
        self._stop_event = threading.Event()

        # Start reading overlay stdout for typed commands
        if self._overlay:
            self._start_overlay_reader()

        # Start background reminders and nightly summary
        if reminder_times:
            print(f"  Reminders active: {', '.join(reminder_times)}")
            start_reminder_loop(self.session, reminder_times, self._stop_event)
        if summary_time:
            print(f"  Nightly summary text at: {summary_time}")
            start_nightly_summary_loop(self.session, summary_time, self._stop_event)

        self._typing_active = False  # True when overlay text field is open

        print("  Ready! Hold Option to speak, Option+Space to type.\n")

    def _overlay_cmd(self, cmd):
        """Send a command to the overlay subprocess."""
        if self._overlay and self._overlay.poll() is None:
            try:
                self._overlay.stdin.write(f"{cmd}\n".encode())
                self._overlay.stdin.flush()
            except Exception:
                pass

    def _start_overlay_reader(self):
        """Background thread that reads typed commands from the overlay's stdout."""
        def reader():
            try:
                for line in self._overlay.stdout:
                    text = line.decode().strip()
                    if text.startswith("typed:"):
                        typed_text = text[6:].strip()
                        if typed_text:
                            threading.Thread(
                                target=self._process_typed, args=(typed_text,), daemon=True
                            ).start()
                    elif text == "dismissed":
                        self._dismiss()
            except Exception:
                pass

        t = threading.Thread(target=reader, daemon=True)
        t.start()

    def _dismiss(self):
        """User clicked outside the overlay — cancel whatever is happening."""
        self._typing_active = False
        with self._lock:
            if self._recording:
                self._recording = False
                if self._stream:
                    try:
                        self._stream.stop()
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
                self._audio_buffer = []
                print("\r  (cancelled)                ")
            self._processing = False

    def _process_typed(self, text):
        """Process a typed command — no TTS, only overlay + terminal output."""
        self._typing_active = False
        with self._lock:
            if self._processing:
                return
            self._processing = True

        self._overlay_cmd("processing")

        try:
            response = process_command(self.session, text)
            print(f"  Assistant: {response}\n")
            self._overlay_cmd(f"result:{response}")
        except Exception as e:
            print(f"  Error: {e}")
            self._overlay_cmd("hide")

        self._processing = False

    def _start_typing_mode(self):
        """Switch to typing mode — show text field in overlay."""
        with self._lock:
            if self._processing:
                return
            # Cancel any recording in progress
            if self._recording:
                self._recording = False
                if self._stream:
                    self._stream.stop()
                    self._stream.close()
                    self._stream = None
        self._typing_active = True
        self._overlay_cmd("typing")
        print("  Type your command in the overlay bar...", flush=True)

    def _start_recording(self):
        """Start capturing audio."""
        with self._lock:
            if self._recording or self._processing:
                return
            self._recording = True
            self._audio_buffer = []

        self._overlay_cmd("recording")
        print("  🎤 Recording...", end="", flush=True)

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCKSIZE,
            callback=self._audio_callback,
        )
        self._stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if self._recording:
            self._audio_buffer.append(indata.copy())

    def _stop_recording(self):
        """Stop recording and process the audio."""
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._processing = True
            buffer = list(self._audio_buffer)

        # Stop the stream
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not buffer:
            print("\r  (no audio captured)        ")
            self._overlay_cmd("hide")
            self._processing = False
            return

        # Convert buffer to raw bytes
        audio_np = np.concatenate(buffer, axis=0)
        raw_audio = audio_np.tobytes()
        duration = len(raw_audio) / (SAMPLE_RATE * 2)

        if duration < 0.3:
            print("\r  (too short)                ")
            self._overlay_cmd("hide")
            self._processing = False
            return

        self._overlay_cmd("processing")
        print(f"\r  Transcribing... ({duration:.1f}s)    ", end="", flush=True)

        text = transcribe_audio(
            self.model, raw_audio, SAMPLE_RATE,
            initial_prompt=FOOD_CONTEXT_PROMPT,
        )

        if not text or len(text.strip()) < 2:
            print("\r  (couldn't transcribe)        ")
            self._overlay_cmd("hide")
            self._processing = False
            return

        print(f"\r                                    ")

        # Process the command
        try:
            response = process_command(self.session, text)
            print(f"  Assistant: {response}\n")
            # Pre-render speech, then play + show overlay perfectly in sync
            speak_synced(response, self._overlay_cmd)
        except Exception as e:
            error_msg = f"Error: {e}"
            print(f"  {error_msg}")
            self._overlay_cmd("hide")
            speak("Something went wrong.")

        self._processing = False

    def run(self):
        """Start listening for Option key (voice) and Option+Space (typing)."""
        option_held = False
        space_pressed_during_option = False
        record_timer = None

        def _delayed_record():
            """Start recording after a short delay (so Option+Space can cancel it)."""
            if option_held and not space_pressed_during_option and not self._typing_active:
                self._start_recording()

        def on_press(key):
            nonlocal option_held, space_pressed_during_option, record_timer
            # Ignore key events while typing in overlay
            if self._typing_active:
                return

            if key == keyboard.Key.alt or key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                if not option_held:
                    option_held = True
                    space_pressed_during_option = False
                    # Delay recording start by 200ms so Space can interrupt
                    record_timer = threading.Timer(0.2, _delayed_record)
                    record_timer.start()

            # Option + Space → typing mode
            if key == keyboard.Key.space and option_held and not space_pressed_during_option:
                space_pressed_during_option = True
                if record_timer:
                    record_timer.cancel()
                    record_timer = None
                self._start_typing_mode()

        def on_release(key):
            nonlocal option_held, space_pressed_during_option, record_timer
            # Ignore key events while typing in overlay
            if self._typing_active:
                return

            if key == keyboard.Key.alt or key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                if option_held:
                    option_held = False
                    if record_timer:
                        record_timer.cancel()
                        record_timer = None
                    # Only process voice if we didn't switch to typing mode
                    if not space_pressed_during_option:
                        threading.Thread(target=self._stop_recording, daemon=True).start()

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\n  Bye!")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    # Ensure user has completed first-time setup
    try:
        from setup_wizard import ensure_user_config
        if not ensure_user_config():
            print("Setup not completed. Run again when ready.")
            sys.exit(1)
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Push-to-talk Diet Assistant")
    parser.add_argument("--whisper-model", type=str, default="tiny",
                        help="Whisper model size: tiny, base, small, medium (default: tiny)")
    parser.add_argument("--reminder-times", type=str, default=DEFAULT_REMINDER_TIMES,
                        help="Comma-separated HH:MM reminders (default: 09:30,13:30,19:30)")
    parser.add_argument("--no-reminders", action="store_true", help="Disable timed reminders")
    parser.add_argument("--summary-time", type=str, default=DEFAULT_SUMMARY_TIME,
                        help="HH:MM time to text daily diet summary (default: 23:00)")
    parser.add_argument("--no-summary", action="store_true", help="Disable nightly summary text")
    args = parser.parse_args()

    reminder_times = [] if args.no_reminders else parse_reminder_times(args.reminder_times)
    summary_time = None if args.no_summary else args.summary_time

    ptt = PushToTalk(
        whisper_model=args.whisper_model,
        reminder_times=reminder_times,
        summary_time=summary_time,
    )
    ptt.run()


if __name__ == "__main__":
    main()
