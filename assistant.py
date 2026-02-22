#!/usr/bin/env python3
"""
Diet Assistant — Conversational food logger for MyFitnessPal.

Talks to you about meals, understands natural language ("I had a protein bar
and chai"), matches against your MFP recent foods, and logs them.

Modes:
  python assistant.py              — Terminal chat mode (typing)
  python hotkey_assistant.py       — Push-to-talk mode (hold Option key)

Requires: OPENAI_API_KEY in .env
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import argparse
from datetime import date, datetime, timedelta

from dotenv import load_dotenv
from openai import OpenAI

# Import MFP functions from the existing script
from myfitnesspal import (
    MEALS,
    create_session,
    check_meals_logged,
    get_diary_details,
    get_recent_foods,
    search_foods,
    log_foods,
    log_searched_food,
    remove_food,
    send_sms,
    build_daily_summary,
    load_presets,
    save_presets,
    get_preset_names,
    get_preset,
    quick_add_food,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

USER_NAME = os.getenv("USER_NAME", "Aaran")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MEAL_NAMES_TO_IDX = {info["name"].lower(): idx for idx, info in MEALS.items()}

# ── Recent foods cache (avoid re-fetching from MFP on every log) ──────────

_recents_cache = {}  # {meal_idx: {"foods": [...], "metadata": ..., "ts": float}}
RECENTS_TTL = 120    # seconds — cache recents for 2 minutes

def get_recent_foods_cached(session, meal_idx):
    """Wrapper around get_recent_foods with a 2-minute in-memory cache."""
    now = time.time()
    entry = _recents_cache.get(meal_idx)
    if entry and (now - entry["ts"]) < RECENTS_TTL:
        return entry["foods"], entry["metadata"]
    foods, metadata = get_recent_foods(session, meal_idx)
    _recents_cache[meal_idx] = {"foods": foods, "metadata": metadata, "ts": now}
    return foods, metadata

def invalidate_recents_cache(meal_idx=None):
    """Clear cache after logging/removing so next fetch is fresh."""
    if meal_idx is not None:
        _recents_cache.pop(meal_idx, None)
    else:
        _recents_cache.clear()


# ── Voice + reminders ──────────────────────────────────────────────────────

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


class Speaker:
    """Simple TTS wrapper that uses local OS commands."""
    def __init__(self, enabled=False, voice_name=None, default_volume=65):
        self.enabled = enabled
        self.voice_name = voice_name
        self.default_volume = max(0, min(100, int(default_volume)))
        self.backend = None

        if not enabled:
            return

        if shutil.which("say"):
            self.backend = "say"
        elif shutil.which("espeak"):
            self.backend = "espeak"
        else:
            print("Voice mode enabled, but no TTS backend found (`say`/`espeak`).")
            self.enabled = False

    @staticmethod
    def _set_volume(level=100):
        """Set macOS system volume (0-100)."""
        try:
            subprocess.run(["osascript", "-e", f"set volume output volume {level}"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _build_cmd(self, text):
        cmd = [self.backend]
        if self.voice_name:
            cmd += ["-v", self.voice_name]
        cmd.append(text)
        return cmd

    def speak(self, text):
        """Non-blocking TTS — fire and forget."""
        if not self.enabled or not text:
            return
        try:
            subprocess.Popen(self._build_cmd(text),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def speak_and_wait(self, text, volume=None):
        """Blocking TTS — waits until speech finishes."""
        if not self.enabled or not text:
            return
        try:
            if volume is not None:
                self._set_volume(volume)
            subprocess.run(self._build_cmd(text),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.3)  # Let audio output buffer drain
            if volume is not None:
                self._set_volume(self.default_volume)
        except Exception:
            pass


class AssistantSpeechInterceptor:
    """Intercept stdout and speak lines that contain `Assistant:`.

    If input_provider is set, only speaks when the last input was voice
    (Option key push-to-talk).
    """
    def __init__(self, wrapped, speaker, input_provider=None):
        self.wrapped = wrapped
        self.speaker = speaker
        self.input_provider = input_provider
        self._buffer = ""

    def _should_speak(self):
        """Only speak if last input was from voice (Option key)."""
        if self.input_provider is None:
            return True
        return getattr(self.input_provider, 'last_was_voice', False)

    def write(self, data):
        self.wrapped.write(data)
        self._buffer += data

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if "Assistant:" in line and self._should_speak():
                spoken = line.split("Assistant:", 1)[1].strip()
                # Keep spoken output concise: don't read parenthetical text.
                spoken = re.sub(r"\([^)]*\)", "", spoken)
                spoken = re.sub(r"\s+", " ", spoken).strip()
                self.speaker.speak_and_wait(spoken, volume=50)

    def flush(self):
        self.wrapped.flush()


def start_reminder_loop(session, reminder_times, stop_event, snooze_state,
                        io_lock=None, input_provider=None, speaker=None):
    """Background reminders — only fires for the specific meal at each time.

    reminder_times: list of "HH:MM" strings (e.g. ["09:30", "13:30", "19:30"])
    snooze_state: dict shared with main thread. Keys are meal_idx (int),
                  values are datetime of when to re-check after a user deferral.
    io_lock: optional threading.Lock shared with VoiceInputProvider to prevent
             reminders from speaking while the mic is recording.
    input_provider: optional AlwaysOnInputProvider — used to open response windows.
    speaker: optional Speaker — used to speak reminders at lower volume.

    Mapping: reminder_times[0] → Breakfast(0), [1] → Lunch(1), [2] → Dinner(2).
    Extra times beyond 3 remind for any unlogged meal (legacy behavior).
    """
    if not reminder_times:
        return

    # Map each reminder time to its meal index
    MEAL_TIME_MAP = {}
    for i, t in enumerate(reminder_times):
        if i < 3:
            MEAL_TIME_MAP[t] = i  # 0=Breakfast, 1=Lunch, 2=Dinner
        # Extra times (if any) won't get a specific meal assignment

    fired_today = set()

    def _check_meal_logged(meal_idx):
        """Check if a specific meal has been logged today."""
        try:
            logged = check_meals_logged(session, date.today())
            return logged.get(meal_idx, False)
        except Exception:
            return False

    def _get_remaining():
        """Get remaining calories and protein for today."""
        try:
            _, summary = get_diary_details(session, date.today())
            remaining = summary.get("remaining", {})
            cal = remaining.get("calories", "?")
            protein = remaining.get("protein", "?")
            return cal, protein
        except Exception:
            return "?", "?"

    def _say(msg):
        """Print a reminder message."""
        print(msg)

    def _loop():
        while not stop_event.is_set():
            now = datetime.now()
            day_key = now.strftime("%Y-%m-%d")
            hhmm = now.strftime("%H:%M")

            # Reset fired set on new day
            if fired_today and not any(k.startswith(day_key) for k in fired_today):
                fired_today.clear()
                snooze_state.clear()

            # ── Check scheduled reminder times ──
            slot_key = f"{day_key} {hhmm}"
            if hhmm in MEAL_TIME_MAP and slot_key not in fired_today:
                meal_idx = MEAL_TIME_MAP[hhmm]
                meal_name = MEALS[meal_idx]["name"]
                if not _check_meal_logged(meal_idx):
                    fired_today.add(slot_key)
                    cal, protein = _get_remaining()
                    _say(f"\n  Assistant: Hey, wanna log {meal_name.lower()}? You still need {cal} cal and {protein}g protein today.\n")

            # ── Check snoozed meals ──
            for meal_idx in list(snooze_state.keys()):
                wake_time = snooze_state[meal_idx]
                if now >= wake_time:
                    del snooze_state[meal_idx]
                    meal_name = MEALS[meal_idx]["name"]
                    if not _check_meal_logged(meal_idx):
                        cal, protein = _get_remaining()
                        _say(f"\n  Assistant: Checking back — did you have {meal_name.lower()}? Still need {cal} cal and {protein}g protein.\n")
                    # else: meal was logged, no need to remind

            stop_event.wait(20)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


SUMMARY_SENT_FILE = os.path.join(SCRIPT_DIR, ".summary_sent")


def _was_summary_sent_today():
    """Check if summary was already sent today by reading the marker file."""
    try:
        with open(SUMMARY_SENT_FILE, "r") as f:
            last_date = f.read().strip()
        return last_date == date.today().isoformat()
    except (FileNotFoundError, ValueError):
        return False


def _mark_summary_sent():
    """Write today's date to the marker file."""
    with open(SUMMARY_SENT_FILE, "w") as f:
        f.write(date.today().isoformat())


def start_nightly_summary_loop(session, summary_time, stop_event):
    """Background thread that texts a daily diet summary at the configured time.
    If the target time was missed (e.g. laptop was closed), sends as soon as
    the program is running again, as long as it's still the same day.
    Uses a marker file so it only sends once per day even across restarts."""
    if not summary_time:
        return

    # Normalize to HH:MM (zero-padded) so "8:52" becomes "08:52"
    try:
        h, m = summary_time.split(":")
        target_h, target_m = int(h), int(m)
        summary_time = f"{target_h:02d}:{target_m:02d}"
    except (ValueError, AttributeError):
        print(f"  [summary] Invalid time format: {summary_time}. Use HH:MM.")
        return

    def _send_summary():
        """Build and send the daily summary. Returns True if sent."""
        try:
            msg = build_daily_summary(session)
            if msg:
                ok = send_sms(msg)
                if ok:
                    _mark_summary_sent()
                    print(f"\n  Assistant: Sent your nightly diet summary!\n")
                    return True
                else:
                    print(f"\n  Assistant: Couldn't send summary. Check config.\n")
            else:
                print(f"\n  Assistant: No diary data to summarize today.\n")
        except Exception as e:
            print(f"\n  Assistant: Error building summary: {e}\n")
        return False

    def _loop():
        print(f"  [summary] Loop started at {datetime.now().strftime('%H:%M:%S')}, "
              f"daily summary at {summary_time}")
        while not stop_event.is_set():
            now = datetime.now()
            current_h, current_m = now.hour, now.minute

            # Check if it's at or past the target time and before midnight
            past_target = (current_h > target_h or
                           (current_h == target_h and current_m >= target_m))

            if past_target and not _was_summary_sent_today():
                print(f"  [summary] Triggering daily summary at {now.strftime('%H:%M:%S')}...")
                _send_summary()

            stop_event.wait(30)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


# ── GPT helpers ───────────────────────────────────────────────────────────

def parse_user_message(user_message):
    """
    Ask GPT to extract the meal type and food description from a casual message.
    Returns: {"meal": str|null, "foods": str|null, "intent": str, ...}
    """
    preset_names = get_preset_names()
    presets_hint = ""
    if preset_names:
        presets_hint = f"\nThe user has these saved meal presets: {', '.join(preset_names)}\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You parse casual messages about food logging. Extract the meal type and food description.\n"
                    f"{presets_hint}\n"
                    "Return ONLY a JSON object with these fields:\n"
                    '- "meal": one of "breakfast", "lunch", "dinner", "snacks", or null if not mentioned\n'
                    '- "foods": the food description extracted from their message, or null\n'
                    '- "intent": one of the following:\n'
                    '  - "log": log/add individual food(s)\n'
                    '  - "remove": remove/delete food from diary\n'
                    '  - "skip": skip a meal / haven\'t eaten\n'
                    '  - "create_meal": create/save a new meal preset '
                    '(e.g. "create a meal called lunch combo", "save a meal preset")\n'
                    '  - "log_meal": log a saved meal preset '
                    '(e.g. "log my lunch combo", "log the chicken and rice meal")\n'
                    '  - "manage_meal": view/edit/delete meal presets '
                    '(e.g. "show my meals", "delete lunch combo", "add rice to lunch combo")\n'
                    '  - "quick_add": user specifies exact calories/macros to log directly '
                    '(e.g. "quick add 1500 cals 88g fat 77g protein to dinner", '
                    '"I had mod pizza, 1500 calories, 88g fat, 77g protein for dinner")\n'
                    '  - "defer": user wants to eat/log later, postpone '
                    '(e.g. "I\'ll eat later", "I\'ll do it later", "not yet", "haven\'t eaten yet", "remind me later")\n'
                    '  - "chat": anything else (questions, greetings, etc.)\n'
                    '- "preset_name": the meal preset name (for create_meal, log_meal, manage_meal), or null\n'
                    '- "sub_action": for manage_meal only — "list", "delete", "add_food", "remove_food", or null\n'
                    '- "calories": number or null (for quick_add)\n'
                    '- "protein": number or null (for quick_add)\n'
                    '- "fat": number or null (for quick_add)\n'
                    '- "carbs": number or null (for quick_add)\n\n'
                    "Examples:\n"
                    '- "had a protein bar and chai for breakfast" → {"meal": "breakfast", "foods": "protein bar and chai", "intent": "log", "preset_name": null, "sub_action": null}\n'
                    '- "protein bar and chai" → {"meal": null, "foods": "protein bar and chai", "intent": "log", "preset_name": null, "sub_action": null}\n'
                    '- "remove the chicken from dinner" → {"meal": "dinner", "foods": "chicken", "intent": "remove", "preset_name": null, "sub_action": null}\n'
                    '- "skipped breakfast" → {"meal": "breakfast", "foods": null, "intent": "skip", "preset_name": null, "sub_action": null}\n'
                    '- "create a meal called lunch combo" → {"meal": null, "foods": null, "intent": "create_meal", "preset_name": "lunch combo", "sub_action": null}\n'
                    '- "save a meal preset called morning fuel with eggs and toast" → {"meal": null, "foods": "eggs and toast", "intent": "create_meal", "preset_name": "morning fuel", "sub_action": null}\n'
                    '- "log my lunch combo for lunch" → {"meal": "lunch", "foods": null, "intent": "log_meal", "preset_name": "lunch combo", "sub_action": null}\n'
                    '- "log the chicken and rice" → {"meal": null, "foods": null, "intent": "log_meal", "preset_name": "chicken and rice", "sub_action": null}\n'
                    '- "show my meals" → {"meal": null, "foods": null, "intent": "manage_meal", "preset_name": null, "sub_action": "list"}\n'
                    '- "what meal presets do I have" → {"meal": null, "foods": null, "intent": "manage_meal", "preset_name": null, "sub_action": "list"}\n'
                    '- "delete the lunch combo" → {"meal": null, "foods": null, "intent": "manage_meal", "preset_name": "lunch combo", "sub_action": "delete"}\n'
                    '- "add rice to lunch combo" → {"meal": null, "foods": "rice", "intent": "manage_meal", "preset_name": "lunch combo", "sub_action": "add_food"}\n'
                    '- "remove sauce from lunch combo" → {"meal": null, "foods": "sauce", "intent": "manage_meal", "preset_name": "lunch combo", "sub_action": "remove_food"}\n'
                    '- "how many calories today" → {"meal": null, "foods": null, "intent": "chat", "preset_name": null, "sub_action": null}\n'
                    '- "what is logged for dinner" → {"meal": "dinner", "foods": null, "intent": "chat", "preset_name": null, "sub_action": null}\n'
                    '- "what does my dinner show" → {"meal": "dinner", "foods": null, "intent": "chat", "preset_name": null, "sub_action": null}\n'
                    '- "what did I log today" → {"meal": null, "foods": null, "intent": "chat", "preset_name": null, "sub_action": null}\n'
                    '- "clear my dinner log" → {"meal": "dinner", "foods": null, "intent": "remove", "preset_name": null, "sub_action": null}\n'
                    '- "clear everything from lunch" → {"meal": "lunch", "foods": null, "intent": "remove", "preset_name": null, "sub_action": null}\n'
                    '- "i had mod pizza, 1500 calories, 88g fat, 77g protein for dinner" → {"meal": "dinner", "foods": "mod pizza", "intent": "quick_add", "preset_name": null, "sub_action": null, "calories": 1500, "fat": 88, "protein": 77, "carbs": null}\n'
                    '- "quick add 500 cals 30g protein to lunch" → {"meal": "lunch", "foods": "Quick Add", "intent": "quick_add", "preset_name": null, "sub_action": null, "calories": 500, "protein": 30, "fat": null, "carbs": null}\n'
                    '- "I\'ll eat later" → {"meal": null, "foods": null, "intent": "defer", "preset_name": null, "sub_action": null}\n'
                    '- "not yet, remind me later" → {"meal": null, "foods": null, "intent": "defer", "preset_name": null, "sub_action": null}\n'
                    '- "I\'ll log lunch later" → {"meal": "lunch", "foods": null, "intent": "defer", "preset_name": null, "sub_action": null}\n'
                    "\nIMPORTANT:\n"
                    "- Questions about what's LOGGED/in the diary (e.g. 'what did I log', 'what's in my dinner') = chat, NOT manage_meal.\n"
                    "- manage_meal is ONLY for meal PRESETS (saved combos), not the diary.\n"
                    "- If the user mentions a saved preset name, use log_meal (not log).\n"
                    "- If the user provides SPECIFIC calorie/macro numbers for a food, use quick_add (not log). "
                    'Put the food name in "foods" (e.g. "mod pizza"). If no food name given, use "Quick Add".\n'
                    "Return ONLY the JSON object, nothing else."
                ),
            },
            {"role": "user", "content": user_message},
        ],
    )

    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"meal": None, "foods": None, "intent": "chat"}


def match_foods_with_gpt(food_description, food_list):
    """
    Ask GPT to match user's food description to MFP recent foods.
    Returns a list of {"description": str, "matches": [int]}.
    """
    food_names = [f"{i}: {f['name']}" for i, f in enumerate(food_list)]
    food_list_str = "\n".join(food_names)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a food-matching assistant. The user describes what they ate. "
                    "You have a numbered list of their MyFitnessPal recent foods.\n\n"
                    "Break the user's message into INDIVIDUAL food items, then find the best match for each.\n\n"
                    "Return a JSON array of objects, one per food item the user mentioned:\n"
                    '[{"description": "what they said", "matches": [index_numbers]}]\n\n'
                    "Rules:\n"
                    "- Each food item the user mentioned gets its own object\n"
                    "- If there's ONE clear best match, return just that index: [5]\n"
                    "- If there are MULTIPLE similar options (e.g. 'protein shake' could be index 5 or 18), "
                    "return all candidates: [5, 18] — the user will pick\n"
                    "- If no match exists, return empty: []\n"
                    "- Match loosely: 'protein bar' matches anything with 'protein' and 'bar'\n"
                    "- 'chai' matches 'Chai Lucerne - Chai'\n"
                    '- "protein bar and chai" → two separate items\n'
                    '- "protein shake" with two shake options → one item with two matches\n'
                    "- Return ONLY the JSON array, nothing else"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"I ate: {food_description}\n\n"
                    f"Available foods:\n{food_list_str}"
                ),
            },
        ],
    )

    raw = response.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            for item in result:
                if "matches" in item:
                    item["matches"] = [i for i in item["matches"] if isinstance(i, int) and 0 <= i < len(food_list)]
            return result
    except json.JSONDecodeError:
        pass
    return []


def match_search_results_with_gpt(food_description, search_results, prefer_custom=False):
    """
    Ask GPT to pick the best match from MFP search results for a single food.
    Returns (matched_food_dict, confidence) or (None, "none").
    Set prefer_custom=True when the user said "my [food]" to force custom entry selection.
    """
    food_names = []
    for i, f in enumerate(search_results):
        tag = "[custom]" if not f.get("verified") else "[mfp]"
        cal = f" ({f['cal_info']})" if f.get("cal_info") else ""
        food_names.append(f"{i}: {tag} {f['name']}{cal}")
    food_list_str = "\n".join(food_names)

    if prefer_custom:
        system_prompt = (
            "You pick the SINGLE best match from MFP search results for a food item.\n\n"
            'Return ONLY a JSON object: {"match": index_number, "confidence": "high"|"medium"|"low"}\n'
            'If nothing matches well, return: {"match": null, "confidence": "none"}\n\n'
            "IMPORTANT: The user said 'my [food]' — they want THEIR OWN saved custom food.\n"
            "Rules (in order of priority):\n"
            "1. Among [custom] entries, pick the one whose name most closely matches the description — "
            "prefer exact or near-exact name matches over partial matches\n"
            "2. Only consider [mfp] entries if absolutely no [custom] entry is present\n"
            "3. Return ONLY the JSON object"
        )
    else:
        system_prompt = (
            "You pick the SINGLE best match from MFP search results for a food item.\n\n"
            'Return ONLY a JSON object: {"match": index_number, "confidence": "high"|"medium"|"low"}\n'
            'If nothing matches well, return: {"match": null, "confidence": "none"}\n\n'
            "Rules (in order of priority):\n"
            "1. If any result's name exactly matches or very closely matches the description, pick it — even if tagged [custom]\n"
            "2. [custom] entries are the user's own saved foods — strongly prefer them over [mfp] generic entries when the name matches\n"
            "3. Only prefer [mfp] entries if no [custom] entry is a reasonable match\n"
            "4. Return ONLY the JSON object"
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Looking for: {food_description}\n\nSearch results:\n{food_list_str}",
            },
        ],
    )

    raw = response.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
        idx = result.get("match")
        if idx is not None and isinstance(idx, int) and 0 <= idx < len(search_results):
            return search_results[idx], result.get("confidence", "medium")
    except json.JSONDecodeError:
        pass
    return None, "none"


def split_food_descriptions(food_description):
    """Split a free-form food phrase into likely individual food items."""
    parts = re.split(r",| and | with |\+", food_description, flags=re.IGNORECASE)
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return cleaned or [food_description.strip()]


def normalize_food_query(food_text):
    """Strip conversational quantity words so search uses a cleaner food name."""
    q = food_text.strip().lower()
    q = re.sub(r"^(can you )?(just )?(log|add)\s+", "", q)
    q = re.sub(r"^(a|an|the|some|my)\s+", "", q)
    q = re.sub(r"^(regular|normal|standard)\s+(serving|portion)\s+of\s+", "", q)
    q = re.sub(r"^(serving|portion)\s+of\s+", "", q)
    q = re.sub(r"\s+", " ", q).strip(" .")
    return q or food_text.strip()


def split_compound_food_for_search(food_text):
    """Break compound dish names into multiple searchable food items."""
    raw = normalize_food_query(food_text)
    if not raw:
        return []

    # Keep known compound dish names as a single searchable item.
    # These frequently get over-split into ingredients and produce bad matches.
    protected_dishes = {
        "palak paneer",
        "saag paneer",
        "shahi paneer",
        "matar paneer",
        "paneer tikka masala",
        "dal makhani",
        "aloo gobi",
        "chole bhature",
        "chana masala",
        "butter chicken",
        "chicken tikka masala",
    }
    if raw.lower() in protected_dishes:
        return [raw]

    # Fast path: obvious separators.
    quick_parts = re.split(r",| and | with |&|\/|\+", raw, flags=re.IGNORECASE)
    quick_parts = [p.strip() for p in quick_parts if p and p.strip()]
    if len(quick_parts) > 1:
        return quick_parts

    # Heuristics for common South Asian terms (can be extended).
    term_map = {
        "chawal": "rice",
        "chawal": "rice",
        "rice": "rice",
        "channay": "channay",
        "chanay": "channay",
        "chana": "channay",
        "chole": "channay",
    }
    tokens = re.findall(r"[a-zA-Z]+", raw.lower())
    mapped = []
    for t in tokens:
        if t in term_map:
            mapped.append(term_map[t])
    if len(set(mapped)) >= 2:
        return list(dict.fromkeys(mapped))

    # GPT fallback for compound phrases without separators.
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Split a food phrase into individual food items for diary logging.\n"
                        "Return ONLY a JSON array of short strings.\n"
                        "Examples:\n"
                        '- "channay chawal" -> ["channay", "rice"]\n'
                        '- "rajma chawal" -> ["rajma", "rice"]\n'
                        '- "palak paneer" -> ["palak paneer"]\n'
                        '- "protein bar" -> ["protein bar"]\n'
                        "If it is one item, return one string in the array."
                    ),
                },
                {"role": "user", "content": raw},
            ],
        )
        parsed = json.loads(response.choices[0].message.content.strip())
        if isinstance(parsed, list):
            cleaned = [str(x).strip() for x in parsed if str(x).strip()]
            if cleaned:
                return cleaned
    except Exception:
        pass

    return [raw]


HOUSEHOLD_UNIT_TO_SERVINGS = {
    "katori": 0.75,
    "katoris": 0.75,
    "bowl": 1.0,
    "bowls": 1.0,
    "plate": 1.5,
    "plates": 1.5,
}


def parse_household_quantity_to_servings(text):
    """Parse quantities like '1-2 katori' / '1 plate' into serving multiplier."""
    t = (text or "").lower().strip()
    if not t:
        return None

    # If no supported household unit is present, skip.
    unit = None
    for u in HOUSEHOLD_UNIT_TO_SERVINGS.keys():
        if re.search(rf"\b{re.escape(u)}\b", t):
            unit = u
            break
    if not unit:
        return None

    base = HOUSEHOLD_UNIT_TO_SERVINGS[unit]

    # Common words
    if re.search(r"\bhalf\b", t):
        amount = 0.5
    elif re.search(r"\bquarter\b", t):
        amount = 0.25
    elif re.search(r"\bdouble\b", t):
        amount = 2.0
    elif re.search(r"\btriple\b", t):
        amount = 3.0
    else:
        # Number range: 1-2 / 1 to 2
        m_range = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", t)
        if m_range:
            lo = float(m_range.group(1))
            hi = float(m_range.group(2))
            amount = (lo + hi) / 2.0
        else:
            m_num = re.search(r"(\d+(?:\.\d+)?)", t)
            amount = float(m_num.group(1)) if m_num else 1.0

    servings = amount * base
    if servings <= 0:
        return None
    return round(servings, 2), unit, amount, base


def _tokenize_food_text(text):
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    stop = {
        "a", "an", "the", "my", "to", "for", "with", "and", "or", "of",
        "please", "log", "add", "meal", "breakfast", "lunch", "dinner", "snacks"
    }
    return {t for t in tokens if t not in stop}


def has_reasonable_recent_overlap(description, candidate_names):
    """Reject clearly unrelated recent-food matches before asking user to pick.

    Requires at least half the description tokens to match at least one candidate.
    E.g. "tandoori curry pizza" (3 tokens) needs >=2 matching tokens, so
    "chicken curry" (only 1 shared token) gets rejected.
    """
    desc_tokens = _tokenize_food_text(description)
    if not desc_tokens:
        return True

    min_overlap = max(1, (len(desc_tokens) + 1) // 2)  # ceiling division

    for name in candidate_names:
        shared = desc_tokens & _tokenize_food_text(name)
        if len(shared) >= min_overlap:
            return True
    return False


def pick_reasonable_search_result(search_query, search_results):
    """Heuristic fallback when GPT confidence is low on search matches."""
    q = search_query.lower().strip()
    best = None
    best_score = -10**9

    q_tokens = _tokenize_food_text(q)
    generic_names = {
        "cooked", "raw", "steamed", "roasted", "fried",
        "boiled", "baked", "grilled", "food", "item",
    }

    for item in search_results:
        name = item.get("name", "").strip()
        if not name:
            continue
        lower = name.lower()
        score = 0
        n_tokens = _tokenize_food_text(lower)

        if lower == q:
            score += 8
        if lower.startswith(q):
            score += 5
        if q in lower:
            score += 3
        if q_tokens and (q_tokens & n_tokens):
            score += 4
        else:
            score -= 6
        if " - " in name:
            score -= 3
        if any(tag in lower for tag in ("raw", "cooked", "steamed", "roasted")):
            score += 1
        if lower in generic_names or (len(n_tokens) == 1 and next(iter(n_tokens), "") in generic_names):
            score -= 8

        if score > best_score:
            best_score = score
            best = item

    if best is None:
        return None

    # Final guard: don't return clearly generic/no-overlap names.
    best_name = best.get("name", "").lower()
    best_tokens = _tokenize_food_text(best_name)
    if q_tokens and not (q_tokens & best_tokens):
        return None
    if best_name in generic_names:
        return None

    return best


def pick_meal_bundle_results(search_query, search_results, max_items=6):
    """Pick multiple search results when user asks to log a named 'meal'."""
    q = (search_query or "").lower()
    if "meal" not in q:
        return []

    stop = {"meal", "combo", "plate", "bowl", "box", "set"}
    q_tokens = [t for t in _tokenize_food_text(q) if t not in stop]
    if not q_tokens:
        return []

    bundle = []
    for item in search_results:
        name = (item.get("name") or "").lower()
        if not name:
            continue
        if any(tok in name for tok in q_tokens):
            bundle.append(item)
        if len(bundle) >= max_items:
            break

    return bundle if len(bundle) >= 2 else []


def smart_reply(user_input, diary_context):
    """Let GPT answer a question using diary data."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a casual, chill diet assistant for {USER_NAME}. "
                    f"Answer based on the diary data below. "
                    f"Be super short and casual — 1 sentence max, like texting a friend. "
                    f"No filler, no pleasantries. Just the answer. "
                    f"When listing macros, always use this order: calories, protein, fat, carbs.\n\n"
                    f"{diary_context}"
                ),
            },
            {"role": "user", "content": user_input},
        ],
    )
    return response.choices[0].message.content.strip()


def parse_per_food_quantities_with_gpt(user_input, pending_names):
    """Extract per-food serving quantities from a confirmation message."""
    if not pending_names:
        return {}
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract per-food serving quantities from the user's sentence.\n"
                        "You are given the pending food names. Return ONLY JSON object mapping food name to quantity.\n"
                        "Use decimal numbers. Handle fractions like 1/2 -> 0.5.\n"
                        "Convert household units to servings when present:\n"
                        "- katori = 0.75 servings\n"
                        "- bowl = 1.0 servings\n"
                        "- plate = 1.5 servings\n"
                        "Examples:\n"
                        '- "2 rotis, 1 katori palak paneer, 1 katori karela" => {"rotis": 2, "palak paneer": 0.75, "karela": 0.75}\n'
                        "If no explicit per-food quantities are present, return {}.\n"
                        "Only include foods from the pending list."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Pending foods: {', '.join(pending_names)}\n"
                        f"User: {user_input}"
                    ),
                },
            ],
        )
        raw = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            out = {}
            for k, v in parsed.items():
                try:
                    qty = float(v)
                except (ValueError, TypeError):
                    continue
                if qty > 0:
                    out[str(k).strip()] = qty
            return out
    except Exception:
        pass
    return {}


def parse_nutrition_from_cal_info(cal_info):
    """Extract calories/protein/carbs/fat from text like '1 cup, 204 calories'."""
    text = (cal_info or "").lower()

    def grab(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    return {
        "calories": grab(r"(\d+(?:\.\d+)?)\s*cal"),
        "protein": grab(r"(\d+(?:\.\d+)?)\s*g\s*protein"),
        "carbs": grab(r"(\d+(?:\.\d+)?)\s*g\s*carb"),
        "fat": grab(r"(\d+(?:\.\d+)?)\s*g\s*fat"),
    }


def _num_or_none(value):
    try:
        if value is None:
            return None
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


# ── Meal state tracking ───────────────────────────────────────────────────

def get_unlogged_meals(session):
    """Return list of (meal_idx, meal_name) that haven't been logged today."""
    logged = check_meals_logged(session, date.today())
    return [(idx, info["name"]) for idx, info in MEALS.items() if not logged.get(idx, False)]


def guess_meal_from_time():
    """Guess which meal the user is probably talking about based on time."""
    hour = datetime.now().hour
    if hour < 11:
        return 0   # Breakfast
    elif hour < 18:
        return 1   # Lunch (noon–6 PM)
    else:
        return 2   # Dinner (after 6 PM)


def resolve_meal_idx(mentioned_meal, unlogged):
    """Figure out which meal index the user means."""
    if mentioned_meal and mentioned_meal in MEAL_NAMES_TO_IDX:
        return MEAL_NAMES_TO_IDX[mentioned_meal]

    unlogged_indices = [idx for idx, _ in unlogged]
    if len(unlogged_indices) == 1:
        return unlogged_indices[0]

    guessed = guess_meal_from_time()
    if guessed in unlogged_indices:
        return guessed
    if unlogged_indices:
        return unlogged_indices[0]
    return guessed


def build_diary_context(session):
    """Build a text summary of today's diary for GPT."""
    diary, summary = get_diary_details(session, date.today())
    unlogged = get_unlogged_meals(session)
    totals = summary.get("totals", {})
    goal = summary.get("goal", {})
    remaining = summary.get("remaining", {})

    lines = []
    for meal_idx, info in MEALS.items():
        foods = diary.get(meal_idx, [])
        if foods:
            food_names = ", ".join(f["name"] for f in foods)
            lines.append(f"{info['name']}: {food_names}")
        else:
            lines.append(f"{info['name']}: (nothing logged)")

    unlogged_names = ", ".join(n for _, n in unlogged) if unlogged else "none"

    return (
        f"Today's diary:\n" + "\n".join(lines) + "\n\n"
        f"Totals: {totals.get('calories', '?')} cal, "
        f"{totals.get('carbs', '?')}g carbs, "
        f"{totals.get('fat', '?')}g fat, "
        f"{totals.get('protein', '?')}g protein\n"
        f"Daily goal: {goal.get('calories', '?')} cal, {goal.get('protein', '?')}g protein\n"
        f"Remaining: {remaining.get('calories', '?')} cal, {remaining.get('protein', '?')}g protein\n"
        f"Unlogged meals: {unlogged_names}"
    )


# ── Terminal chat mode ────────────────────────────────────────────────────

def run_terminal_chat(reminder_times=None, summary_time=None, input_provider=None, io_lock=None, speaker=None):
    """Interactive terminal chat — the assistant asks about meals."""
    if input_provider is None:
        from voice_io import TextInputProvider
        input_provider = TextInputProvider()
    print(f"\n{'═' * 50}")
    print(f"  Diet Assistant for {USER_NAME}")
    print(f"  {datetime.now().strftime('%A, %B %d %Y — %I:%M %p')}")
    print(f"{'═' * 50}\n")

    session = create_session()
    reminder_stop = threading.Event()
    summary_stop = threading.Event()
    snooze_state = {}  # shared with reminder thread: {meal_idx: datetime_to_recheck}
    start_reminder_loop(session, reminder_times or [], reminder_stop, snooze_state,
                        io_lock, input_provider=input_provider, speaker=speaker)
    start_nightly_summary_loop(session, summary_time, summary_stop)

    unlogged = get_unlogged_meals(session)

    if not unlogged:
        opener = f"Hey {USER_NAME}, all meals logged today, nice."
    else:
        opener = f"Hey {USER_NAME}, how can I be of assistance?"
    print(f"  Assistant: {opener}\n")

    # ── State ──
    pending_meal_idx = None
    last_meal_idx = None        # remembers the meal even after reset (for follow-up)
    pending_foods = []          # resolved foods ready to log
    pending_searched_foods = [] # foods resolved via MFP search fallback
    pending_search_metadata = None  # token/date from search_foods (for log_searched_food)
    pending_quantity = "1"      # number of servings to log (adjustable via corrections)
    pending_item_quantities = {}  # optional per-food quantities, e.g. {"rice": 0.5}
    pending_all_foods = None    # all MFP foods for this meal (needed for POST)
    pending_metadata = None
    disambiguation_queue = []   # list of {"description": str, "options": [food_dicts]}
    not_found = []              # food descriptions with no match
    awaiting_confirmation = False
    awaiting_pick = False
    pick_options = []

    def reset_state():
        nonlocal pending_meal_idx, pending_foods, pending_searched_foods, pending_search_metadata
        nonlocal pending_all_foods, pending_metadata, pending_quantity, pending_item_quantities
        nonlocal disambiguation_queue, not_found, awaiting_confirmation, awaiting_pick, pick_options
        pending_meal_idx = None
        pending_foods = []
        pending_searched_foods = []
        pending_search_metadata = None
        pending_quantity = "1"
        pending_item_quantities = {}
        pending_all_foods = None
        pending_metadata = None
        disambiguation_queue = []
        not_found = []
        awaiting_confirmation = False
        awaiting_pick = False
        pick_options = []

    def add_or_increment_pending(food, is_search=False):
        """Add a pending food once; duplicates increase servings instead."""
        nonlocal pending_foods, pending_searched_foods, pending_item_quantities

        target = pending_searched_foods if is_search else pending_foods

        def same_item(a, b):
            if is_search:
                a_id = a.get("original_id") or a.get("external_id") or a.get("name", "").lower()
                b_id = b.get("original_id") or b.get("external_id") or b.get("name", "").lower()
                return str(a_id) == str(b_id)
            return str(a.get("food_id", "")) == str(b.get("food_id", ""))

        existing = next((x for x in target if same_item(x, food)), None)
        if existing is None:
            target.append(food)
            return

        name = existing.get("name", food.get("name", ""))
        base_qty = _num_or_none(existing.get("quantity")) or 1.0
        current_qty = _num_or_none(pending_item_quantities.get(name))
        if current_qty is None:
            current_qty = base_qty
        increment = _num_or_none(food.get("quantity")) or 1.0
        pending_item_quantities[name] = round(current_qty + increment, 3)

    def process_next_disambiguation():
        """Pop the next ambiguous item and ask the user to pick."""
        nonlocal awaiting_pick, pick_options, awaiting_confirmation
        if disambiguation_queue:
            item = disambiguation_queue.pop(0)
            options = item["options"]
            desc = item["description"]
            print(f"\n  Assistant: Which \"{desc}\" did you mean?")
            for i, opt in enumerate(options, 1):
                print(f"    [{i}] {opt['name']}")
            print()
            awaiting_pick = True
            pick_options = options
        else:
            # All disambiguations resolved — go to confirmation
            show_confirmation()

    def show_confirmation():
        """Show all resolved foods and ask to confirm.

        If ALL foods came from recents (no searched foods), auto-log 1 serving
        without asking.  Only prompt for quantity when there are searched/unfamiliar
        foods (e.g. Indian food estimation).
        """
        nonlocal awaiting_confirmation
        if not pending_foods and not pending_searched_foods:
            if not_found:
                descs = ", ".join(not_found)
                print(f"\n  Assistant: Couldn't find: {descs}")
            reset_state()
            return

        meal_name = MEALS[pending_meal_idx]["name"]
        all_pending_names = [f["name"] for f in pending_foods] + [f["name"] for f in pending_searched_foods]
        food_names = ", ".join(all_pending_names)

        if not_found:
            descs = ", ".join(not_found)
            print(f"\n  Assistant: Couldn't find: {descs}")

        # Auto-log if everything matched from recents (known foods)
        if pending_foods and not pending_searched_foods:
            count = len(pending_foods)
            _, _, logged_names, failed_names, any_success = commit_pending_logs()
            if any_success:
                print(f"\n  Assistant: Got it, logged {count} item(s) to {meal_name}.")
                if failed_names:
                    print(f"  Assistant: Couldn't log: {', '.join(failed_names)}")
            else:
                print(f"\n  Assistant: Something went wrong logging those.")
            print(f"  Assistant: Anything else?\n")
            reset_state()
            return

        # Searched/unfamiliar foods — ask about quantity
        count = len(pending_foods) + len(pending_searched_foods)
        print(f"\n  Assistant: Found {count} item(s) for {meal_name}. How many servings? (e.g. '1.5 servings', '2', or 'yes' for 1 serving)\n")
        awaiting_confirmation = True

    def commit_pending_logs():
        """Log both recent-food matches and search-fallback matches."""
        recent_success = False
        searched_success_count = 0
        logged_names = []
        failed_names = []

        # Check if user explicitly set a quantity (not the default "1")
        user_set_quantity = (pending_quantity != "1" or bool(pending_item_quantities))

        def quantity_for_food(food_name, default_qty=None):
            """Get quantity for a food. If user didn't set one, use MFP default."""
            for k, v in pending_item_quantities.items():
                if k.lower() in food_name.lower() or food_name.lower() in k.lower():
                    return float(v)
            if user_set_quantity:
                try:
                    return float(pending_quantity)
                except (ValueError, TypeError):
                    return 1.0
            # Use the food's own default quantity from MFP recents
            if default_qty is not None:
                try:
                    return float(default_qty)
                except (ValueError, TypeError):
                    pass
            return 1.0

        if pending_foods:
            # Apply quantity override to the all_foods list entries that match selected foods
            if pending_all_foods:
                selected_ids = {f["food_id"] for f in pending_foods}
                for f in pending_all_foods:
                    if f["food_id"] in selected_ids:
                        matched = next((x for x in pending_foods if x["food_id"] == f["food_id"]), None)
                        if matched:
                            q = quantity_for_food(matched["name"], default_qty=matched.get("quantity"))
                            f["quantity"] = str(q)
            recent_success = log_foods(session, pending_foods, pending_all_foods, pending_meal_idx, pending_metadata)
            if recent_success:
                for f in pending_foods:
                    q = quantity_for_food(f["name"], default_qty=f.get("quantity"))
                    qty_display = f" (x{q:g})" if q != 1 else ""
                    logged_names.append(f["name"] + qty_display)
            else:
                failed_names.extend([f["name"] for f in pending_foods])

        for food in pending_searched_foods:
            q = quantity_for_food(food["name"], default_qty=food.get("quantity"))
            if log_searched_food(session, food, pending_meal_idx, pending_search_metadata, quantity=str(q)):
                searched_success_count += 1
                qty_display = f" (x{q:g})" if q != 1 else ""
                logged_names.append(food["name"] + qty_display)
            else:
                failed_names.append(food["name"])

        any_success = len(logged_names) > 0
        if any_success:
            invalidate_recents_cache(pending_meal_idx)
        return recent_success, searched_success_count, logged_names, failed_names, any_success

    def start_food_matching(food_desc, meal_idx, rejected_names=None):
        """Fetch MFP foods and match. Sets up disambiguation queue."""
        nonlocal pending_meal_idx, pending_foods, pending_searched_foods, pending_search_metadata
        nonlocal pending_all_foods, pending_metadata
        nonlocal disambiguation_queue, not_found, last_meal_idx

        last_meal_idx = meal_idx
        meal_name = MEALS[meal_idx]["name"]
        print(f"\n  Assistant: Checking {meal_name}...")
        all_foods, metadata = get_recent_foods_cached(session, meal_idx)

        # Filter out recently rejected foods so they don't get re-matched
        if rejected_names and all_foods:
            rejected_lower = {n.lower() for n in rejected_names}
            all_foods = [f for f in all_foods if f["name"].lower() not in rejected_lower]

        pending_meal_idx = meal_idx
        pending_all_foods = all_foods
        pending_metadata = metadata
        pending_foods = []
        pending_searched_foods = []
        disambiguation_queue = []
        not_found = []

        if all_foods:
            match_results = match_foods_with_gpt(food_desc, all_foods)
            for item in match_results:
                matches = item.get("matches", [])
                desc = item.get("description", "")

                if len(matches) == 1:
                    candidate = all_foods[matches[0]]
                    if has_reasonable_recent_overlap(desc, [candidate["name"]]):
                        add_or_increment_pending(candidate, is_search=False)
                    else:
                        not_found.append(desc)
                elif len(matches) > 1:
                    options = [all_foods[i] for i in matches]
                    option_names = [o["name"] for o in options]
                    if has_reasonable_recent_overlap(desc, option_names):
                        disambiguation_queue.append({
                            "description": desc,
                            "options": options,
                        })
                    else:
                        not_found.append(desc)
                else:
                    not_found.append(desc)

            if not match_results:
                not_found.extend(split_food_descriptions(food_desc))
        else:
            print(f"  Assistant: Nothing in recents for {meal_name}, searching MFP...")
            not_found.extend(split_food_descriptions(food_desc))

        # Search fallback for items not in recents
        search_cache = {}  # query -> (search_results, search_meta)
        still_not_found = []
        for desc in not_found:
            parts = split_compound_food_for_search(desc)
            # Preserve counts but avoid repeated searches for duplicate parts.
            part_counts = {}
            unique_parts = []
            for p in parts:
                key = normalize_food_query(p).lower()
                part_counts[key] = part_counts.get(key, 0) + 1
                if part_counts[key] == 1:
                    unique_parts.append(p)
            unresolved_parts = []
            if len(parts) > 1:
                display_parts = []
                for p in unique_parts:
                    key = normalize_food_query(p).lower()
                    count = part_counts.get(key, 1)
                    display_parts.append(f"{p} x{count}" if count > 1 else p)
                print(f"  Assistant: Splitting \"{desc}\" into: {', '.join(display_parts)}")

            for part in unique_parts:
                search_query = normalize_food_query(part)
                part_key = search_query.lower()
                multiplier = part_counts.get(part_key, 1)
                if search_query in search_cache:
                    search_results, search_meta = search_cache[search_query]
                else:
                    count_note = f" (x{multiplier})" if multiplier > 1 else ""
                    print(f"  Assistant: Searching MFP for \"{search_query}\"{count_note}...")
                    search_results, search_meta = search_foods(session, search_query, meal_idx)
                    search_cache[search_query] = (search_results, search_meta)
                if search_meta and not pending_search_metadata:
                    pending_search_metadata = search_meta
                if search_results:
                    meal_bundle = pick_meal_bundle_results(search_query, search_results)
                    if meal_bundle:
                        print(f"  Assistant: Found meal \"{search_query}\" with {len(meal_bundle)} item(s):")
                        for m in meal_bundle:
                            print(f"    - {m['name']}")
                            for _ in range(multiplier):
                                add_or_increment_pending(m, is_search=True)
                        continue

                    best_match, confidence = match_search_results_with_gpt(search_query, search_results)
                    if best_match and confidence in ("high", "medium"):
                        print(f"  Assistant: Found: {best_match['name']}")
                        for _ in range(multiplier):
                            add_or_increment_pending(best_match, is_search=True)
                    else:
                        fallback = pick_reasonable_search_result(search_query, search_results)
                        if fallback:
                            print(f"  Assistant: Found likely match: {fallback['name']}")
                            for _ in range(multiplier):
                                add_or_increment_pending(fallback, is_search=True)
                        else:
                            unresolved_parts.extend([part] * multiplier)
                else:
                    unresolved_parts.extend([part] * multiplier)

            still_not_found.extend(unresolved_parts)
        not_found = still_not_found

        # If there are disambiguations, start asking
        process_next_disambiguation()

    # ── Main loop ──
    try:
        while True:
            try:
                user_input = input_provider.get_input(f"  {USER_NAME}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nBye!")
                break

            if not user_input:
                continue
            lower_input_check = user_input.lower().strip()
            goodbye_keywords = ("quit", "exit", "bye", "q")
            goodbye_phrases = ["good day", "good night", "see you", "see ya", "talk later",
                               "talk to you later", "gotta go", "i'm done", "that's all",
                               "that's it", "peace", "later", "cya", "goodbye", "good bye",
                               "have a good", "take care", "catch you later", "im out",
                               "nope that's it", "nope thats it", "that should be it",
                               "sounds good", "alright thanks", "all right thanks",
                               "all right sounds good", "thanks that's it", "thank you",
                               "thanks bye", "ok thanks", "ok bye", "okay bye",
                               "perfect thanks", "cool thanks", "great thanks", "appreciate it",
                               "thats it", "thats all", "im done"]
            import string
            trans = str.maketrans("", "", string.punctuation)
            clean_input = lower_input_check.translate(trans)
            is_goodbye = (lower_input_check in goodbye_keywords or
                          any(phrase.translate(trans) in clean_input for phrase in goodbye_phrases))
            if is_goodbye:
                print("\n  Assistant: Talk to you later!\n")
                break

            # ── Picking from disambiguation options ──
            if awaiting_pick:
                choice = None
                raw_pick = user_input.strip().lower()

                # Numeric input: "1", "2", ...
                if re.fullmatch(r"\d+", raw_pick):
                    choice = int(raw_pick) - 1
                else:
                    # Natural language: "first one", "option 2", "the last one"
                    clean_pick = re.sub(r"[^a-z0-9\s#]", " ", raw_pick)
                    clean_pick = re.sub(r"\s+", " ", clean_pick).strip()

                    ord_map = {
                        "first": 0,
                        "second": 1,
                        "third": 2,
                        "fourth": 3,
                        "fifth": 4,
                    }
                    for word, idx in ord_map.items():
                        if re.search(rf"\b{word}\b", clean_pick):
                            choice = idx
                            break

                    if choice is None and re.search(r"\blast\b", clean_pick):
                        choice = len(pick_options) - 1

                    if choice is None:
                        m = re.search(r"(?:option|number|#)?\s*(\d+)", clean_pick)
                        if m:
                            choice = int(m.group(1)) - 1

                if choice is not None and 0 <= choice < len(pick_options):
                    chosen = pick_options[choice]
                    add_or_increment_pending(chosen, is_search=False)
                    awaiting_pick = False
                    print(f"\n  Assistant: Got it — {chosen['name']}.")
                    process_next_disambiguation()
                    continue

                print(f"\n  Assistant: I didn't catch the pick. Say 1-{len(pick_options)} or \"first one\".\n")
                continue

            # ── Confirmation flow ──
            if awaiting_confirmation:
                pending_names_for_parse = [f["name"] for f in pending_foods] + [f["name"] for f in pending_searched_foods]

                # Prefer per-food parsing first for multi-item inputs.
                per_food_qty = parse_per_food_quantities_with_gpt(user_input, pending_names_for_parse)
                if per_food_qty:
                    pending_item_quantities = per_food_qty
                    print(f"\n  Assistant: Got it — logging: {user_input.strip()}.")
                    print("  Assistant: Go ahead? (yes/no)\n")
                    continue

                # Fast path: household quantity expressions (katori/plate/bowl).
                lower_qty_input = user_input.lower()
                mentions_multiple_items = ("," in lower_qty_input or " and " in lower_qty_input)
                mentions_food_name = any(
                    any(tok in lower_qty_input for tok in _tokenize_food_text(name))
                    for name in pending_names_for_parse
                )
                simple_household_input = not (mentions_multiple_items or mentions_food_name)

                qty_parse = parse_household_quantity_to_servings(user_input) if simple_household_input else None
                if qty_parse:
                    servings, unit, amount, base = qty_parse
                    pending_quantity = str(servings)
                    print(
                        f"\n  Assistant: Got it — {amount:g} {unit} "
                        f"(~{base:g} serving each) => {servings:g} servings."
                    )
                    print("  Assistant: Go ahead? (yes/no)\n")
                    continue

                # Build pending food info for context (scaled by current quantity choices)
                pending_info_parts = []
                pending_names = []
                total_cal = 0.0
                total_protein = 0.0
                total_cal_known = False
                total_protein_known = False

                def qty_for_display(food_name):
                    for k, v in pending_item_quantities.items():
                        if k.lower() in food_name.lower() or food_name.lower() in k.lower():
                            try:
                                return float(v)
                            except (ValueError, TypeError):
                                return 1.0
                    try:
                        return float(pending_quantity)
                    except (ValueError, TypeError):
                        return 1.0

                for f in pending_foods:
                    info = f"- {f['name']}"
                    pending_names.append(f['name'])
                    q = qty_for_display(f["name"])
                    cal_val = None
                    protein_val = None
                    try:
                        cal_val = float(str(f.get("calories", "")).replace(",", ""))
                    except (ValueError, TypeError):
                        cal_val = None
                    try:
                        protein_val = float(str(f.get("protein", "")).replace(",", ""))
                    except (ValueError, TypeError):
                        protein_val = None

                    if q != 1:
                        info += f" (x{q:g})"

                    if f.get("calories"):
                        scaled_cal = cal_val * q if cal_val is not None else None
                        scaled_protein = protein_val * q if protein_val is not None else None
                        details = []
                        if scaled_cal is not None:
                            details.append(f"Calories: {scaled_cal:.0f}")
                            total_cal += scaled_cal
                            total_cal_known = True
                        if scaled_protein is not None:
                            details.append(f"Protein: {scaled_protein:.1f}g")
                            total_protein += scaled_protein
                            total_protein_known = True
                        if details:
                            info += " [" + ", ".join(details) + "]"
                    pending_info_parts.append(info)

                for f in pending_searched_foods:
                    info = f"- {f['name']}"
                    pending_names.append(f['name'])
                    q = qty_for_display(f["name"])
                    if q != 1:
                        info += f" (x{q:g})"

                    parsed_nutri = parse_nutrition_from_cal_info(f.get("cal_info"))
                    scaled_cal = parsed_nutri["calories"] * q if parsed_nutri["calories"] is not None else None
                    scaled_protein = parsed_nutri["protein"] * q if parsed_nutri["protein"] is not None else None

                    if f.get("cal_info"):
                        info += f" ({f['cal_info']})"
                    details = []
                    if scaled_cal is not None:
                        details.append(f"Calories: {scaled_cal:.0f}")
                        total_cal += scaled_cal
                        total_cal_known = True
                    if scaled_protein is not None:
                        details.append(f"Protein: {scaled_protein:.1f}g")
                        total_protein += scaled_protein
                        total_protein_known = True
                    if details:
                        info += " [" + ", ".join(details) + "]"
                    pending_info_parts.append(info)

                pending_info = "\n".join(pending_info_parts) if pending_info_parts else "No nutrition info available."
                meal_name = MEALS[pending_meal_idx]["name"]
                totals_line_parts = []
                if total_cal_known:
                    totals_line_parts.append(f"Calories: {total_cal:.0f}")
                if total_protein_known:
                    totals_line_parts.append(f"Protein: {total_protein:.1f}g")
                if totals_line_parts:
                    pending_info += "\n\nEstimated total for selected quantities: " + ", ".join(totals_line_parts)

                # Deterministic macro Q&A path while awaiting confirmation.
                lower_input = user_input.lower()
                asks_macros = any(k in lower_input for k in ["macro", "calorie", "protein", "carb", "fat"])
                if asks_macros:
                    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
                    known = {"calories": False, "protein": False, "carbs": False, "fat": False}

                    def qty_for_macro(food_name):
                        for k, v in pending_item_quantities.items():
                            if k.lower() in food_name.lower() or food_name.lower() in k.lower():
                                try:
                                    return float(v)
                                except (ValueError, TypeError):
                                    return 1.0
                        try:
                            return float(pending_quantity)
                        except (ValueError, TypeError):
                            return 1.0

                    for f in pending_foods:
                        q = qty_for_macro(f["name"])
                        vals = {
                            "calories": _num_or_none(f.get("calories")),
                            "protein": _num_or_none(f.get("protein")),
                            "carbs": _num_or_none(f.get("carbs")),
                            "fat": _num_or_none(f.get("fat")),
                        }
                        for k, v in vals.items():
                            if v is not None:
                                totals[k] += v * q
                                known[k] = True

                    for f in pending_searched_foods:
                        q = qty_for_macro(f["name"])
                        vals = parse_nutrition_from_cal_info(f.get("cal_info"))
                        for k, v in vals.items():
                            if v is not None:
                                totals[k] += v * q
                                known[k] = True

                    lines = []
                    if known["calories"]:
                        lines.append(f"- Calories: {totals['calories']:.0f}")
                    if known["protein"]:
                        lines.append(f"- Protein: {totals['protein']:.1f} g")
                    if known["carbs"]:
                        lines.append(f"- Carbs: {totals['carbs']:.1f} g")
                    if known["fat"]:
                        lines.append(f"- Fat: {totals['fat']:.1f} g")

                    if lines:
                        print("\n  Assistant: Estimated macros:")
                        for line in lines:
                            print(f"  {line}")
                        missing = [k for k, v in known.items() if not v]
                        if missing:
                            print(f"  Assistant: No data for: {', '.join(missing)}.")
                        print("\n  Assistant: Yes to log, no to cancel.\n")
                    else:
                        print("\n  Assistant: No macro data available for these.")
                        print("  Assistant: Yes to log, no to cancel.\n")
                    continue

                # Single GPT call to understand what the user wants
                confirm_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a food logging assistant. The user was asked to confirm logging food.\n"
                                f"Pending food(s) to log to {meal_name}:\n{pending_info}\n\n"
                                "Classify the user's response into ONE of these actions:\n\n"
                                '1. "confirm" — they said yes/ok/sure/yeah/go ahead/log it/do it (confirming the pending food)\n'
                                '2. "reject" — they said no/nah/cancel/wrong/nevermind (rejecting the pending food). '
                                "They may also mention a REPLACEMENT food to search for instead "
                                '(e.g. "no, banza pasta" or "wrong, I meant chicken breast")\n'
                                '3. "adjust_qty" — they want the SAME food but a different number of servings. '
                                'Examples: "log 2 servings" → quantity=2, "I had 3 of these" → quantity=3, '
                                '"can we log 2 servings of this?" → quantity=2, "I had 4 oz not 2 oz" → quantity=2, '
                                '"I had double" → quantity=2, "half a serving" → quantity=0.5. '
                                "Use this when the food is correct but they just need more/less.\n"
                                '4. "correction" — they want a COMPLETELY DIFFERENT ENTRY (different brand, different food name). '
                                'They mention a specific brand or food name that differs from the pending food '
                                '(e.g. "it should be under BANZA", "search for Kirkland chicken instead"). '
                                "Use this only when they want a different product, not just more servings.\n"
                                '5. "add_more" — they confirm the current food AND want to add more foods '
                                '(e.g. "yes, and also rice", "yeah along with asparagus")\n'
                                '6. "question" — they are asking about the pending food '
                                '(e.g. "what are the macros?", "how much protein?")\n\n'
                                "Return ONLY a JSON object:\n"
                                "{\n"
                                '  "action": "confirm"|"reject"|"adjust_qty"|"correction"|"add_more"|"question",\n'
                                '  "replacement_food": "food to search for instead (for reject), or null",\n'
                                '  "quantity": number of servings as a decimal (for adjust_qty, e.g. 2 or 1.5) or null,\n'
                                '  "search_term": "brand/food name to search MFP for (for correction), or null",\n'
                                '  "target_calories": approximate calorie target (for correction) or null,\n'
                                '  "target_protein": approximate protein grams target (for correction) or null,\n'
                                '  "extra_foods": "additional foods to log (for add_more), or null"\n'
                                "}\n\n"
                                "IMPORTANT:\n"
                                '- "ok" or "sure" at the START of a longer message about calories/macros is NOT a confirmation — '
                                "it's an adjust_qty or correction\n"
                                "- If they say 'log X servings' or 'I had X servings' or 'X of these', "
                                "that is adjust_qty with quantity=X. The quantity is the NUMBER they say, not divided by anything.\n"
                                "- If they mention katori/bowl/plate, convert approximately:\n"
                                "  katori ≈ 0.75 servings, bowl ≈ 1 serving, plate ≈ 1.5 servings.\n"
                                "  Example: '1-2 katori' => about 1.125 servings.\n"
                                "- If they specify a physical amount (e.g. '4 oz') and the entry has a different amount (e.g. '2 oz'), "
                                "divide to get quantity (4/2=2)\n"
                                "- 'double' = quantity 2, 'triple' = quantity 3, 'half' = quantity 0.5\n"
                                "- If they mention a DIFFERENT brand or food name, use correction\n"
                                "- If they say no + mention a food, put that food in replacement_food"
                            ),
                        },
                        {"role": "user", "content": user_input},
                    ],
                )
                confirm_raw = confirm_resp.choices[0].message.content.strip()
                try:
                    decision = json.loads(confirm_raw)
                except json.JSONDecodeError:
                    decision = {"action": "question"}

                action = decision.get("action", "question")

                # ── CONFIRM ──
                if action == "confirm":
                    _, _, logged_names, failed_names, any_success = commit_pending_logs()
                    if any_success:
                        print(f"\n  Assistant: Done, logged to {meal_name}.")
                        if failed_names:
                            print(f"  Assistant: Couldn't log: {', '.join(failed_names)}")
                    elif failed_names:
                        print(f"\n  Assistant: Couldn't log: {', '.join(failed_names)}")
                    else:
                        print(f"\n  Assistant: Something went wrong. Check MFP.")
                    reset_state()
                    print(f"  Assistant: Anything else?\n")
                    continue

                # ── REJECT (optionally with replacement food) ──
                elif action == "reject":
                    replacement = decision.get("replacement_food")
                    saved_meal_idx = pending_meal_idx
                    rejected = [f["name"] for f in pending_foods]
                    rejected += [f["name"] for f in pending_searched_foods]
                    reset_state()
                    if replacement:
                        print(f"\n  Assistant: Ok, looking up {replacement} instead...")
                        start_food_matching(replacement, saved_meal_idx, rejected_names=rejected)
                    else:
                        print(f"\n  Assistant: Ok, what'd you have instead?\n")
                    continue

                # ── ADJUST QUANTITY (same food, different number of servings) ──
                elif action == "adjust_qty":
                    qty = decision.get("quantity")
                    if qty and qty > 0:
                        pending_quantity = str(qty)
                        # Show the user what the adjusted entry looks like
                        food_name = pending_names[0] if pending_names else "food"
                        # Try to compute adjusted macros from the pending food info
                        base_food = (pending_searched_foods[0] if pending_searched_foods
                                     else pending_foods[0] if pending_foods else None)
                        cal_info = ""
                        if base_food:
                            ci = base_food.get("cal_info", "")
                            if ci:
                                cal_info = f" (base: {ci})"
                        qty_display = int(qty) if qty == int(qty) else qty
                        print(f"\n  Assistant: Got it — logging {qty_display} servings of {food_name}{cal_info}.")
                        print(f"  Assistant: Go ahead? (yes/no)\n")
                    else:
                        print(f"\n  Assistant: Didn't catch that — how many servings?\n")
                    continue

                # ── ADD MORE (confirm current + search for additional foods) ──
                elif action == "add_more":
                    extra_food = decision.get("extra_foods")
                    if extra_food:
                        _, _, logged_names, failed_names, any_success = commit_pending_logs()
                        if any_success:
                            food_names = ", ".join(logged_names)
                            print(f"\n  Assistant: Logged: {food_names}")
                            if failed_names:
                                print(f"  Assistant: Couldn't auto-log: {', '.join(failed_names)}")
                        elif failed_names:
                            print(f"\n  Assistant: I couldn't auto-log: {', '.join(failed_names)}")
                        saved_meal_idx = pending_meal_idx
                        reset_state()
                        start_food_matching(extra_food, saved_meal_idx)
                    continue

                # ── CORRECTION (search for a better-matching entry) ──
                elif action == "correction":
                    target_cals = decision.get("target_calories")
                    target_protein = decision.get("target_protein")
                    search_term = decision.get("search_term")
                    search_name = search_term if search_term else (pending_names[0] if pending_names else "food")
                    search_query = normalize_food_query(search_name)

                    target_parts = []
                    if target_cals:
                        target_parts.append(f"~{target_cals} cal")
                    if target_protein:
                        target_parts.append(f"~{target_protein}g protein")
                    target_display = ", ".join(target_parts) if target_parts else "a better match"
                    print(f"\n  Assistant: Looking for {target_display} entry for \"{search_query}\"...")

                    search_results, search_meta = search_foods(session, search_query, pending_meal_idx)

                    if search_results:
                        target_desc = search_query
                        if target_cals:
                            target_desc += f" (around {target_cals} calories"
                            if target_protein:
                                target_desc += f", ~{target_protein}g protein"
                            target_desc += ")"

                        food_names_list = [
                            f"{i}: {f['name']}" + (f" ({f['cal_info']})" if f.get("cal_info") else "")
                            for i, f in enumerate(search_results)
                        ]
                        food_list_str = "\n".join(food_names_list)

                        pick_resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            temperature=0,
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "The user wants to log a food but needs a different entry/serving size.\n"
                                        "Pick the search result whose calories/macros are CLOSEST to the user's target.\n"
                                        "Also prefer entries whose NAME matches the search term.\n\n"
                                        'Return ONLY: {"match": index_number, "confidence": "high"|"medium"|"low"}\n'
                                        'If nothing is close: {"match": null, "confidence": "none"}\n'
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": f"Target: {target_desc}\n\nSearch results:\n{food_list_str}",
                                },
                            ],
                        )
                        pick_raw = pick_resp.choices[0].message.content.strip()
                        try:
                            pick_result = json.loads(pick_raw)
                        except json.JSONDecodeError:
                            pick_result = {"match": None}

                        idx = pick_result.get("match")
                        if idx is not None and isinstance(idx, int) and 0 <= idx < len(search_results):
                            new_food = search_results[idx]
                            if search_meta and not pending_search_metadata:
                                pending_search_metadata = search_meta
                            pending_foods.clear()
                            pending_searched_foods.clear()
                            pending_searched_foods.append(new_food)
                            cal_display = f" ({new_food['cal_info']})" if new_food.get('cal_info') else ""
                            print(f"  Assistant: Found: {new_food['name']}{cal_display}")
                            print(f"  Assistant: Go ahead? (yes/no)\n")
                        else:
                            print(f"  Assistant: Couldn't find an entry close to {target_display}.")
                            print(f"  Assistant: Say yes to log current one, or no to cancel.\n")
                    else:
                        print(f"  Assistant: No results for \"{search_query}\".")
                        print(f"  Assistant: Say yes to log current one, or no to cancel.\n")
                    continue

                # ── QUESTION (answer about the pending food) ──
                else:
                    food_question_prompt = (
                        f"The user is about to log these foods to {meal_name} and has a question.\n"
                        f"Pending foods:\n{pending_info}\n\n"
                        f"User question: {user_input}\n\n"
                        f"Answer their question about the pending food(s) based ONLY on the info above. "
                        f"If a macro is missing from the data, say it's not available instead of guessing. "
                        f"Be concise. After answering, remind them they can say yes to log or no to cancel."
                    )
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0,
                        messages=[
                            {"role": "system", "content": food_question_prompt},
                            {"role": "user", "content": user_input},
                        ],
                    )
                    answer = resp.choices[0].message.content.strip()
                    print(f"\n  Assistant: {answer}\n")
                    continue

            # ── Parse what the user said ──
            parsed = parse_user_message(user_input)
            intent = parsed.get("intent", "chat")
            mentioned_meal = parsed.get("meal")
            food_desc = parsed.get("foods")
            preset_name = parsed.get("preset_name")
            sub_action = parsed.get("sub_action")

            # Guardrail: requests about "option menu" are usually about the
            # recent-food disambiguation list, not meal presets.
            option_words = ("option", "menu", "first one", "second one", "third one")
            if (intent == "manage_meal" and not preset_name and
                    any(w in lower_input_check for w in option_words)):
                print(
                    "\n  Assistant: I can't edit that option menu directly. "
                    "It's pulled from your MyFitnessPal recents. "
                    "Tell me the exact food to remove from your diary, like "
                    "\"remove Premier protein bar from breakfast.\"\n"
                )
                continue

            # Guardrail: GPT can misclassify "add X to lunch/dinner" as preset management.
            # If no preset name is provided, treat meal+food phrasing as normal diary logging.
            if intent in ("manage_meal", "log_meal") and not preset_name:
                if mentioned_meal and food_desc:
                    if intent == "manage_meal" and sub_action == "remove_food":
                        intent = "remove"
                    else:
                        intent = "log"

            # Guardrail: "can we log lunch?" can be misclassified as defer.
            # If user asks to log but didn't provide foods yet, prompt for foods.
            if intent == "defer":
                defer_markers = (
                    "later", "not yet", "remind me", "check back", "haven't eaten",
                    "havent eaten", "in an hour", "after", "postpone", "defer",
                )
                asks_to_log = any(k in lower_input_check for k in ("log", "add", "track"))
                is_true_defer = any(k in lower_input_check for k in defer_markers)
                if asks_to_log and not is_true_defer:
                    if mentioned_meal:
                        print(f"\n  Assistant: Sure — what should we log for {mentioned_meal}?\n")
                    else:
                        print("\n  Assistant: Sure — what should we log?\n")
                    continue

            # ── Skip intent ──
            if intent == "skip":
                if mentioned_meal and mentioned_meal in MEAL_NAMES_TO_IDX:
                    print(f"\n  Assistant: Got it, skipping {mentioned_meal.title()}.\n")
                else:
                    print(f"\n  Assistant: No problem, lmk when you're ready.\n")
                continue

            # ── Defer intent (snooze reminder for 1 hour) ──
            if intent == "defer":
                wake_time = datetime.now() + timedelta(hours=1)

                if mentioned_meal and mentioned_meal in MEAL_NAMES_TO_IDX:
                    meal_idx = MEAL_NAMES_TO_IDX[mentioned_meal]
                    snooze_state[meal_idx] = wake_time
                    print(f"\n  Assistant: Got it, I'll check back about {mentioned_meal.lower()} in an hour.\n")
                else:
                    # Snooze all currently unlogged meals
                    try:
                        unlogged = get_unlogged_meals(session)
                    except Exception:
                        unlogged = []
                    if unlogged:
                        for idx, name in unlogged:
                            snooze_state[idx] = wake_time
                        names = ", ".join(n.lower() for _, n in unlogged)
                        print(f"\n  Assistant: Got it, I'll check back about {names} in an hour.\n")
                    else:
                        print(f"\n  Assistant: All meals are logged already.\n")
                continue

            # ── Remove intent ──
            if intent == "remove":
                # "clear my dinner" / "remove everything" without specific food = clear all
                if not food_desc:
                    food_desc = "everything"
                diary, _ = get_diary_details(session, date.today())

                # Collect all logged foods across meals (or specific meal)
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
                    print(f"\n  Assistant: Nothing logged to remove.\n")
                    continue

                # "everything" = remove all from that meal
                if food_desc.lower() in ("everything", "all", "all of it"):
                    meal_name = MEALS[candidates[0]["meal_idx"]]["name"] if mentioned_meal else "all meals"
                    food_names = ", ".join(f["name"] for f in candidates)
                    print(f"\n  Assistant: Remove from {meal_name}: {food_names}? (yes/no)\n")
                    # Store for confirmation
                    pending_meal_idx = candidates[0]["meal_idx"] if mentioned_meal else None
                    pending_foods = candidates
                    awaiting_confirmation = False

                    # Inline yes/no for remove
                    try:
                        confirm = input_provider.get_input(f"  {USER_NAME}: ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        break
                    confirm_words = set(w.strip(".,!?;:'\"") for w in confirm.split())
                    if confirm_words & {"yes", "y", "yeah", "yep", "yup", "sure", "ok", "do", "please"}:
                        removed = 0
                        for f in candidates:
                            if f.get("entry_id") and remove_food(session, f["entry_id"]):
                                removed += 1
                        if removed:
                            invalidate_recents_cache()
                        print(f"\n  Assistant: Done! Removed {removed} item(s).\n")
                    else:
                        print(f"\n  Assistant: No worries, kept everything.\n")
                    continue

                # Use GPT to match which logged food(s) to remove
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
                except (json.JSONDecodeError, IndexError):
                    to_remove = []

                if not to_remove:
                    print(f"\n  Assistant: Couldn't find \"{food_desc}\" in your logged foods.\n")
                    continue

                food_names = ", ".join(f["name"] for f in to_remove)
                print(f"\n  Assistant: Remove: {food_names}? (yes/no)\n")

                try:
                    confirm = input_provider.get_input(f"  {USER_NAME}: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    break
                confirm_words = set(w.strip(".,!?;:'\"") for w in confirm.split())
                if confirm_words & {"yes", "y", "yeah", "yep", "yup", "sure", "ok", "do", "please"}:
                    removed = 0
                    for f in to_remove:
                        if f.get("entry_id") and remove_food(session, f["entry_id"]):
                            removed += 1
                    if removed:
                        invalidate_recents_cache()
                    print(f"\n  Assistant: Done! Removed {removed} item(s).\n")
                else:
                    print(f"\n  Assistant: Kept everything as is.\n")
                continue

            # ── Create meal preset ──
            if intent == "create_meal":
                preset_name = parsed.get("preset_name")
                if not preset_name:
                    print(f"\n  Assistant: What do you want to call this meal preset?\n")
                    try:
                        preset_name = input_provider.get_input(f"  {USER_NAME}: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if not preset_name:
                        print(f"\n  Assistant: Never mind.\n")
                        continue

                # Check if foods were provided in the same message
                food_list_str = food_desc
                if not food_list_str:
                    print(f"\n  Assistant: What foods go in \"{preset_name}\"?\n")
                    try:
                        food_list_str = input_provider.get_input(f"  {USER_NAME}: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if not food_list_str:
                        print(f"\n  Assistant: Never mind.\n")
                        continue

                food_items = split_food_descriptions(food_list_str)
                presets = load_presets()
                presets[preset_name.lower()] = {
                    "foods": [{"search_name": f, "servings": 1} for f in food_items]
                }
                save_presets(presets)

                foods_display = "\n".join(f"    - {f} (1 serving)" for f in food_items)
                print(f"\n  Assistant: Saved \"{preset_name}\" with {len(food_items)} food(s):")
                print(foods_display)
                print()
                continue

            # ── Log a saved meal preset ──
            if intent == "log_meal":
                preset_name = parsed.get("preset_name", "")
                matched_name, preset = get_preset(preset_name) if preset_name else (None, None)

                if not matched_name:
                    # If user says "... meal ..." but it's not one of our local presets,
                    # treat it as an MFP meal search phrase instead of hard-failing.
                    fallback_phrase = (food_desc or preset_name or "").strip()
                    if fallback_phrase and "meal" in lower_input_check:
                        unlogged = get_unlogged_meals(session)
                        if mentioned_meal:
                            meal_idx = resolve_meal_idx(mentioned_meal, unlogged)
                        elif last_meal_idx is not None:
                            meal_idx = last_meal_idx
                        else:
                            meal_idx = resolve_meal_idx(None, unlogged)
                        start_food_matching(fallback_phrase, meal_idx)
                        continue

                    names = get_preset_names()
                    if names:
                        print(f"\n  Assistant: I don't have a preset called \"{preset_name}\".")
                        print(f"  Assistant: Your saved presets: {', '.join(names)}\n")
                    else:
                        print(f"\n  Assistant: You don't have any saved meal presets yet.")
                        print(f"  Assistant: Create one with: \"create a meal called [name]\"\n")
                    continue

                # Determine which MFP meal to log to
                unlogged = get_unlogged_meals(session)
                if mentioned_meal:
                    meal_idx = resolve_meal_idx(mentioned_meal, unlogged)
                elif last_meal_idx is not None:
                    meal_idx = last_meal_idx
                else:
                    meal_idx = resolve_meal_idx(None, unlogged)

                meal_name = MEALS[meal_idx]["name"]
                foods = preset["foods"]
                print(f"\n  Assistant: Logging \"{matched_name}\" to {meal_name} ({len(foods)} foods)...")

                # Get search metadata once for all foods
                search_meta = None
                logged = []
                failed = []

                for item in foods:
                    name = item["search_name"]
                    servings = str(item.get("servings", 1))

                    # First try recent foods
                    all_foods, metadata = get_recent_foods_cached(session, meal_idx)
                    if all_foods:
                        match_results = match_foods_with_gpt(name, all_foods)
                        if match_results and match_results[0].get("matches"):
                            matched_food = all_foods[match_results[0]["matches"][0]]
                            if servings != "1":
                                matched_food["quantity"] = servings
                            if log_foods(session, [matched_food], all_foods, meal_idx, metadata):
                                qty_note = f" (x{servings})" if servings != "1" else ""
                                logged.append(f"{matched_food['name']}{qty_note}")
                                continue

                    # Fall back to search
                    search_query = normalize_food_query(name)
                    search_results, s_meta = search_foods(session, search_query, meal_idx)
                    if not search_meta and s_meta:
                        search_meta = s_meta

                    if search_results:
                        best_match, confidence = match_search_results_with_gpt(search_query, search_results)
                        if best_match and confidence in ("high", "medium"):
                            meta = search_meta or s_meta
                            if meta and log_searched_food(session, best_match, meal_idx, meta, quantity=servings):
                                qty_note = f" (x{servings})" if servings != "1" else ""
                                logged.append(f"{best_match['name']}{qty_note}")
                                continue

                    failed.append(name)

                # Report results
                for name in logged:
                    print(f"    Logged: {name}")
                for name in failed:
                    print(f"    Failed: {name}")

                if logged:
                    invalidate_recents_cache(meal_idx)
                    print(f"  Assistant: Done! Logged {len(logged)}/{len(foods)} foods to {meal_name}.")
                else:
                    print(f"  Assistant: Couldn't log any foods from \"{matched_name}\".")
                if failed:
                    print(f"  Assistant: Couldn't find: {', '.join(failed)}")
                last_meal_idx = meal_idx
                print()
                continue

            # ── Manage meal presets (list, delete, add/remove food) ──
            if intent == "manage_meal":
                sub = parsed.get("sub_action", "list")
                preset_name = parsed.get("preset_name", "")

                if sub == "list":
                    presets = load_presets()
                    if not presets:
                        print(f"\n  Assistant: No saved meal presets yet.")
                        print(f"  Assistant: Create one with: \"create a meal called [name]\"\n")
                    else:
                        print(f"\n  Assistant: Your saved meal presets:")
                        for name, data in presets.items():
                            foods = data.get("foods", [])
                            food_names = ", ".join(
                                f"{f['search_name']}" + (f" (x{f['servings']})" if f.get("servings", 1) != 1 else "")
                                for f in foods
                            )
                            print(f"    - {name}: {food_names}")
                        print()
                    continue

                elif sub == "delete":
                    matched_name, _ = get_preset(preset_name) if preset_name else (None, None)
                    if matched_name:
                        presets = load_presets()
                        del presets[matched_name]
                        save_presets(presets)
                        print(f"\n  Assistant: Deleted \"{matched_name}\".\n")
                    else:
                        show_name = preset_name or "(missing preset name)"
                        print(f"\n  Assistant: No preset found matching \"{show_name}\".\n")
                    continue

                elif sub == "add_food" and food_desc:
                    matched_name, preset = get_preset(preset_name) if preset_name else (None, None)
                    if matched_name and preset:
                        new_foods = split_food_descriptions(food_desc)
                        presets = load_presets()
                        for f in new_foods:
                            presets[matched_name]["foods"].append({"search_name": f, "servings": 1})
                        save_presets(presets)
                        print(f"\n  Assistant: Added {', '.join(new_foods)} to \"{matched_name}\".\n")
                    else:
                        show_name = preset_name or "(missing preset name)"
                        print(f"\n  Assistant: No preset found matching \"{show_name}\".\n")
                    continue

                elif sub == "remove_food" and food_desc:
                    matched_name, preset = get_preset(preset_name) if preset_name else (None, None)
                    if matched_name and preset:
                        presets = load_presets()
                        lower_desc = food_desc.lower()
                        original_count = len(presets[matched_name]["foods"])
                        presets[matched_name]["foods"] = [
                            f for f in presets[matched_name]["foods"]
                            if lower_desc not in f["search_name"].lower()
                        ]
                        removed_count = original_count - len(presets[matched_name]["foods"])
                        if removed_count > 0:
                            save_presets(presets)
                            print(f"\n  Assistant: Removed {removed_count} item(s) matching \"{food_desc}\" from \"{matched_name}\".\n")
                        else:
                            print(f"\n  Assistant: No food matching \"{food_desc}\" found in \"{matched_name}\".\n")
                    else:
                        show_name = preset_name or "(missing preset name)"
                        print(f"\n  Assistant: No preset found matching \"{show_name}\".\n")
                    continue

                continue

            # ── Quick add (custom macros) ──
            if intent == "quick_add":
                calories = parsed.get("calories", 0) or 0
                protein = parsed.get("protein", 0) or 0
                fat = parsed.get("fat", 0) or 0
                carbs = parsed.get("carbs", 0) or 0
                description = food_desc or "Quick Add"

                unlogged = get_unlogged_meals(session)
                if mentioned_meal:
                    meal_idx = resolve_meal_idx(mentioned_meal, unlogged)
                elif last_meal_idx is not None:
                    meal_idx = last_meal_idx
                else:
                    meal_idx = resolve_meal_idx(None, unlogged)

                meal_name = MEALS[meal_idx]["name"]
                print(f"\n  Assistant: Quick adding \"{description}\" to {meal_name}:")
                print(f"    Calories: {calories} | Protein: {protein}g | Fat: {fat}g | Carbs: {carbs}g")

                if quick_add_food(session, meal_idx, description, calories, protein, carbs, fat):
                    print(f"  Assistant: Done! Logged to {meal_name}.\n")
                    last_meal_idx = meal_idx
                else:
                    print(f"  Assistant: Something went wrong with the quick add. Try again.\n")
                continue

            # ── Log intent ──
            if intent == "log" and food_desc:
                unlogged = get_unlogged_meals(session)
                if mentioned_meal:
                    meal_idx = resolve_meal_idx(mentioned_meal, unlogged)
                elif last_meal_idx is not None:
                    # User just canceled and is retrying — keep the same meal
                    meal_idx = last_meal_idx
                else:
                    meal_idx = resolve_meal_idx(None, unlogged)
                start_food_matching(food_desc, meal_idx)
                continue

            # ── Chat — answer questions with diary context ──
            if intent == "chat":
                lower = user_input.lower()
                if any(w in lower for w in ["help", "how do i", "how does"]):
                    print(f"\n  Assistant: Just tell me what you ate, like \"chicken and rice for dinner\".\n")
                    continue

                context = build_diary_context(session)
                reply = smart_reply(user_input, context)
                print(f"\n  Assistant: {reply}\n")
                continue
    finally:
        reminder_stop.set()
        summary_stop.set()
        time.sleep(0.05)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diet Assistant for MyFitnessPal")
    parser.add_argument("--voice", action="store_true", help="Speak Assistant messages through local speakers")
    parser.add_argument("--voice-name", type=str, default=None, help="Optional TTS voice name (e.g. 'Samantha')")
    parser.add_argument(
        "--voice-volume",
        type=int,
        default=int(os.getenv("VOICE_VOLUME", "65")),
        help="Assistant speech volume 0-100 (default 65)",
    )
    parser.add_argument(
        "--reminder-times",
        type=str,
        default=os.getenv("REMINDER_TIMES", "09:30,13:30,19:30"),
        help="Comma-separated HH:MM reminders when meals are unlogged",
    )
    parser.add_argument("--no-reminders", action="store_true", help="Disable timed reminders")
    parser.add_argument(
        "--summary-time",
        type=str,
        default=os.getenv("SUMMARY_TIME", "23:00"),
        help="HH:MM time to text daily diet summary (default 23:00)",
    )
    parser.add_argument("--no-summary", action="store_true", help="Disable nightly summary text")
    parser.add_argument("--test-sms", action="store_true", help="Send a test SMS and exit (to verify Twilio config)")
    args = parser.parse_args()

    if args.test_sms:
        print("Building daily summary from MFP...")
        session = create_session()
        msg = build_daily_summary(session)
        if not msg:
            print("No diary data to summarize today.")
            sys.exit(1)
        print(f"Sending summary:\n{msg}\n")
        ok = send_sms(msg)
        if ok:
            print("Summary sent successfully! Check your phone.")
        else:
            print("Failed to send summary.")
        sys.exit(0)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY in .env!")
        print("Add it like: OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # Hybrid input: type OR hold Option key to speak
    try:
        from voice_io import HybridInputProvider
        input_provider = HybridInputProvider()
    except (ImportError, OSError):
        # Fallback to text-only if pynput/whisper not installed
        from voice_io import TextInputProvider
        input_provider = TextInputProvider()
    io_lock = None

    # Speaker: always enabled at low volume for voice responses only.
    # When input comes from Option key, the interceptor speaks the response.
    # When input is typed, it stays silent.
    voice_volume = max(0, min(100, int(args.voice_volume)))
    speaker = Speaker(enabled=True, voice_name=args.voice_name, default_volume=voice_volume)
    original_stdout = sys.stdout
    sys.stdout = AssistantSpeechInterceptor(
        sys.stdout, speaker, input_provider=input_provider
    )

    reminder_times = [] if args.no_reminders else parse_reminder_times(args.reminder_times)
    if reminder_times:
        print(f"Reminder times active: {', '.join(reminder_times)}")

    summary_time = None if args.no_summary else args.summary_time
    if summary_time:
        print(f"Nightly summary text at: {summary_time}")

    try:
        run_terminal_chat(
            reminder_times=reminder_times,
            summary_time=summary_time,
            input_provider=input_provider,
            io_lock=io_lock,
            speaker=speaker,
        )
    finally:
        input_provider.cleanup()
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()
