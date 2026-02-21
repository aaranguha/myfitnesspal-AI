"""
Stateless assistant core for Alexa Lambda (and potentially other serverless environments).

Extracts the conversation logic from assistant.py into a single
`process_input(user_input, state) -> (response, new_state, should_end)` interface.
Each invocation is independent — all state is passed in and returned as a dict.
"""

import json
import os
import re
import time
from datetime import date, datetime

from openai import OpenAI

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
    quick_add_food,
    load_presets,
    save_presets,
    get_preset_names,
    get_preset,
)

# ── Constants ────────────────────────────────────────────────────────────

MEAL_NAMES_TO_IDX = {info["name"].lower(): idx for idx, info in MEALS.items()}

HOUSEHOLD_UNIT_TO_SERVINGS = {
    "katori": 0.75, "katoris": 0.75,
    "bowl": 1.0, "bowls": 1.0,
    "plate": 1.5, "plates": 1.5,
}


# ── Helper functions (ported from assistant.py) ──────────────────────────

def _num_or_none(value):
    try:
        if value is None:
            return None
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def split_food_descriptions(food_description):
    parts = re.split(r",| and | with |\+", food_description, flags=re.IGNORECASE)
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return cleaned or [food_description.strip()]


def normalize_food_query(food_text):
    q = food_text.strip().lower()
    q = re.sub(r"^(can you )?(just )?(log|add)\s+", "", q)
    q = re.sub(r"^(a|an|the|some)\s+", "", q)
    q = re.sub(r"^(regular|normal|standard)\s+(serving|portion)\s+of\s+", "", q)
    q = re.sub(r"^(serving|portion)\s+of\s+", "", q)
    q = re.sub(r"\s+", " ", q).strip(" .")
    return q or food_text.strip()


def split_compound_food_for_search(client, food_text):
    raw = normalize_food_query(food_text)
    if not raw:
        return []
    quick_parts = re.split(r",| and | with |&|\/|\+", raw, flags=re.IGNORECASE)
    quick_parts = [p.strip() for p in quick_parts if p and p.strip()]
    if len(quick_parts) > 1:
        return quick_parts
    # GPT fallback
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0,
            messages=[
                {"role": "system", "content": (
                    "Split a food phrase into individual food items for diary logging.\n"
                    "Return ONLY a JSON array of short strings.\n"
                    'Examples:\n- "channay chawal" -> ["channay", "rice"]\n'
                    '- "protein bar" -> ["protein bar"]\n'
                    "If it is one item, return one string in the array."
                )},
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


def _tokenize_food_text(text):
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    stop = {"a", "an", "the", "my", "to", "for", "with", "and", "or", "of",
            "please", "log", "add", "meal", "breakfast", "lunch", "dinner", "snacks"}
    return {t for t in tokens if t not in stop}


def has_reasonable_recent_overlap(description, candidate_names):
    desc_tokens = _tokenize_food_text(description)
    if not desc_tokens:
        return True
    for name in candidate_names:
        if desc_tokens & _tokenize_food_text(name):
            return True
    return False


def pick_reasonable_search_result(search_query, search_results):
    q = search_query.lower().strip()
    best, best_score = None, -10**9
    for item in search_results:
        name = item.get("name", "").strip()
        if not name:
            continue
        lower = name.lower()
        score = 0
        if lower == q: score += 8
        if lower.startswith(q): score += 5
        if q in lower: score += 3
        if " - " in name: score -= 3
        if any(tag in lower for tag in ("raw", "cooked", "steamed", "roasted")): score += 1
        if score > best_score:
            best_score = score
            best = item
    return best


def parse_household_quantity_to_servings(text):
    t = (text or "").lower().strip()
    if not t:
        return None
    unit = None
    for u in HOUSEHOLD_UNIT_TO_SERVINGS:
        if re.search(rf"\b{re.escape(u)}\b", t):
            unit = u
            break
    if not unit:
        return None
    base = HOUSEHOLD_UNIT_TO_SERVINGS[unit]
    if re.search(r"\bhalf\b", t): amount = 0.5
    elif re.search(r"\bquarter\b", t): amount = 0.25
    elif re.search(r"\bdouble\b", t): amount = 2.0
    elif re.search(r"\btriple\b", t): amount = 3.0
    else:
        m_range = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", t)
        if m_range:
            amount = (float(m_range.group(1)) + float(m_range.group(2))) / 2.0
        else:
            m_num = re.search(r"(\d+(?:\.\d+)?)", t)
            amount = float(m_num.group(1)) if m_num else 1.0
    servings = amount * base
    if servings <= 0:
        return None
    return round(servings, 2), unit, amount, base


def parse_nutrition_from_cal_info(cal_info):
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


# ── GPT helpers ──────────────────────────────────────────────────────────

def parse_user_message(client, user_message):
    preset_names = get_preset_names()
    presets_hint = ""
    if preset_names:
        presets_hint = f"\nThe user has these saved meal presets: {', '.join(preset_names)}\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[
            {"role": "system", "content": (
                "You parse casual messages about food logging. Extract the meal type and food description.\n"
                f"{presets_hint}\n"
                "Return ONLY a JSON object with these fields:\n"
                '- "meal": one of "breakfast", "lunch", "dinner", "snacks", or null if not mentioned\n'
                '- "foods": the food description extracted from their message, or null\n'
                '- "intent": one of: "log", "remove", "skip", "create_meal", "log_meal", "manage_meal", "quick_add", "defer", "chat"\n'
                '- "preset_name": the meal preset name, or null\n'
                '- "sub_action": for manage_meal only — "list", "delete", "add_food", "remove_food", or null\n'
                '- "calories": number or null (for quick_add)\n'
                '- "protein": number or null (for quick_add)\n'
                '- "fat": number or null (for quick_add)\n'
                '- "carbs": number or null (for quick_add)\n\n'
                "Examples:\n"
                '- "had a protein bar and chai for breakfast" → {"meal": "breakfast", "foods": "protein bar and chai", "intent": "log", "preset_name": null, "sub_action": null}\n'
                '- "remove the chicken from dinner" → {"meal": "dinner", "foods": "chicken", "intent": "remove", "preset_name": null, "sub_action": null}\n'
                '- "how many calories today" → {"meal": null, "foods": null, "intent": "chat", "preset_name": null, "sub_action": null}\n'
                '- "quick add 500 cals 30g protein to lunch" → {"meal": "lunch", "foods": "Quick Add", "intent": "quick_add", "calories": 500, "protein": 30, "fat": null, "carbs": null}\n'
                '- "I\'ll eat later" → {"meal": null, "foods": null, "intent": "defer", "preset_name": null, "sub_action": null}\n'
                "\nIMPORTANT:\n"
                "- Questions about what's LOGGED/in the diary = chat, NOT manage_meal.\n"
                "- manage_meal is ONLY for meal PRESETS (saved combos).\n"
                "- If the user provides SPECIFIC calorie/macro numbers, use quick_add.\n"
                "Return ONLY the JSON object, nothing else."
            )},
            {"role": "user", "content": user_message},
        ],
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"meal": None, "foods": None, "intent": "chat"}


def match_foods_with_gpt(client, food_description, food_list):
    food_names = [f"{i}: {f['name']}" for i, f in enumerate(food_list)]
    food_list_str = "\n".join(food_names)

    response = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[
            {"role": "system", "content": (
                "You are a food-matching assistant. Break the user's message into INDIVIDUAL food items, "
                "then find the best match from the numbered list.\n\n"
                'Return a JSON array: [{"description": "what they said", "matches": [index_numbers]}]\n\n'
                "Rules:\n"
                "- ONE clear match → [5]\n"
                "- MULTIPLE similar → [5, 18] (user will pick)\n"
                "- No match → []\n"
                "- Match loosely\n"
                "- Return ONLY the JSON array"
            )},
            {"role": "user", "content": f"I ate: {food_description}\n\nAvailable foods:\n{food_list_str}"},
        ],
    )
    raw = response.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            for item in result:
                if "matches" in item:
                    item["matches"] = [i for i in item["matches"]
                                       if isinstance(i, int) and 0 <= i < len(food_list)]
            return result
    except json.JSONDecodeError:
        pass
    return []


def match_search_results_with_gpt(client, food_description, search_results):
    food_names = [
        f"{i}: {f['name']}" + (f" ({f['cal_info']})" if f.get("cal_info") else "")
        for i, f in enumerate(search_results)
    ]
    food_list_str = "\n".join(food_names)

    response = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[
            {"role": "system", "content": (
                "Pick the SINGLE best match from MFP search results.\n"
                'Return ONLY: {"match": index_number, "confidence": "high"|"medium"|"low"}\n'
                'If nothing matches: {"match": null, "confidence": "none"}'
            )},
            {"role": "user", "content": f"Looking for: {food_description}\n\nSearch results:\n{food_list_str}"},
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


def smart_reply(client, user_input, diary_context, user_name="Aaran"):
    response = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0.5,
        messages=[
            {"role": "system", "content": (
                f"You are a casual, chill diet assistant for {user_name}. "
                f"Answer based on the diary data below. "
                f"Be super short and casual — 1 sentence max, like texting a friend. "
                f"No filler, no pleasantries. Just the answer. "
                f"When listing macros, always use this order: calories, protein, fat, carbs.\n\n"
                f"{diary_context}"
            )},
            {"role": "user", "content": user_input},
        ],
    )
    return response.choices[0].message.content.strip()


def classify_confirmation(client, user_input, meal_name, pending_info, pending_names):
    """Classify user's response during confirmation flow."""
    response = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[
            {"role": "system", "content": (
                "You are a food logging assistant. The user was asked to confirm logging food.\n"
                f"Pending food(s) to log to {meal_name}:\n{pending_info}\n\n"
                "Classify the user's response into ONE action:\n"
                '1. "confirm" — yes/ok/sure/log it\n'
                '2. "reject" — no/cancel/wrong (may include replacement food)\n'
                '3. "adjust_qty" — same food, different servings\n'
                '4. "correction" — different brand/entry\n'
                '5. "add_more" — confirm + add more foods\n'
                '6. "question" — asking about pending food\n\n'
                "Return ONLY JSON:\n"
                "{\n"
                '  "action": "confirm"|"reject"|"adjust_qty"|"correction"|"add_more"|"question",\n'
                '  "replacement_food": "food name or null",\n'
                '  "quantity": number or null,\n'
                '  "search_term": "brand/food name or null",\n'
                '  "extra_foods": "additional foods or null"\n'
                "}"
            )},
            {"role": "user", "content": user_input},
        ],
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"action": "question"}


# ── Meal helpers ─────────────────────────────────────────────────────────

def get_unlogged_meals(session):
    logged = check_meals_logged(session, date.today())
    return [(idx, info["name"]) for idx, info in MEALS.items() if not logged.get(idx, False)]


def guess_meal_from_time():
    hour = datetime.now().hour
    if hour < 11: return 0
    elif hour < 18: return 1
    else: return 2


def resolve_meal_idx(mentioned_meal, unlogged):
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


# ── Recents cache (persists across warm Lambda invocations) ──────────────

_recents_cache = {}
RECENTS_TTL = 120


def get_recent_foods_cached(session, meal_idx):
    now = time.time()
    entry = _recents_cache.get(meal_idx)
    if entry and (now - entry["ts"]) < RECENTS_TTL:
        return entry["foods"], entry["metadata"]
    foods, metadata = get_recent_foods(session, meal_idx)
    _recents_cache[meal_idx] = {"foods": foods, "metadata": metadata, "ts": now}
    return foods, metadata


def invalidate_recents_cache(meal_idx=None):
    if meal_idx is not None:
        _recents_cache.pop(meal_idx, None)
    else:
        _recents_cache.clear()


# ── AssistantCore ────────────────────────────────────────────────────────

def empty_state():
    """Return a fresh conversation state dict."""
    return {
        "pending_meal_idx": None,
        "last_meal_idx": None,
        "pending_foods": [],
        "pending_searched_foods": [],
        "pending_search_metadata": None,
        "pending_quantity": "1",
        "pending_item_quantities": {},
        "pending_all_foods": None,
        "pending_metadata": None,
        "disambiguation_queue": [],
        "not_found": [],
        "awaiting_confirmation": False,
        "awaiting_pick": False,
        "pick_options": [],
        "awaiting_remove_confirm": False,
        "remove_candidates": [],
    }


class AssistantCore:
    """Stateless conversation engine for the diet assistant.

    Each call to process_input() takes the current state and returns
    (response_text, new_state, should_end_session).
    """

    def __init__(self, mfp_cookie=None, openai_key=None, user_name="Aaran"):
        self.user_name = user_name
        self.client = OpenAI(api_key=openai_key or os.getenv("OPENAI_API_KEY"))
        self.session = create_session(cookie_override=mfp_cookie)

    def process_input(self, user_input, state=None):
        """Process one turn of conversation.

        Returns: (response_text: str, new_state: dict, should_end: bool)
        """
        if state is None:
            state = empty_state()
        # Make a mutable copy
        state = dict(state)

        user_input = (user_input or "").strip()
        if not user_input:
            return "I didn't catch that.", state, False

        # ── Disambiguation pick ──
        if state.get("awaiting_pick") and state.get("pick_options"):
            return self._handle_pick(user_input, state)

        # ── Remove confirmation ──
        if state.get("awaiting_remove_confirm"):
            return self._handle_remove_confirm(user_input, state)

        # ── Confirmation flow ──
        if state.get("awaiting_confirmation"):
            return self._handle_confirmation(user_input, state)

        # ── Parse new intent ──
        parsed = parse_user_message(self.client, user_input)
        intent = parsed.get("intent", "chat")
        mentioned_meal = parsed.get("meal")
        food_desc = parsed.get("foods")
        preset_name = parsed.get("preset_name")
        sub_action = parsed.get("sub_action")

        # Guardrail: misclassified preset management
        if intent in ("manage_meal", "log_meal") and not preset_name:
            if mentioned_meal and food_desc:
                intent = "remove" if (intent == "manage_meal" and sub_action == "remove_food") else "log"

        if intent == "skip":
            return self._handle_skip(mentioned_meal, state)
        elif intent == "defer":
            return self._handle_defer(mentioned_meal, state)
        elif intent == "remove":
            return self._handle_remove(food_desc, mentioned_meal, state)
        elif intent == "quick_add":
            return self._handle_quick_add(parsed, mentioned_meal, food_desc, state)
        elif intent == "log":
            return self._handle_log(food_desc, mentioned_meal, state)
        elif intent == "log_meal":
            return self._handle_log_meal(preset_name, mentioned_meal, state)
        elif intent == "manage_meal":
            return self._handle_manage_meal(sub_action, preset_name, food_desc, state)
        elif intent == "create_meal":
            return self._handle_create_meal(preset_name, food_desc, state)
        else:  # chat
            return self._handle_chat(user_input, state)

    # ── Intent handlers ──────────────────────────────────────────────────

    def _handle_skip(self, mentioned_meal, state):
        if mentioned_meal and mentioned_meal in MEAL_NAMES_TO_IDX:
            return f"Got it, skipping {mentioned_meal.title()}.", state, False
        return "No problem, lmk when you're ready.", state, False

    def _handle_defer(self, mentioned_meal, state):
        if mentioned_meal and mentioned_meal in MEAL_NAMES_TO_IDX:
            return f"Got it, I'll check back about {mentioned_meal.lower()} later.", state, False
        return "Got it, just say when you're ready.", state, False

    def _handle_chat(self, user_input, state):
        lower = user_input.lower()
        if any(w in lower for w in ["help", "how do i", "how does"]):
            return "Just tell me what you ate, like 'chicken and rice for dinner'.", state, False
        context = build_diary_context(self.session)
        reply = smart_reply(self.client, user_input, context, self.user_name)
        return reply, state, False

    def _handle_quick_add(self, parsed, mentioned_meal, food_desc, state):
        calories = parsed.get("calories", 0) or 0
        protein = parsed.get("protein", 0) or 0
        fat = parsed.get("fat", 0) or 0
        carbs = parsed.get("carbs", 0) or 0
        description = food_desc or "Quick Add"

        unlogged = get_unlogged_meals(self.session)
        meal_idx = self._resolve_meal(mentioned_meal, state.get("last_meal_idx"), unlogged)
        meal_name = MEALS[meal_idx]["name"]

        if quick_add_food(self.session, meal_idx, description, calories, protein, carbs, fat):
            state["last_meal_idx"] = meal_idx
            return (
                f"Quick added \"{description}\" to {meal_name}: "
                f"{calories} cal, {protein}g protein, {fat}g fat, {carbs}g carbs.",
                state, False
            )
        return "Something went wrong with the quick add.", state, False

    def _handle_log(self, food_desc, mentioned_meal, state):
        if not food_desc:
            return "What did you eat?", state, False

        unlogged = get_unlogged_meals(self.session)
        meal_idx = self._resolve_meal(mentioned_meal, state.get("last_meal_idx"), unlogged)
        state["last_meal_idx"] = meal_idx

        return self._start_food_matching(food_desc, meal_idx, state)

    def _handle_log_meal(self, preset_name, mentioned_meal, state):
        matched_name, preset = get_preset(preset_name) if preset_name else (None, None)
        if not matched_name:
            names = get_preset_names()
            if names:
                return f"I don't have a preset called \"{preset_name}\". Your presets: {', '.join(names)}", state, False
            return "You don't have any saved meal presets. Create one first.", state, False

        unlogged = get_unlogged_meals(self.session)
        meal_idx = self._resolve_meal(mentioned_meal, state.get("last_meal_idx"), unlogged)
        meal_name = MEALS[meal_idx]["name"]
        foods = preset["foods"]

        logged, failed = [], []
        for item in foods:
            name = item["search_name"]
            servings = str(item.get("servings", 1))

            # Try recent foods first
            all_foods, metadata = get_recent_foods_cached(self.session, meal_idx)
            if all_foods:
                match_results = match_foods_with_gpt(self.client, name, all_foods)
                if match_results and match_results[0].get("matches"):
                    matched_food = all_foods[match_results[0]["matches"][0]]
                    if servings != "1":
                        matched_food["quantity"] = servings
                    if log_foods(self.session, [matched_food], all_foods, meal_idx, metadata):
                        logged.append(matched_food["name"])
                        continue

            # Fall back to search
            search_query = normalize_food_query(name)
            search_results, s_meta = search_foods(self.session, search_query, meal_idx)
            if search_results:
                best_match, confidence = match_search_results_with_gpt(self.client, search_query, search_results)
                if best_match and confidence in ("high", "medium"):
                    if log_searched_food(self.session, best_match, meal_idx, s_meta, quantity=servings):
                        logged.append(best_match["name"])
                        continue
            failed.append(name)

        if logged:
            invalidate_recents_cache(meal_idx)

        parts = []
        if logged:
            parts.append(f"Logged {len(logged)}/{len(foods)} foods to {meal_name}.")
        if failed:
            parts.append(f"Couldn't find: {', '.join(failed)}")
        state["last_meal_idx"] = meal_idx
        return " ".join(parts) or "Couldn't log any foods.", state, False

    def _handle_manage_meal(self, sub_action, preset_name, food_desc, state):
        sub = sub_action or "list"

        if sub == "list":
            presets = load_presets()
            if not presets:
                return "No saved meal presets yet. Create one with 'create a meal called [name]'.", state, False
            lines = []
            for name, data in presets.items():
                foods = data.get("foods", [])
                food_names = ", ".join(f["search_name"] for f in foods)
                lines.append(f"{name}: {food_names}")
            return "Your meal presets: " + "; ".join(lines), state, False

        elif sub == "delete":
            matched_name, _ = get_preset(preset_name) if preset_name else (None, None)
            if matched_name:
                presets = load_presets()
                del presets[matched_name]
                save_presets(presets)
                return f"Deleted \"{matched_name}\".", state, False
            return f"No preset found matching \"{preset_name or ''}\".", state, False

        elif sub == "add_food" and food_desc:
            matched_name, preset = get_preset(preset_name) if preset_name else (None, None)
            if matched_name and preset:
                new_foods = split_food_descriptions(food_desc)
                presets = load_presets()
                for f in new_foods:
                    presets[matched_name]["foods"].append({"search_name": f, "servings": 1})
                save_presets(presets)
                return f"Added {', '.join(new_foods)} to \"{matched_name}\".", state, False
            return f"No preset found matching \"{preset_name or ''}\".", state, False

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
                    return f"Removed {removed_count} item(s) matching \"{food_desc}\" from \"{matched_name}\".", state, False
                return f"No food matching \"{food_desc}\" found in \"{matched_name}\".", state, False
            return f"No preset found matching \"{preset_name or ''}\".", state, False

        return "Not sure what you want to do with meal presets.", state, False

    def _handle_create_meal(self, preset_name, food_desc, state):
        if not preset_name:
            return "What do you want to call this meal preset?", state, False
        if not food_desc:
            return f"What foods go in \"{preset_name}\"?", state, False

        food_items = split_food_descriptions(food_desc)
        presets = load_presets()
        presets[preset_name.lower()] = {
            "foods": [{"search_name": f, "servings": 1} for f in food_items]
        }
        save_presets(presets)
        return f"Saved \"{preset_name}\" with {len(food_items)} food(s): {', '.join(food_items)}", state, False

    def _handle_remove(self, food_desc, mentioned_meal, state):
        if not food_desc:
            food_desc = "everything"

        diary, _ = get_diary_details(self.session, date.today())

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
            return "Nothing logged to remove.", state, False

        # "everything" = remove all
        if food_desc.lower() in ("everything", "all", "all of it"):
            meal_name = MEALS[candidates[0]["meal_idx"]]["name"] if mentioned_meal else "all meals"
            food_names = ", ".join(f["name"] for f in candidates)
            state["awaiting_remove_confirm"] = True
            state["remove_candidates"] = candidates
            return f"Remove from {meal_name}: {food_names}? Say yes or no.", state, False

        # GPT match specific food
        food_names_list = [f"{i}: {f['name']}" for i, f in enumerate(candidates)]
        response = self.client.chat.completions.create(
            model="gpt-4o-mini", temperature=0,
            messages=[
                {"role": "system", "content": (
                    "Match what the user wants to REMOVE to the numbered list of logged foods. "
                    "Return ONLY a JSON array of matching index numbers. Return [] if no match."
                )},
                {"role": "user", "content": f"Remove: {food_desc}\n\nLogged foods:\n" + "\n".join(food_names_list)},
            ],
        )
        raw = response.choices[0].message.content.strip()
        try:
            indices = json.loads(raw)
            to_remove = [candidates[i] for i in indices if isinstance(i, int) and 0 <= i < len(candidates)]
        except (json.JSONDecodeError, IndexError):
            to_remove = []

        if not to_remove:
            return f"Couldn't find \"{food_desc}\" in your logged foods.", state, False

        food_names = ", ".join(f["name"] for f in to_remove)
        state["awaiting_remove_confirm"] = True
        state["remove_candidates"] = to_remove
        return f"Remove: {food_names}? Say yes or no.", state, False

    def _handle_remove_confirm(self, user_input, state):
        state["awaiting_remove_confirm"] = False
        candidates = state.pop("remove_candidates", [])

        lower = user_input.lower().strip()
        confirm_words = set(w.strip(".,!?;:'\"") for w in lower.split())
        if confirm_words & {"yes", "y", "yeah", "yep", "sure", "ok", "please"}:
            removed = 0
            for f in candidates:
                if f.get("entry_id") and remove_food(self.session, f["entry_id"]):
                    removed += 1
            if removed:
                invalidate_recents_cache()
            return f"Done! Removed {removed} item(s).", state, False
        return "No worries, kept everything.", state, False

    # ── Food matching ────────────────────────────────────────────────────

    def _start_food_matching(self, food_desc, meal_idx, state):
        """Match foods against MFP recents + search fallback. Returns response."""
        meal_name = MEALS[meal_idx]["name"]
        all_foods, metadata = get_recent_foods_cached(self.session, meal_idx)

        state["pending_meal_idx"] = meal_idx
        state["pending_all_foods"] = all_foods
        state["pending_metadata"] = metadata
        state["pending_foods"] = []
        state["pending_searched_foods"] = []
        state["disambiguation_queue"] = []
        state["not_found"] = []

        if all_foods:
            match_results = match_foods_with_gpt(self.client, food_desc, all_foods)
            for item in match_results:
                matches = item.get("matches", [])
                desc = item.get("description", "")
                if len(matches) == 1:
                    candidate = all_foods[matches[0]]
                    if has_reasonable_recent_overlap(desc, [candidate["name"]]):
                        state["pending_foods"].append(candidate)
                    else:
                        state["not_found"].append(desc)
                elif len(matches) > 1:
                    options = [all_foods[i] for i in matches]
                    if has_reasonable_recent_overlap(desc, [o["name"] for o in options]):
                        state["disambiguation_queue"].append({"description": desc, "options": options})
                    else:
                        state["not_found"].append(desc)
                else:
                    state["not_found"].append(desc)
            if not match_results:
                state["not_found"].extend(split_food_descriptions(food_desc))
        else:
            state["not_found"].extend(split_food_descriptions(food_desc))

        # Search fallback for not-found items
        still_not_found = []
        for desc in state["not_found"]:
            parts = split_compound_food_for_search(self.client, desc)
            unresolved = []
            for part in parts:
                search_query = normalize_food_query(part)
                search_results, search_meta = search_foods(self.session, search_query, meal_idx)
                if search_meta and not state.get("pending_search_metadata"):
                    state["pending_search_metadata"] = search_meta
                if search_results:
                    best_match, confidence = match_search_results_with_gpt(self.client, search_query, search_results)
                    if best_match and confidence in ("high", "medium"):
                        state["pending_searched_foods"].append(best_match)
                    else:
                        fallback = pick_reasonable_search_result(search_query, search_results)
                        if fallback:
                            state["pending_searched_foods"].append(fallback)
                        else:
                            unresolved.append(part)
                else:
                    unresolved.append(part)
            still_not_found.extend(unresolved)
        state["not_found"] = still_not_found

        # Process disambiguation or go to confirmation
        return self._process_next_disambiguation(state)

    def _process_next_disambiguation(self, state):
        """Pop next ambiguous item or proceed to confirmation."""
        if state["disambiguation_queue"]:
            item = state["disambiguation_queue"].pop(0)
            options = item["options"]
            desc = item["description"]
            state["awaiting_pick"] = True
            state["pick_options"] = options
            option_list = ". ".join(f"{i+1}, {opt['name']}" for i, opt in enumerate(options))
            return f"Which \"{desc}\" did you mean? {option_list}", state, False

        return self._show_confirmation(state)

    def _show_confirmation(self, state):
        """Auto-log recents or ask for quantity on searched foods."""
        pending_foods = state.get("pending_foods", [])
        pending_searched = state.get("pending_searched_foods", [])
        not_found = state.get("not_found", [])
        meal_idx = state["pending_meal_idx"]
        meal_name = MEALS[meal_idx]["name"]

        if not pending_foods and not pending_searched:
            if not_found:
                resp = f"Couldn't find: {', '.join(not_found)}"
            else:
                resp = "Couldn't find any matching foods."
            state.update(empty_state())
            state["last_meal_idx"] = meal_idx
            return resp, state, False

        not_found_msg = ""
        if not_found:
            not_found_msg = f"Couldn't find: {', '.join(not_found)}. "

        # Auto-log if all from recents
        if pending_foods and not pending_searched:
            success, _, logged_names, failed_names = self._commit_logs(state)
            count = len(pending_foods)
            last_meal = meal_idx
            state.update(empty_state())
            state["last_meal_idx"] = last_meal
            if logged_names:
                msg = f"{not_found_msg}Got it, logged {count} item(s) to {meal_name}."
                if failed_names:
                    msg += f" Couldn't log: {', '.join(failed_names)}"
                return msg, state, False
            return f"{not_found_msg}Something went wrong logging those.", state, False

        # Searched foods need confirmation
        count = len(pending_foods) + len(pending_searched)
        searched_names = ", ".join(f["name"] for f in pending_searched)
        state["awaiting_confirmation"] = True
        return (
            f"{not_found_msg}Found {count} item(s) for {meal_name} "
            f"including: {searched_names}. Log them? Say yes, or specify a quantity.",
            state, False
        )

    def _handle_pick(self, user_input, state):
        """Handle disambiguation pick (user says a number)."""
        pick_options = state.get("pick_options", [])

        # Try to parse as a number
        try:
            choice = int(user_input.strip()) - 1
            if 0 <= choice < len(pick_options):
                chosen = pick_options[choice]
                state["pending_foods"].append(chosen)
                state["awaiting_pick"] = False
                state["pick_options"] = []
                # Process next disambiguation or go to confirmation
                resp_text, state, should_end = self._process_next_disambiguation(state)
                return f"Got it, {chosen['name']}. {resp_text}", state, should_end
        except ValueError:
            pass

        # Try GPT to match by name
        for i, opt in enumerate(pick_options):
            if user_input.lower().strip() in opt["name"].lower():
                chosen = pick_options[i]
                state["pending_foods"].append(chosen)
                state["awaiting_pick"] = False
                state["pick_options"] = []
                resp_text, state, should_end = self._process_next_disambiguation(state)
                return f"Got it, {chosen['name']}. {resp_text}", state, should_end

        state["awaiting_pick"] = False
        state["pick_options"] = []
        return f"Pick a number between 1 and {len(pick_options)}.", state, False

    def _handle_confirmation(self, user_input, state):
        """Handle the confirmation flow for searched/unfamiliar foods."""
        meal_idx = state["pending_meal_idx"]
        meal_name = MEALS[meal_idx]["name"]

        # Check for household quantity
        qty_parse = parse_household_quantity_to_servings(user_input)
        if qty_parse:
            servings, unit, amount, base = qty_parse
            state["pending_quantity"] = str(servings)
            return (
                f"Got it, {amount:g} {unit} (~{base:g} serving each) = {servings:g} serving(s). "
                "Go ahead?",
                state, False
            )

        # Build pending info for GPT context
        pending_names = ([f["name"] for f in state.get("pending_foods", [])] +
                        [f["name"] for f in state.get("pending_searched_foods", [])])
        pending_info = ", ".join(pending_names) or "unknown"

        # Classify the response
        decision = classify_confirmation(self.client, user_input, meal_name, pending_info, pending_names)
        action = decision.get("action", "question")

        if action == "confirm":
            success, _, logged_names, failed_names = self._commit_logs(state)
            last_meal = meal_idx
            state.update(empty_state())
            state["last_meal_idx"] = last_meal
            if logged_names:
                msg = f"Done, logged to {meal_name}."
                if failed_names:
                    msg += f" Couldn't log: {', '.join(failed_names)}"
                return msg, state, False
            return "Something went wrong. Check MFP.", state, False

        elif action == "reject":
            replacement = decision.get("replacement_food")
            saved_meal = meal_idx
            state.update(empty_state())
            state["last_meal_idx"] = saved_meal
            if replacement:
                return self._start_food_matching(replacement, saved_meal, state)
            return "Ok, what'd you have instead?", state, False

        elif action == "adjust_qty":
            qty = decision.get("quantity")
            if qty and qty > 0:
                state["pending_quantity"] = str(qty)
                food_name = pending_names[0] if pending_names else "food"
                qty_display = int(qty) if qty == int(qty) else qty
                return f"Got it, {qty_display} serving(s) of {food_name}. Go ahead?", state, False
            return "Didn't catch that, how many servings?", state, False

        elif action == "add_more":
            extra_food = decision.get("extra_foods")
            if extra_food:
                _, _, logged_names, failed_names = self._commit_logs(state)
                saved_meal = meal_idx
                state.update(empty_state())
                state["last_meal_idx"] = saved_meal
                msg_parts = []
                if logged_names:
                    msg_parts.append(f"Logged: {', '.join(logged_names)}.")
                if failed_names:
                    msg_parts.append(f"Couldn't log: {', '.join(failed_names)}.")
                # Start matching the extra food
                resp, state, should_end = self._start_food_matching(extra_food, saved_meal, state)
                prefix = " ".join(msg_parts) + " " if msg_parts else ""
                return prefix + resp, state, should_end
            return "What else do you want to add?", state, False

        elif action == "correction":
            search_term = decision.get("search_term")
            search_name = search_term or (pending_names[0] if pending_names else "food")
            search_query = normalize_food_query(search_name)

            search_results, search_meta = search_foods(self.session, search_query, meal_idx)
            if search_results:
                best_match, confidence = match_search_results_with_gpt(self.client, search_query, search_results)
                if best_match and confidence in ("high", "medium"):
                    state["pending_foods"] = []
                    state["pending_searched_foods"] = [best_match]
                    if search_meta:
                        state["pending_search_metadata"] = search_meta
                    cal_display = f" ({best_match['cal_info']})" if best_match.get("cal_info") else ""
                    return f"Found: {best_match['name']}{cal_display}. Go ahead?", state, False
            return f"No results for \"{search_query}\". Say yes to log current, or no to cancel.", state, False

        else:  # question
            return f"You have {', '.join(pending_names)} pending for {meal_name}. Say yes to log or no to cancel.", state, False

    # ── Commit logs ──────────────────────────────────────────────────────

    def _commit_logs(self, state):
        """Execute the actual logging to MFP. Returns (success, count, logged_names, failed_names)."""
        pending_foods = state.get("pending_foods", [])
        pending_searched = state.get("pending_searched_foods", [])
        pending_quantity = state.get("pending_quantity", "1")
        pending_item_quantities = state.get("pending_item_quantities", {})
        pending_all_foods = state.get("pending_all_foods")
        pending_metadata = state.get("pending_metadata")
        pending_search_metadata = state.get("pending_search_metadata")
        meal_idx = state["pending_meal_idx"]

        user_set_quantity = (pending_quantity != "1" or bool(pending_item_quantities))

        def quantity_for_food(food_name, default_qty=None):
            for k, v in pending_item_quantities.items():
                if k.lower() in food_name.lower() or food_name.lower() in k.lower():
                    return float(v)
            if user_set_quantity:
                try:
                    return float(pending_quantity)
                except (ValueError, TypeError):
                    return 1.0
            if default_qty is not None:
                try:
                    return float(default_qty)
                except (ValueError, TypeError):
                    pass
            return 1.0

        logged_names = []
        failed_names = []

        if pending_foods:
            if pending_all_foods:
                selected_ids = {f["food_id"] for f in pending_foods}
                for f in pending_all_foods:
                    if f["food_id"] in selected_ids:
                        matched = next((x for x in pending_foods if x["food_id"] == f["food_id"]), None)
                        if matched:
                            q = quantity_for_food(matched["name"], default_qty=matched.get("quantity"))
                            f["quantity"] = str(q)
            if log_foods(self.session, pending_foods, pending_all_foods, meal_idx, pending_metadata):
                for f in pending_foods:
                    logged_names.append(f["name"])
            else:
                failed_names.extend(f["name"] for f in pending_foods)

        for food in pending_searched:
            q = quantity_for_food(food["name"], default_qty=food.get("quantity"))
            if log_searched_food(self.session, food, meal_idx, pending_search_metadata, quantity=str(q)):
                logged_names.append(food["name"])
            else:
                failed_names.append(food["name"])

        if logged_names:
            invalidate_recents_cache(meal_idx)

        return bool(logged_names), len(logged_names), logged_names, failed_names

    # ── Helpers ──────────────────────────────────────────────────────────

    def _resolve_meal(self, mentioned_meal, last_meal_idx, unlogged):
        if mentioned_meal:
            return resolve_meal_idx(mentioned_meal, unlogged)
        if last_meal_idx is not None:
            return last_meal_idx
        return resolve_meal_idx(None, unlogged)
