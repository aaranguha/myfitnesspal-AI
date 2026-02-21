"""
AWS Lambda handler for the Diet Assistant Alexa Skill.

Routes Alexa requests to AssistantCore and manages session state.
"""

import json
import os
import traceback

from assistant_core import AssistantCore, empty_state

# ── Globals (persist across warm Lambda invocations) ─────────────────────

_assistant = None


def _load_mfp_cookie():
    """Load MFP cookie from env var or cookie.txt file (cookie is too large for env vars)."""
    cookie = os.environ.get("MFP_COOKIE")
    if cookie:
        return cookie
    # Fall back to cookie.txt bundled in the Lambda package
    cookie_path = os.path.join(os.path.dirname(__file__), "cookie.txt")
    try:
        with open(cookie_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise RuntimeError("No MFP cookie found. Add cookie.txt to the Lambda package.")


def get_assistant():
    """Lazy-init the assistant (reused across warm invocations)."""
    global _assistant
    if _assistant is None:
        _assistant = AssistantCore(
            mfp_cookie=_load_mfp_cookie(),
            openai_key=os.environ.get("OPENAI_API_KEY"),
            user_name=os.environ.get("USER_NAME", "Aaran"),
        )
    return _assistant


# ── Session state helpers ────────────────────────────────────────────────

MAX_SESSION_SIZE = 7500  # Leave room under Alexa's 8KB limit


def load_state(session_attrs):
    """Deserialize state from Alexa session attributes."""
    if not session_attrs:
        return empty_state()

    state = dict(session_attrs)
    # Alexa stringifies complex objects — deserialize JSON strings
    json_keys = [
        "pending_foods", "pending_searched_foods", "pending_all_foods",
        "pending_metadata", "pending_search_metadata", "pending_item_quantities",
        "disambiguation_queue", "not_found", "pick_options", "remove_candidates",
    ]
    for key in json_keys:
        if key in state and isinstance(state[key], str):
            try:
                state[key] = json.loads(state[key])
            except (json.JSONDecodeError, TypeError):
                pass

    # Restore types
    for bool_key in ["awaiting_confirmation", "awaiting_pick", "awaiting_remove_confirm"]:
        if bool_key in state:
            state[bool_key] = bool(state[bool_key])

    if "pending_meal_idx" in state and state["pending_meal_idx"] is not None:
        try:
            state["pending_meal_idx"] = int(state["pending_meal_idx"])
        except (ValueError, TypeError):
            state["pending_meal_idx"] = None

    if "last_meal_idx" in state and state["last_meal_idx"] is not None:
        try:
            state["last_meal_idx"] = int(state["last_meal_idx"])
        except (ValueError, TypeError):
            state["last_meal_idx"] = None

    return state


def save_state(state):
    """Serialize state for Alexa session attributes."""
    serialized = {}
    json_keys = {
        "pending_foods", "pending_searched_foods", "pending_all_foods",
        "pending_metadata", "pending_search_metadata", "pending_item_quantities",
        "disambiguation_queue", "not_found", "pick_options", "remove_candidates",
    }

    for key, value in state.items():
        if key in json_keys and isinstance(value, (list, dict)):
            serialized[key] = json.dumps(value)
        elif value is None:
            serialized[key] = None
        else:
            serialized[key] = value

    # Check size — prune if too large
    state_json = json.dumps(serialized, default=str)
    if len(state_json) > MAX_SESSION_SIZE:
        # Drop the heaviest fields to fit
        for drop_key in ["pending_all_foods", "pick_options", "remove_candidates"]:
            if drop_key in serialized:
                serialized[drop_key] = json.dumps([])
        # Recheck
        state_json = json.dumps(serialized, default=str)
        if len(state_json) > MAX_SESSION_SIZE:
            # Nuclear option — reset state
            serialized = {}
            for k, v in empty_state().items():
                serialized[k] = json.dumps(v) if isinstance(v, (list, dict)) else v

    return serialized


# ── Response builders ────────────────────────────────────────────────────

def build_response(speech, should_end=False, session_attrs=None, reprompt=None):
    """Build a standard Alexa response."""
    response = {
        "version": "1.0",
        "sessionAttributes": session_attrs or {},
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": speech,
            },
            "shouldEndSession": should_end,
        },
    }
    if reprompt and not should_end:
        response["response"]["reprompt"] = {
            "outputSpeech": {
                "type": "PlainText",
                "text": reprompt,
            }
        }
    return response


# ── Main handler ─────────────────────────────────────────────────────────

def lambda_handler(event, context):
    """Main entry point for Alexa requests."""
    try:
        request_type = event.get("request", {}).get("type", "")

        # ── LaunchRequest: "Alexa, open diet assistant" ──
        if request_type == "LaunchRequest":
            return build_response(
                speech="Hey, how can I help?",
                should_end=False,
                reprompt="Tell me what you ate, or ask about your diary.",
            )

        # ── SessionEndedRequest ──
        if request_type == "SessionEndedRequest":
            return build_response("", should_end=True)

        # ── IntentRequest ──
        if request_type == "IntentRequest":
            intent_name = event["request"]["intent"]["name"]

            if intent_name == "AMAZON.HelpIntent":
                return build_response(
                    speech=(
                        "You can say things like: log chicken breast to lunch, "
                        "what did I eat today, remove the protein bar from dinner, "
                        "or quick add 500 calories to snacks."
                    ),
                    should_end=False,
                    reprompt="What would you like to do?",
                )

            if intent_name in ("AMAZON.CancelIntent", "AMAZON.StopIntent"):
                return build_response("Talk to you later!", should_end=True)

            if intent_name == "AMAZON.FallbackIntent":
                return build_response(
                    "I didn't catch that. Try saying something like 'log chicken to lunch'.",
                    should_end=False,
                    reprompt="What would you like to log?",
                )

            if intent_name == "CatchAllIntent":
                return handle_catch_all(event)

        # Unknown request type
        return build_response("Sorry, I didn't understand that.", should_end=False)

    except Exception as e:
        print(f"ERROR: {traceback.format_exc()}")

        # Check for MFP auth failure
        error_msg = str(e).lower()
        if "cookie" in error_msg or "login" in error_msg or "auth" in error_msg:
            return build_response(
                "Your MyFitnessPal session expired. Please update your cookie in Lambda.",
                should_end=True,
            )

        return build_response(
            "Something went wrong. Try again in a moment.",
            should_end=False,
        )


def handle_catch_all(event):
    """Process the user's command through AssistantCore."""
    # Extract user text
    slots = event.get("request", {}).get("intent", {}).get("slots", {})
    user_command = slots.get("userCommand", {}).get("value", "")

    if not user_command:
        return build_response(
            "What would you like to do?",
            should_end=False,
            reprompt="Try saying something like 'log chicken breast to lunch'.",
        )

    # Load state from session
    session_attrs = event.get("session", {}).get("attributes", {})
    state = load_state(session_attrs)

    # Process through assistant core
    assistant = get_assistant()
    response_text, new_state, should_end = assistant.process_input(user_command, state)

    # Save state back to session
    saved_attrs = save_state(new_state)

    # Add a reprompt if session stays open
    reprompt = None
    if not should_end:
        if new_state.get("awaiting_confirmation"):
            reprompt = "Say yes to log, no to cancel, or specify a quantity."
        elif new_state.get("awaiting_pick"):
            reprompt = "Which number?"
        elif new_state.get("awaiting_remove_confirm"):
            reprompt = "Say yes to remove or no to keep."
        else:
            reprompt = "Anything else?"

    return build_response(
        speech=response_text,
        should_end=should_end,
        session_attrs=saved_attrs,
        reprompt=reprompt,
    )
