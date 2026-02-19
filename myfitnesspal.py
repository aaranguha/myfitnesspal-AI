#!/usr/bin/env python3
"""
MyFitnessPal Auto-Logger
Logs food to MFP using saved cookies from your browser.
No Selenium, no browser automation — pure HTTP requests.

Setup: Copy your MFP cookie string from Chrome DevTools into .env
"""

import argparse
import json
import os
import sys
from datetime import date, datetime

import requests
from dotenv import load_dotenv
from lxml import html

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

BASE_URL = "https://www.myfitnesspal.com"

MEALS = {
    0: {"name": "Breakfast", "by_hour": 11},
    1: {"name": "Lunch",     "by_hour": 15},
    2: {"name": "Dinner",    "by_hour": 21},
    3: {"name": "Snacks",    "by_hour": 23},
}

PRESETS_FILE = os.path.join(SCRIPT_DIR, "meals.json")


# ── Meal presets ───────────────────────────────────────────────────────────

def load_presets():
    """Load saved meal presets from meals.json."""
    try:
        with open(PRESETS_FILE, "r") as f:
            data = json.load(f)
        return data.get("presets", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_presets(presets):
    """Write meal presets dict to meals.json."""
    with open(PRESETS_FILE, "w") as f:
        json.dump({"presets": presets}, f, indent=2)


def get_preset_names():
    """Return list of all saved preset names."""
    return list(load_presets().keys())


def get_preset(name):
    """Fuzzy-match a preset name. Returns (matched_name, preset_dict) or (None, None)."""
    presets = load_presets()
    lower = name.lower().strip()
    # Exact match
    for k, v in presets.items():
        if k.lower() == lower:
            return k, v
    # Substring match
    for k, v in presets.items():
        if lower in k.lower() or k.lower() in lower:
            return k, v
    return None, None


# ── Session setup ──────────────────────────────────────────────────────────

def create_session():
    """Create a requests session using the cookie string from .env."""
    cookie_string = os.getenv("MFP_COOKIE")

    if not cookie_string or cookie_string == "PASTE_YOUR_COOKIE_HERE":
        print("No MFP cookie found! Here's how to get it:\n")
        print("  1. Open Chrome and go to: myfitnesspal.com/food/diary")
        print("  2. Press F12 (or Cmd+Option+I) to open DevTools")
        print("  3. Click the 'Network' tab, then reload the page (Cmd+R)")
        print("  4. Click the first request in the list")
        print("  5. Under 'Request Headers', find 'Cookie:'")
        print("  6. Copy the ENTIRE value after 'Cookie: '")
        print("  7. Paste it in your .env file as:")
        print("     MFP_COOKIE=the_whole_cookie_string_here")
        sys.exit(1)

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Cookie": cookie_string,
    })

    # Verify login
    print("Verifying MFP login...")
    resp = session.get(f"{BASE_URL}/food/diary", allow_redirects=False)

    if resp.status_code in (301, 302, 303):
        location = resp.headers.get("Location", "")
        if "login" in location or "account" in location:
            print("Cookie expired or invalid! Get a fresh one from Chrome DevTools.")
            sys.exit(1)

    if resp.status_code != 200:
        print(f"Unexpected response ({resp.status_code}). Cookie may be expired.")
        sys.exit(1)

    print("Logged in!")
    return session


# ── Diary checking ─────────────────────────────────────────────────────────

def check_meals_logged(session, target_date):
    """Check which meals already have food logged."""
    date_str = target_date.strftime("%Y-%m-%d")
    resp = session.get(f"{BASE_URL}/food/diary?date={date_str}")
    tree = html.fromstring(resp.content)

    logged = {}
    for meal_idx, meal_info in MEALS.items():
        meal_name = meal_info["name"]
        has_food = False

        food_rows = tree.xpath(
            f"//tr[contains(@class, 'meal_{meal_idx}') and contains(@class, 'food')]"
        )
        if not food_rows:
            section = tree.xpath(
                f"//td[contains(@class, 'first') and contains(text(), '{meal_name}')]"
                f"/ancestor::tr/following-sibling::tr"
            )
            for row in section:
                classes = row.get("class", "")
                if "bottom" in classes or "total" in classes:
                    break
                text = row.text_content().strip()
                if text and "Add Food" not in text:
                    has_food = True
                    break

        logged[meal_idx] = has_food or len(food_rows) > 0

    return logged


def _parse_summary_row(row):
    """Parse a totals/goal/remaining row into a dict of macros."""
    tds = row.xpath(".//td")
    # Columns: label, calories, carbs, fat, protein, sodium, sugar
    def get_val(td):
        macro = td.xpath(".//span[@class='macro-value']")
        if macro:
            return macro[0].text_content().strip().replace(",", "")
        return td.text_content().strip().replace(",", "")

    return {
        "calories": get_val(tds[1]) if len(tds) > 1 else "0",
        "carbs": get_val(tds[2]) if len(tds) > 2 else "0",
        "fat": get_val(tds[3]) if len(tds) > 3 else "0",
        "protein": get_val(tds[4]) if len(tds) > 4 else "0",
    }


def get_diary_details(session, target_date):
    """Get detailed diary info: foods per meal, totals, goals, and remaining."""
    date_str = target_date.strftime("%Y-%m-%d")
    resp = session.get(f"{BASE_URL}/food/diary?date={date_str}")

    # Save diary HTML for debugging (overwritten each call)
    debug_path = os.path.join(SCRIPT_DIR, "debug_diary.html")
    with open(debug_path, "w") as f:
        f.write(resp.text)

    tree = html.fromstring(resp.content)

    diary = {}
    for meal_idx, meal_info in MEALS.items():
        meal_name = meal_info["name"]
        foods = []

        header = tree.xpath(
            f"//td[contains(@class, 'first') and contains(@class, 'alt') "
            f"and normalize-space(text())='{meal_name}']/ancestor::tr"
        )
        if not header:
            diary[meal_idx] = foods
            continue

        for row in header[0].itersiblings():
            classes = row.get("class", "")
            if "bottom" in classes or "total" in classes or "meal_header" in classes:
                break

            # Look for regular food entries
            link = row.xpath(".//a[contains(@class, 'js-show-edit-food')]")
            entry_id = ""
            if link:
                entry_id = link[0].get("data-food-entry-id", "")
            else:
                # Quick Add entries use <a href="/food/quick_add/ID"> with no special class
                link = row.xpath(".//a[@data-food-entry-id]")
                if link:
                    entry_id = link[0].get("data-food-entry-id", "")
                else:
                    # Also catch plain links in first <td> (Quick Add rows)
                    link = row.xpath(".//td[contains(@class, 'first')]//a")

            if link:
                food_name = link[0].text_content().strip()
                if not entry_id:
                    # Extract entry ID from href like /food/quick_add/12553481572
                    href = link[0].get("href", "")
                    if "/" in href:
                        entry_id = href.rstrip("/").rsplit("/", 1)[-1]
                tds = row.xpath(".//td")
                cals = tds[1].text_content().strip() if len(tds) > 1 else ""
                protein_el = tds[4].xpath(".//span[@class='macro-value']") if len(tds) > 4 else []
                protein = protein_el[0].text_content().strip() if protein_el else ""

                foods.append({
                    "name": food_name,
                    "entry_id": entry_id,
                    "calories": cals,
                    "protein": protein,
                })

        diary[meal_idx] = foods

    # Parse totals, goals, remaining from the summary rows
    totals_row = tree.xpath("//tr[@class='total']/td[@class='first' and contains(text(), 'Totals')]/ancestor::tr")
    goal_row = tree.xpath("//tr[contains(@class, 'total')]/td[@class='first' and contains(text(), 'Daily Goal')]/ancestor::tr")
    remaining_row = tree.xpath("//tr[contains(@class, 'remaining')]/td[@class='first' and contains(text(), 'Remaining')]/ancestor::tr")

    summary = {
        "totals": _parse_summary_row(totals_row[0]) if totals_row else {},
        "goal": _parse_summary_row(goal_row[0]) if goal_row else {},
        "remaining": _parse_summary_row(remaining_row[0]) if remaining_row else {},
    }

    return diary, summary


def get_meals_to_log(logged_status):
    """Based on current time and what's already logged, return meal indices to offer."""
    now_hour = datetime.now().hour
    to_log = []
    for meal_idx, meal_info in MEALS.items():
        if now_hour >= meal_info["by_hour"] - 4 and not logged_status.get(meal_idx, False):
            to_log.append(meal_idx)
    return to_log


# ── Recent foods ───────────────────────────────────────────────────────────

def get_recent_foods(session, meal_idx):
    """Fetch the add_to_diary page and parse the recent/frequent foods list."""
    resp = session.get(f"{BASE_URL}/food/add_to_diary?meal={meal_idx}")
    page_text = resp.text
    tree = html.fromstring(resp.content)

    foods = []

    # MFP uses checkboxes with data-food-id attributes inside a form that POSTs
    # to /food/add_favorites. Each food entry has:
    #   - hidden input: favorites[N][food_id] with value=<food_id>
    #   - checkbox: with data-food-id, data-index
    #   - <td> with food name
    #   - quantity input: favorites[N][quantity]
    #   - select: favorites[N][weight_id]
    checkboxes = tree.xpath("//input[@class='checkbox' and @data-food-id]")

    for cb in checkboxes:
        food_id = cb.get("data-food-id")
        index = cb.get("data-index")

        # Get food name from the next <td> sibling
        # The checkbox is inside a <td>, the name is in the next <td>
        parent_td = cb.getparent()
        if parent_td is not None:
            next_td = parent_td.getnext()
            if next_td is not None:
                food_name = next_td.text_content().strip()
            else:
                food_name = f"Food #{food_id}"
        else:
            food_name = f"Food #{food_id}"

        # Get the default quantity from the input
        qty_input = tree.xpath(
            f"//input[@name='favorites[{index}][quantity]']/@value"
        )
        quantity = qty_input[0] if qty_input else "1"

        # Get the default weight_id from the select element
        weight_select = tree.xpath(
            f"//select[@name='favorites[{index}][weight_id]']//option[@selected]/@value"
        )
        weight_id = weight_select[0] if weight_select else "0"

        foods.append({
            "name": food_name,
            "food_id": food_id,
            "index": index,
            "quantity": quantity,
            "weight_id": weight_id,
        })

    # Also extract the authenticity_token and date from the form
    token = tree.xpath("//form[@action='/food/add_favorites']//input[@name='authenticity_token']/@value")
    form_date = tree.xpath("//form[@action='/food/add_favorites']//input[@name='date']/@value")

    metadata = {
        "token": token[0] if token else None,
        "date": form_date[0] if form_date else None,
    }

    if not foods:
        debug_path = os.path.join(SCRIPT_DIR, f"debug_meal_{meal_idx}.html")
        with open(debug_path, "w") as f:
            f.write(page_text)
        print(f"  [debug] No foods found. Page saved to debug_meal_{meal_idx}.html")

    return foods, metadata


# ── Food search ────────────────────────────────────────────────────────────

def search_foods(session, query, meal_idx):
    """Search MFP's food database by name. Returns (results_list, metadata).

    Each result includes external_id and weight_ids extracted from the search
    results page's <a class="search"> data attributes, which are needed to log
    the food via POST to /food/add.
    """
    # Get token and date from the add_to_diary page's search form
    resp = session.get(f"{BASE_URL}/food/add_to_diary?meal={meal_idx}")
    tree = html.fromstring(resp.content)

    token = tree.xpath(
        "//form[@action='/food/search']//input[@name='authenticity_token']/@value"
    )
    form_date = tree.xpath(
        "//form[@action='/food/search']//input[@name='date']/@value"
    )
    auth_token = token[0] if token else _get_authenticity_token(session)
    search_date = form_date[0] if form_date else date.today().strftime("%Y-%m-%d")

    # POST search
    search_resp = session.post(
        f"{BASE_URL}/food/search",
        data={
            "authenticity_token": auth_token,
            "meal": str(meal_idx),
            "date": search_date,
            "search": query,
        },
        headers={"Referer": f"{BASE_URL}/food/add_to_diary?meal={meal_idx}"},
        allow_redirects=True,
    )

    search_tree = html.fromstring(search_resp.content)
    foods = []

    # Extract the authenticity_token from the add-to-diary form on the search page
    add_token = search_tree.xpath(
        "//form[@id='food-nutritional-details-form']//input[@name='authenticity_token']/@value"
    )
    if add_token:
        auth_token = add_token[0]

    # Search results are <li class="matched-food"> with <a class="search"> links
    # Each link has data-external-id, data-weight-ids, data-verified, etc.
    search_links = search_tree.xpath("//a[@class='search' and @data-external-id]")

    for link in search_links[:15]:
        name = link.text_content().strip()
        external_id = link.get("data-external-id", "")
        original_id = link.get("data-original-id", "")
        version = link.get("data-version", "")
        weight_ids_str = link.get("data-weight-ids", "")
        verified = link.get("data-verified", "false") == "true"

        weight_ids = [w.strip() for w in weight_ids_str.split(",") if w.strip()]

        # Get calorie/serving info from the sibling <p class="search-nutritional-info">
        parent_li = link.getparent()
        if parent_li is not None:
            parent_li = parent_li.getparent()  # search-title-container -> li
        cal_info = ""
        if parent_li is not None:
            info_el = parent_li.xpath(".//p[contains(@class, 'search-nutritional-info')]")
            if info_el:
                cal_info = info_el[0].text_content().strip()

        foods.append({
            "name": name,
            "external_id": external_id,
            "original_id": original_id,
            "version": version,
            "weight_ids": weight_ids,
            "default_weight_id": weight_ids[0] if weight_ids else "",
            "verified": verified,
            "cal_info": cal_info,
        })

    metadata = {"token": auth_token, "date": search_date}

    if not foods:
        debug_path = os.path.join(SCRIPT_DIR, "debug_search_results.html")
        with open(debug_path, "w") as f:
            f.write(search_resp.text)
        print(f"  [debug] No search results parsed. Saved to debug_search_results.html")

    return foods, metadata


def _get_api_auth_token(session):
    """Fetch MFP's internal OAuth token from /user/auth_token.

    MFP's callV2 AJAX wrapper adds Authorization, mfp-client-id, mfp-user-id,
    and Accept: application/json headers to all API calls.
    """
    resp = session.get(f"{BASE_URL}/user/auth_token")
    if resp.status_code == 200:
        try:
            data = resp.json()
            return {
                "token_type": data.get("token_type", "Bearer"),
                "access_token": data.get("access_token", ""),
                "user_id": data.get("user_id", 0),
            }
        except Exception:
            pass
    return None


def log_searched_food(session, food, meal_idx, metadata, quantity="1"):
    """Log a food found via search to the diary.

    Replicates MFP's addToDiary() JS which uses callV2 (jQuery AJAX with
    OAuth auth headers) to POST to /food/add.

    Args:
        food: dict from search_foods() with external_id, original_id, etc.
        meal_idx: meal index (0=Breakfast, 1=Lunch, 2=Dinner, 3=Snacks)
        metadata: dict with 'token' and 'date' from search_foods()
        quantity: number of servings as string (default "1")
    """
    search_date = metadata.get("date", date.today().strftime("%Y-%m-%d"))

    # MFP's addToDiary() JS uses original_id for food_entry[food_id]
    food_id = food.get("original_id") or food["external_id"]
    weight_id = food.get("default_weight_id", "")

    csrf_token = metadata.get("token") or _get_authenticity_token(session)

    data = {
        "authenticity_token": csrf_token,
        "food_entry[food_id]": food_id,
        "food_entry[date]": search_date,
        "food_entry[quantity]": quantity,
        "food_entry[weight_id]": weight_id,
        "food_entry[meal_id]": str(meal_idx),
        "ajax": "true",
    }

    # MFP's callV2 adds OAuth headers from /user/auth_token
    auth_info = _get_api_auth_token(session)
    headers = {
        "Referer": f"{BASE_URL}/food/search",
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json",
        "mfp-client-id": "mfp-main-js",
    }
    if auth_info:
        headers["Authorization"] = f"{auth_info['token_type']} {auth_info['access_token']}"
        headers["mfp-user-id"] = str(auth_info.get("user_id", 0))

    resp = session.post(
        f"{BASE_URL}/food/add",
        data=data,
        headers=headers,
        allow_redirects=False,
    )

    if resp.status_code in (200, 201):
        return True
    if resp.status_code in (301, 302, 303):
        loc = resp.headers.get("Location", "")
        if loc:
            session.get(BASE_URL + loc if loc.startswith("/") else loc)
        if "diary" in loc:
            return True

    # 204 = "no content" — verify by checking diary
    if resp.status_code == 204:
        diary, _ = get_diary_details(session, date.fromisoformat(search_date))
        food_names = [f["name"].lower() for f in diary.get(meal_idx, [])]
        food_name_lower = food.get("name", "").lower()
        if any(food_name_lower in n or n in food_name_lower for n in food_names):
            return True

    return False


# ── SMS / daily summary ──────────────────────────────────────────────────

def send_sms(body):
    """Send an iMessage via macOS Messages app to all numbers in SMS_PHONE_NUMBERS. Returns True if any succeeded."""
    import subprocess

    phones_raw = os.getenv("SMS_PHONE_NUMBERS", "")
    if not phones_raw:
        print("  [sms] No phone numbers configured in SMS_PHONE_NUMBERS.")
        return False

    phones = [p.strip() for p in phones_raw.split(",") if p.strip()]
    if not phones:
        print("  [sms] No phone numbers configured in SMS_PHONE_NUMBERS.")
        return False

    # Escape special characters for AppleScript
    escaped_body = body.replace("\\", "\\\\").replace('"', '\\"')

    any_ok = False
    for phone in phones:
        try:
            script = f'''
            tell application "Messages"
                set targetService to 1st account whose service type = iMessage
                set targetBuddy to participant "{phone}" of targetService
                send "{escaped_body}" to targetBuddy
            end tell
            '''
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                print(f"  [sms] Sent iMessage to {phone}")
                any_ok = True
            else:
                print(f"  [sms] Failed to send to {phone}: {result.stderr.strip()}")
        except Exception as e:
            print(f"  [sms] Failed to send to {phone}: {e}")

    return any_ok


def build_daily_summary(session, target_date=None):
    """Build a formatted daily diet summary string from MFP diary data."""
    if target_date is None:
        target_date = date.today()

    _, summary = get_diary_details(session, target_date)
    totals = summary.get("totals", {})
    goals = summary.get("goal", {})

    if not totals or not goals:
        return None

    def _check_goal(key, t, g):
        """Returns True if goal is met based on per-macro rules."""
        if key == "calories":
            # Must be at or above goal, but not more than 300 over
            return t >= g and t <= g + 300
        elif key == "protein":
            # Must hit or exceed goal
            return t >= g
        else:
            # Carbs/fat: stay at or below goal (5% buffer)
            return t <= g * 1.05

    def _fmt(label, key, total_val, goal_val, unit=""):
        try:
            t = int(float(total_val))
            g = int(float(goal_val))
        except (ValueError, TypeError):
            return f"{label}: ? / ?", False
        met = _check_goal(key, t, g)
        icon = "\u2705" if met else "\u274c"
        t_str = f"{t:,}"
        g_str = f"{g:,}"
        return f"{label}: {t_str}{unit} / {g_str}{unit}  {icon}", met

    cal_line, cal_met = _fmt("Calories", "calories", totals.get("calories", "0"), goals.get("calories", "0"))
    pro_line, pro_met = _fmt("Protein", "protein", totals.get("protein", "0"), goals.get("protein", "0"), "g")
    carb_line, carb_met = _fmt("Carbs", "carbs", totals.get("carbs", "0"), goals.get("carbs", "0"), "g")
    fat_line, fat_met = _fmt("Fat", "fat", totals.get("fat", "0"), goals.get("fat", "0"), "g")

    lines = [
        f"\U0001f37d MFP Daily Summary \u2014 {target_date.strftime('%b %d')}",
        "",
        cal_line,
        pro_line,
        carb_line,
        fat_line,
    ]

    results = [cal_met, pro_met, carb_met, fat_met]
    met = sum(1 for r in results if r)
    total_goals = len(results)

    lines.append("")
    lines.append(f"Overall: {met}/{total_goals} goals met!")

    return "\n".join(lines)


# ── Food logging ───────────────────────────────────────────────────────────

def log_foods(session, selected_foods, all_foods, meal_idx, metadata):
    """
    Log selected foods to MFP diary by POSTing the add_favorites form.
    MFP expects ALL food entries in the form, with checked=1 for selected ones.
    """
    data = {}

    if metadata.get("token"):
        data["authenticity_token"] = metadata["token"]
    data["meal"] = str(meal_idx)
    data["date"] = metadata.get("date", "")

    selected_ids = {f["food_id"] for f in selected_foods}

    for food in all_foods:
        idx = food["index"]
        data[f"favorites[{idx}][food_id]"] = food["food_id"]
        data[f"favorites[{idx}][quantity]"] = food.get("quantity", "1")
        data[f"favorites[{idx}][weight_id]"] = food["weight_id"]

        if food["food_id"] in selected_ids:
            data[f"favorites[{idx}][checked]"] = "1"
        else:
            data[f"favorites[{idx}][checked]"] = "0"

    # The submit button name=value tells MFP to add (not delete)
    data["add"] = "Add Checked"

    resp = session.post(
        f"{BASE_URL}/food/add_favorites",
        data=data,
        headers={"Referer": f"{BASE_URL}/food/add_to_diary?meal={meal_idx}"},
        allow_redirects=False,
    )

    if resp.status_code in (301, 302, 303):
        location = resp.headers.get("Location", "")
        session.get(BASE_URL + location if location.startswith("/") else location)
        return "diary" in location

    # Save response for debugging if something went wrong
    debug_path = os.path.join(SCRIPT_DIR, "debug_post_response.html")
    with open(debug_path, "w") as f:
        f.write(resp.text)
    print(f"    [debug] Unexpected response ({resp.status_code}). Saved to debug_post_response.html")

    return False


def quick_add_food(session, meal_idx, description, calories, protein=0, carbs=0, fat=0):
    """Quick-add a food entry with custom macros directly to the diary.

    Creates a named custom food via MFP API, then adds it to the diary
    as a food_entry so it shows with a proper name (not "Quick Add").
    Falls back to type=quick_add if the named approach fails.
    """
    today_str = date.today().strftime("%Y-%m-%d")
    meal_name = MEALS[meal_idx]["name"]

    auth_info = _get_api_auth_token(session)
    if not auth_info:
        print("    [debug] Could not get API auth token for quick add")
        return False

    api_headers = {
        "Authorization": f"{auth_info['token_type']} {auth_info['access_token']}",
        "mfp-user-id": str(auth_info.get("user_id", 0)),
        "mfp-client-id": "mfp-main-js",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest",
    }

    api_base = "https://api.myfitnesspal.com"
    diary_url = f"{api_base}/v2/diary"
    debug_info = []

    nutrition = {
        "energy": {"value": int(calories), "unit": "calories"},
        "fat": int(fat),
        "carbohydrates": int(carbs),
        "protein": int(protein),
    }

    # Use quick_add — try with description field for custom name
    debug_info.append("Using quick_add...")
    qa_variants = [
        # Try with description to override "Quick Add" name
        {"type": "quick_add", "date": today_str, "meal_name": meal_name,
         "nutritional_contents": nutrition, "description": description},
        # Try with food object containing description
        {"type": "quick_add", "date": today_str, "meal_name": meal_name,
         "nutritional_contents": nutrition,
         "food": {"description": description}},
        # Plain quick_add (known to work)
        {"type": "quick_add", "date": today_str, "meal_name": meal_name,
         "nutritional_contents": nutrition},
    ]

    resp = None
    for qi, qa_entry in enumerate(qa_variants):
        resp = session.post(diary_url, json={"items": [qa_entry]}, headers=api_headers)
        debug_info.append(f"quick_add try {qi+1}: {resp.status_code} - {resp.text[:400]}")
        if resp.status_code in (200, 201):
            # Check if the name was actually set
            try:
                result = resp.json()
                items = result.get("items", [])
                if items:
                    food_desc = items[0].get("food", {}).get("description", "")
                    debug_info.append(f"  Food name in response: {food_desc}")
            except Exception:
                pass
            break

    # Save debug output
    debug_path = os.path.join(SCRIPT_DIR, "debug_quick_add.html")
    with open(debug_path, "w") as f:
        f.write("\n".join(debug_info))

    if resp.status_code in (200, 201):
        return True

    print(f"    [debug] Quick add failed. Saved to debug_quick_add.html")
    return False


def _get_authenticity_token(session):
    """Fetch the diary page and extract the CSRF authenticity_token."""
    resp = session.get(f"{BASE_URL}/food/diary")
    tree = html.fromstring(resp.content)
    token = tree.xpath("//meta[@name='csrf-token']/@content")
    if token:
        return token[0]
    # Fallback: look in forms
    token = tree.xpath("//input[@name='authenticity_token']/@value")
    return token[0] if token else None


def remove_food(session, entry_id):
    """Remove a food entry from the diary by its entry ID."""
    token = _get_authenticity_token(session)

    headers = {
        "Referer": f"{BASE_URL}/food/diary",
        "X-Requested-With": "XMLHttpRequest",
    }
    if token:
        headers["X-CSRF-Token"] = token

    # Try DELETE (MFP's JS uses this)
    resp = session.delete(
        f"{BASE_URL}/food/diary/{entry_id}",
        headers=headers,
    )
    if resp.status_code in (200, 204):
        return True

    # Fallback: POST to /food/remove/<id> with token
    data = {}
    if token:
        data["authenticity_token"] = token
    resp = session.post(
        f"{BASE_URL}/food/remove/{entry_id}",
        data=data,
        headers={"Referer": f"{BASE_URL}/food/diary"},
        allow_redirects=False,
    )
    if resp.status_code in (301, 302, 303):
        location = resp.headers.get("Location", "")
        if "diary" in location and "error" not in location:
            return True

    print(f"    [debug] remove_food failed for entry {entry_id} (status: {resp.status_code})")
    return False


# ── CLI interaction ────────────────────────────────────────────────────────

def select_foods(foods, meal_name):
    """Show recent foods and let user pick which to log."""
    if not foods:
        print(f"  No recent foods found for {meal_name}.")
        return []

    print(f"\n  Your recent foods for {meal_name}:")
    for i, food in enumerate(foods, 1):
        print(f"    [{i}] {food['name']}")

    print(f"    [a] Add ALL")
    print(f"    [s] Skip {meal_name}")

    pick = input(f"  Pick (comma-separated, 'a' for all, 's' to skip): ").strip().lower()

    if pick == "s":
        return []
    if pick == "a":
        return foods

    selected = []
    for p in pick.split(","):
        p = p.strip()
        try:
            idx = int(p) - 1
            if 0 <= idx < len(foods):
                selected.append(foods[idx])
            else:
                print(f"    Invalid: {p}")
        except ValueError:
            print(f"    Invalid: {p}")
    return selected


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MyFitnessPal Auto-Logger")
    parser.add_argument("--date", type=str, default=None, help="Date to log for (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="Check all meals regardless of time")
    args = parser.parse_args()

    if args.date:
        try:
            target_date = date.fromisoformat(args.date)
        except ValueError:
            print(f"Invalid date: {args.date}. Use YYYY-MM-DD.")
            return
    else:
        target_date = date.today()

    now = datetime.now()
    print(f"MFP Auto-Logger — {target_date.strftime('%A, %B %d, %Y')} ({now.strftime('%I:%M %p')})")
    print("=" * 55)

    session = create_session()

    print("\nChecking which meals are already logged...")
    logged_status = check_meals_logged(session, target_date)

    for meal_idx, is_logged in logged_status.items():
        status = "already logged" if is_logged else "NOT logged"
        print(f"  {MEALS[meal_idx]['name']}: {status}")

    if args.all:
        meals_to_log = [idx for idx, logged in logged_status.items() if not logged]
    else:
        meals_to_log = get_meals_to_log(logged_status)

    if not meals_to_log:
        print("\nAll meals are logged! Nothing to do.")
        return

    print(f"\nMeals to log: {', '.join(MEALS[m]['name'] for m in meals_to_log)}")

    summary = []
    for meal_idx in meals_to_log:
        meal_name = MEALS[meal_idx]["name"]
        print(f"\n{'─' * 40}")
        print(f"  {meal_name}")
        print(f"{'─' * 40}")

        all_foods, metadata = get_recent_foods(session, meal_idx)
        selected = select_foods(all_foods, meal_name)

        if not selected:
            print(f"  Skipping {meal_name}.")
            continue

        success = log_foods(session, selected, all_foods, meal_idx, metadata)
        if success:
            for food in selected:
                print(f"    Logged: {food['name']}")
                summary.append((meal_name, food["name"]))
        else:
            print(f"    Failed to log foods for {meal_name}.")

    print(f"\n{'=' * 55}")
    print("DONE!")
    if summary:
        for meal_name, food_name in summary:
            print(f"  [{meal_name}] {food_name}")
        print(f"\nTotal: {len(summary)} food(s) logged")
    else:
        print("  No foods were logged.")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
