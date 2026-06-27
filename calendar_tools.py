import datetime
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
CALENDAR_TOKEN_FILE = "calendar_token.json"

# Working-hours window used when proposing slots. Adjust to taste.
WORKDAY_START_HOUR = 9
WORKDAY_END_HOUR = 17
SLOT_LENGTH_MINUTES = 30
LOOKAHEAD_DAYS = 7
MAX_SLOTS_TO_PROPOSE = 3


def get_calendar_service():
    """
    Authenticate against the Calendar API, separately from Gmail's
    token.json. Kept as its own token file so a person who only wants
    the email features doesn't have to grant calendar access at all —
    this function is only called when flag_for_scheduling actually runs.
    """
    creds = None
    if os.path.exists(CALENDAR_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(CALENDAR_TOKEN_FILE, CALENDAR_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", CALENDAR_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(CALENDAR_TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


def get_busy_blocks(service, days_ahead=LOOKAHEAD_DAYS):
    """
    Query free/busy for the primary calendar over the next `days_ahead`
    days. Returns a list of (start_dt, end_dt) tuples, timezone-aware,
    sorted chronologically.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    time_min = now.isoformat()
    time_max = (now + datetime.timedelta(days=days_ahead)).isoformat()

    body = {
        "timeMin": time_min,
        "timeMax": time_max,
        "items": [{"id": "primary"}],
    }
    response = service.freebusy().query(body=body).execute()
    busy_raw = response.get("calendars", {}).get("primary", {}).get("busy", [])

    blocks = []
    for entry in busy_raw:
        start = datetime.datetime.fromisoformat(entry["start"].replace("Z", "+00:00"))
        end = datetime.datetime.fromisoformat(entry["end"].replace("Z", "+00:00"))
        blocks.append((start, end))
    return sorted(blocks, key=lambda b: b[0])


def compute_open_slots(busy_blocks, days_ahead=LOOKAHEAD_DAYS,
                        workday_start=WORKDAY_START_HOUR, workday_end=WORKDAY_END_HOUR,
                        slot_minutes=SLOT_LENGTH_MINUTES, max_slots=MAX_SLOTS_TO_PROPOSE):
    """
    Given busy blocks (as returned by get_busy_blocks), compute candidate
    open slots within working hours over the next `days_ahead` days.

    This is deliberately simple: it walks each working day, checks fixed
    candidate start times against the busy blocks, and collects the
    first `max_slots` that don't overlap anything busy. It does not try
    to find every possible gap — just enough good options to propose.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    open_slots = []

    for day_offset in range(days_ahead):
        day = now + datetime.timedelta(days=day_offset)
        if day.weekday() >= 5:  # skip Saturday/Sunday
            continue

        day_start = day.replace(hour=workday_start, minute=0, second=0, microsecond=0)
        day_end = day.replace(hour=workday_end, minute=0, second=0, microsecond=0)

        slot_start = max(day_start, now) if day_offset == 0 else day_start
        # Round up to the next slot boundary so we don't propose a slot
        # that started in the past or mid-hour.
        slot_start = _round_up_to_slot(slot_start, slot_minutes)

        while slot_start + datetime.timedelta(minutes=slot_minutes) <= day_end:
            slot_end = slot_start + datetime.timedelta(minutes=slot_minutes)
            if not _overlaps_any(slot_start, slot_end, busy_blocks):
                open_slots.append((slot_start, slot_end))
                if len(open_slots) >= max_slots:
                    return open_slots
            slot_start += datetime.timedelta(minutes=slot_minutes)

    return open_slots


def _round_up_to_slot(dt, slot_minutes):
    discard = datetime.timedelta(minutes=dt.minute % slot_minutes, seconds=dt.second, microseconds=dt.microsecond)
    if discard:
        dt = dt - discard + datetime.timedelta(minutes=slot_minutes)
    return dt


def _overlaps_any(start, end, busy_blocks):
    for busy_start, busy_end in busy_blocks:
        if start < busy_end and end > busy_start:
            return True
    return False


def format_slots_for_email(slots, timezone_label="UTC"):
    """
    Render computed slots as a short human-readable list to embed in a
    drafted reply, e.g.:
      - Tuesday, July 1 at 10:00 AM - 10:30 AM
      - Tuesday, July 1 at 2:00 PM - 2:30 PM
    """
    if not slots:
        return "No open slots were found in the next few working days."
    lines = []
    for start, end in slots:
        lines.append(f"- {start.strftime('%A, %B %d')} at {start.strftime('%I:%M %p').lstrip('0')} "
                      f"- {end.strftime('%I:%M %p').lstrip('0')} ({timezone_label})")
    return "\n".join(lines)


def get_proposed_slots_text(days_ahead=LOOKAHEAD_DAYS):
    """
    Convenience entry point used by react_actions: authenticate, fetch
    busy blocks, compute open slots, and return them as ready-to-embed
    text. Returns None if calendar access isn't available or fails,
    so callers can fall back to asking the sender for availability
    instead of crashing.

    If no slots are found in the requested window (e.g. the run happens
    late Friday and the window doesn't reach into the next work week),
    automatically retries once with a wider window before giving up.
    """
    try:
        service = get_calendar_service()
        busy = get_busy_blocks(service, days_ahead=days_ahead)
        slots = compute_open_slots(busy, days_ahead=days_ahead)

        if not slots and days_ahead < 14:
            wider_days = 14
            busy = get_busy_blocks(service, days_ahead=wider_days)
            slots = compute_open_slots(busy, days_ahead=wider_days)

        return format_slots_for_email(slots)
    except Exception as e:
        print(f"  [CALENDAR-WARNING] Could not check calendar availability: {e}")
        return None