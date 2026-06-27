
import base64
import json
from email.mime.text import MIMEText

from google import genai
from google.genai import types

import calendar_tools

DRAFT_REPLY_TOOL = {
    "name": "draft_reply",
    "description": (
        "Create a draft reply to this email in the user's Gmail account. "
        "Use this when the email expects a response from the user (a "
        "question, a request, or an explicit ask) and a short reply is "
        "appropriate. The draft is saved for human review and is never "
        "sent automatically."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reply_body": {
                "type": "string",
                "description": "The full plain-text body of the reply, written in a professional tone, 2-5 sentences.",
            },
            "rationale": {
                "type": "string",
                "description": "One sentence on why a reply is warranted.",
            },
        },
        "required": ["reply_body", "rationale"],
    },
}

FLAG_FOR_SCHEDULING_TOOL = {
    "name": "flag_for_scheduling",
    "description": (
        "Use this when the email is requesting a meeting, call, or "
        "appointment and proposes or asks for availability. The agent "
        "checks the user's actual calendar (read-only) and drafts a "
        "reply proposing real open time slots, falling back to asking "
        "the sender for availability if calendar access isn't set up."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "acknowledgement": {
                "type": "string",
                "description": "A short opening line acknowledging the scheduling request, to prepend before the proposed time slots.",
            },
        },
        "required": ["acknowledgement"],
    },
}

NO_ACTION_TOOL = {
    "name": "no_action_needed",
    "description": (
        "Use this when the email is high-priority but does not need a "
        "reply or scheduling action right now (e.g. an FYI notice, an "
        "alert, a receipt). No draft is created."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "One sentence on why no action is needed.",
            },
        },
        "required": ["reason"],
    },
}

ACTION_TOOLS = types.Tool(
    function_declarations=[DRAFT_REPLY_TOOL, FLAG_FOR_SCHEDULING_TOOL, NO_ACTION_TOOL]
)


def decide_and_act(client, model, service, message_id, sender, subject, body, triage):
    """
    The ReAct step. Given an already-triaged high-priority email, ask the
    model to reason over the content and pick exactly one tool to call.
    The chosen tool is then executed for real against the Gmail API
    (drafts only). Returns a short string describing what happened, for
    logging in the console output.

    If a draft already exists on this thread, the agent does NOT create
    a second one. Instead it asks the model to review the existing draft
    against the (possibly updated) email and either suggest edits or
    confirm it still looks right — but it never writes a new draft.
    """
    thread_id, _ = _get_thread_context(service, message_id)
    existing_draft = _find_existing_draft_for_thread(service, thread_id)

    if existing_draft is not None:
        return _review_existing_draft(client, model, existing_draft, sender, subject, body, triage)

    prompt = f"""You are deciding the next action for a high-priority email that has already been triaged.

Sender: {sender}
Subject: {subject}
Triage category: {triage.get("category", "Unknown")}
Triage reason: {triage.get("reason", "N/A")}
Body:
{body[:2000]}

Decide which single tool best fits this email and call it. If the email
clearly requests scheduling/availability, prefer flag_for_scheduling over
draft_reply. If no response is warranted, call no_action_needed.
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(tools=[ACTION_TOOLS]),
    )

    call = _extract_function_call(response)
    if call is None:
        return "  [REACT] Model returned no tool call; skipping."

    name = call.name
    args = dict(call.args) if call.args else {}

    if name == "draft_reply":
        created = _create_draft(service, message_id, sender, subject, args.get("reply_body", ""))
        if created:
            return f"  [REACT->draft_reply] Created draft reply. ({args.get('rationale', '')})"
        return "  [REACT->draft_reply] Skipped — see warning above."

    if name == "flag_for_scheduling":
        acknowledgement = args.get("acknowledgement", "Happy to find a time that works.")
        reply_body = _build_scheduling_reply(acknowledgement)
        created = _create_draft(service, message_id, sender, subject, reply_body)
        if created:
            return "  [REACT->flag_for_scheduling] Created draft proposing scheduling follow-up."
        return "  [REACT->flag_for_scheduling] Skipped — see warning above."

    if name == "no_action_needed":
        return f"  [REACT->no_action_needed] {args.get('reason', 'No reason given.')}"

    return f"  [REACT] Unrecognized tool call: {name}"


def _build_scheduling_reply(acknowledgement):
    """
    Build the body of a scheduling reply. Tries to check real calendar
    availability via calendar_tools; if that's not set up or fails for
    any reason, falls back to asking the sender for their availability
    instead — the agent never blocks or crashes on a missing calendar
    connection, it just degrades to the simpler behavior.
    """
    slots_text = calendar_tools.get_proposed_slots_text()

    if slots_text and "No open slots" not in slots_text:
        return (
            f"{acknowledgement}\n\n"
            f"Here are a few times that work on my end:\n\n"
            f"{slots_text}\n\n"
            f"Let me know if any of these work, or suggest another time."
        )

    return (
        f"{acknowledgement}\n\n"
        f"Could you share a couple of times that work for you, and I'll confirm?"
    )


def _find_existing_draft_for_thread(service, thread_id):
    """
    Look for a draft already attached to this thread. Returns the draft
    dict (as returned by drafts().get()) if found, else None.

    drafts().list() has no direct threadId filter, so we list drafts and
    check each candidate's threadId. This is fine at the scale this
    agent operates at (a handful of unread emails per run); if your
    Drafts folder grows very large, narrow the q= below to cut down the
    candidates fetched.
    """
    try:
        listing = service.users().drafts().list(userId="me", maxResults=50).execute()
    except Exception:
        return None

    for draft_stub in listing.get("drafts", []):
        try:
            draft = service.users().drafts().get(
                userId="me", id=draft_stub["id"], format="full"
            ).execute()
        except Exception:
            continue
        if draft.get("message", {}).get("threadId") == thread_id:
            return draft
    return None


def _review_existing_draft(client, model, existing_draft, sender, subject, body, triage):
    """
    A draft already exists for this thread. Rather than create a
    duplicate, ask the model to look at the existing draft text against
    the current email and either say it still looks fine, or suggest
    specific edits — as text only. Nothing is written back to Gmail.
    """
    draft_text = _get_draft_body_text(existing_draft)
    draft_id = existing_draft.get("id", "unknown")

    prompt = f"""A draft reply already exists for this email thread. Review it
and decide if it still fits, given the email content below. Do not assume
the draft needs changing just because time has passed — only flag genuine
problems (factual mismatch, missing answer to a new question, wrong tone).

Sender: {sender}
Subject: {subject}
Latest email body:
{body[:2000]}

Triage category: {triage.get("category", "Unknown")}

Existing draft reply text:
{draft_text or "(could not read draft text)"}

Respond with either:
- "OK: <one sentence>" if the existing draft still looks appropriate, or
- "SUGGEST: <specific edit suggestion>" if something should change.
"""
    response = client.models.generate_content(model=model, contents=prompt)
    verdict = (response.text or "").strip()

    return (
        f"  [REACT->existing_draft] Draft {draft_id} already exists for this thread; "
        f"not creating a new one. Review: {verdict}"
    )


def _get_draft_body_text(draft):
    """Best-effort plain-text extraction from a drafts().get(format='full') response."""
    try:
        payload = draft["message"]["payload"]
    except (KeyError, TypeError):
        return None
    return _extract_plain_text(payload)


def _extract_plain_text(payload):
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
            elif "parts" in part:
                nested = _extract_plain_text(part)
                if nested:
                    return nested
        return None
    data = payload.get("body", {}).get("data")
    if data:
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    return None


def _extract_function_call(response):
    """Pull the first function_call out of a Gemini response, if present."""
    try:
        for part in response.candidates[0].content.parts:
            if getattr(part, "function_call", None):
                return part.function_call
    except (AttributeError, IndexError):
        pass
    return None


def _create_draft(service, message_id, sender, subject, body_text):
    """
    Create a Gmail draft replying to message_id. This NEVER sends — it
    only saves a draft visible in the user's Gmail account for review.
    """
    reply_subject = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    thread_id, original_message_id_header = _get_thread_context(service, message_id)

    if not original_message_id_header:
        print(
            "  [WARNING] No Message-ID header found on the original email; "
            "skipping draft creation to avoid an unanchored/phantom draft. "
            "You can reply manually for this one."
        )
        return False

    message = MIMEText(body_text)
    message["to"] = _extract_email_address(sender)
    message["subject"] = reply_subject
    message["In-Reply-To"] = original_message_id_header
    message["References"] = original_message_id_header

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    service.users().drafts().create(
        userId="me",
        body={
            "message": {
                "raw": raw,
                "threadId": thread_id,
            }
        },
    ).execute()
    return True


def _get_thread_context(service, message_id):
    """Return (threadId, Message-Id header) for the message being replied to."""
    msg = service.users().messages().get(
        userId="me", id=message_id, format="metadata",
        metadataHeaders=["Message-Id", "Message-ID"]
    ).execute()
    thread_id = msg.get("threadId")
    headers = msg.get("payload", {}).get("headers", [])
    message_id_header = next(
        (h["value"] for h in headers if h["name"].lower() == "message-id"), None
    )
    return thread_id, message_id_header


def _extract_email_address(sender_header):
    """Pull a bare email address out of a 'Name <email@x.com>' header value."""
    if "<" in sender_header and ">" in sender_header:
        return sender_header.split("<", 1)[1].split(">", 1)[0].strip()
    return sender_header.strip()