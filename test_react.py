"""
Tests ReAct decide_and_act() function and draft creation with fake Gmail
and Gemini clients, so it does not need real API keys or a real inbox.
"""

import sys
import base64
import types as pytypes
from unittest.mock import MagicMock

sys.path.insert(0, ".")
from react_actions import (
    decide_and_act,
    _extract_email_address,
    _create_draft,
    _find_existing_draft_for_thread,
)


def make_fake_genai_response(tool_name, args):
    """Build a fake google.genai response object shaped like the real SDK."""
    fake_function_call = MagicMock()
    fake_function_call.name = tool_name
    fake_function_call.args = args

    fake_part = MagicMock()
    fake_part.function_call = fake_function_call

    fake_content = MagicMock()
    fake_content.parts = [fake_part]

    fake_candidate = MagicMock()
    fake_candidate.content = fake_content

    fake_response = MagicMock()
    fake_response.candidates = [fake_candidate]
    return fake_response


def make_fake_service(get_return_value, existing_drafts=None):
    """
    Build a fake Gmail service. existing_drafts, if given, is a list of
    dicts like {"id": ..., "threadId": ...} representing drafts already
    in the mailbox — used to exercise the dedup-check path. Defaults to
    no existing drafts.
    """
    fake_service = MagicMock()
    fake_service.users().messages().get().execute.return_value = get_return_value

    existing_drafts = existing_drafts or []
    fake_service.users().drafts().list().execute.return_value = {
        "drafts": [{"id": d["id"]} for d in existing_drafts]
    }

    def fake_drafts_get(userId, id, format=None):
        match = next((d for d in existing_drafts if d["id"] == id), None)
        mock_request = MagicMock()
        mock_request.execute.return_value = match.get("full", {"id": id, "message": {"threadId": match["threadId"]}}) if match else {}
        return mock_request

    fake_service.users().drafts().get.side_effect = fake_drafts_get
    return fake_service


def test_extract_email_address():
    assert _extract_email_address("Jane Doe <jane@example.com>") == "jane@example.com"
    assert _extract_email_address("plain@example.com") == "plain@example.com"
    print("PASS: _extract_email_address")


def test_draft_reply_path():
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = make_fake_genai_response(
        "draft_reply", {"reply_body": "Sounds good, see you then.", "rationale": "Direct question needing a reply."}
    )

    fake_service = make_fake_service({
        "threadId": "thread-123",
        "payload": {"headers": [{"name": "Message-ID", "value": "<orig-123@mail.gmail.com>"}]},
    })
    fake_drafts_create = fake_service.users().drafts().create
    fake_drafts_create().execute.return_value = {"id": "draft-1"}

    result = decide_and_act(
        fake_client, "gemini-2.5-flash", fake_service, "msg-1",
        "Jane Doe <jane@example.com>", "Lunch tomorrow?",
        "Are you free for lunch tomorrow at noon?",
        {"category": "Personal", "reason": "Direct question"}
    )

    assert "draft_reply" in result
    assert "Created draft" in result
    assert fake_drafts_create.called
    print("PASS: draft_reply path calls drafts().create() and never calls send()")
    assert not fake_service.users().messages().send.called
    print("PASS: send() is never called — drafts only, as required")


def test_flag_for_scheduling_path_no_calendar_configured():
    """
    With no credentials.json present (as in this sandboxed test run),
    calendar_tools.get_proposed_slots_text() should fail gracefully and
    the agent should still produce a draft, just falling back to asking
    the sender for their availability instead of crashing.
    """
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = make_fake_genai_response(
        "flag_for_scheduling", {"acknowledgement": "Happy to find a time."}
    )
    fake_service = make_fake_service({
        "threadId": "thread-456",
        "payload": {"headers": [{"name": "Message-ID", "value": "<orig-456@mail.gmail.com>"}]},
    })
    fake_drafts_create = fake_service.users().drafts().create

    result = decide_and_act(
        fake_client, "gemini-2.5-flash", fake_service, "msg-2",
        "Bob <bob@example.com>", "Quick call?",
        "Do you have time for a call this week?",
        {"category": "Meeting Request", "reason": "Scheduling ask"}
    )
    assert "flag_for_scheduling" in result
    assert "Created draft" in result
    assert fake_drafts_create.called
    # Inspect what was actually drafted to confirm the fallback text was used
    sent_body = fake_drafts_create.call_args.kwargs["body"]["message"]["raw"]
    decoded = base64.urlsafe_b64decode(sent_body).decode()
    assert "share a couple of times" in decoded
    print("PASS: flag_for_scheduling falls back to asking sender when calendar isn't available")


def test_flag_for_scheduling_path_with_calendar_slots():
    """
    With calendar_tools.get_proposed_slots_text mocked to return real
    slots, the drafted reply should include those specific times rather
    than the generic fallback ask.
    """
    import react_actions as react_actions_module

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = make_fake_genai_response(
        "flag_for_scheduling", {"acknowledgement": "Happy to find a time."}
    )
    fake_service = make_fake_service({
        "threadId": "thread-777",
        "payload": {"headers": [{"name": "Message-ID", "value": "<orig-777@mail.gmail.com>"}]},
    })
    fake_drafts_create = fake_service.users().drafts().create

    original_get_slots = react_actions_module.calendar_tools.get_proposed_slots_text
    react_actions_module.calendar_tools.get_proposed_slots_text = lambda *a, **k: (
        "- Tuesday, July 01 at 10:00 AM - 10:30 AM (UTC)\n- Tuesday, July 01 at 2:00 PM - 2:30 PM (UTC)"
    )
    try:
        result = decide_and_act(
            fake_client, "gemini-2.5-flash", fake_service, "msg-6",
            "Bob <bob@example.com>", "Quick call?",
            "Do you have time for a call this week?",
            {"category": "Meeting Request", "reason": "Scheduling ask"}
        )
    finally:
        react_actions_module.calendar_tools.get_proposed_slots_text = original_get_slots

    assert fake_drafts_create.called
    sent_body = fake_drafts_create.call_args.kwargs["body"]["message"]["raw"]
    decoded = base64.urlsafe_b64decode(sent_body).decode()
    assert "10:00 AM" in decoded
    assert "2:00 PM" in decoded
    print("PASS: flag_for_scheduling embeds real calendar slots when available")


def test_no_action_path():
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = make_fake_genai_response(
        "no_action_needed", {"reason": "This is an automated receipt, no reply expected."}
    )
    fake_service = make_fake_service({
        "threadId": "thread-789",
        "payload": {"headers": [{"name": "Message-ID", "value": "<orig-789@mail.gmail.com>"}]},
    })

    result = decide_and_act(
        fake_client, "gemini-2.5-flash", fake_service, "msg-3",
        "Billing <billing@example.com>", "Your receipt",
        "Thanks for your payment of $9.99.",
        {"category": "Notification", "reason": "Automated receipt"}
    )
    assert "no_action_needed" in result
    print("PASS: no_action_needed path takes no drafting action")


def test_existing_draft_skips_new_draft():
    """
    Core regression test for the duplicate-draft bug: if a draft already
    exists on the thread, decide_and_act must NOT call drafts().create()
    again — it should only review the existing draft via text.
    """
    fake_client = MagicMock()
    # This response is used for the "review existing draft" prompt —
    # decide_and_act should never reach the tool-calling prompt at all
    # once an existing draft is found, so we give it a plain text reply.
    fake_review_response = MagicMock()
    fake_review_response.text = "OK: The draft still answers the sender's question."
    fake_client.models.generate_content.return_value = fake_review_response

    existing = [{
        "id": "existing-draft-1",
        "threadId": "thread-999",
        "full": {
            "id": "existing-draft-1",
            "message": {
                "threadId": "thread-999",
                "payload": {
                    "parts": [{
                        "mimeType": "text/plain",
                        "body": {"data": base64.urlsafe_b64encode(b"Sounds great, see you then!").decode()}
                    }]
                }
            }
        }
    }]

    fake_service = make_fake_service(
        {
            "threadId": "thread-999",
            "payload": {"headers": [{"name": "Message-ID", "value": "<orig-999@mail.gmail.com>"}]},
        },
        existing_drafts=existing,
    )
    fake_drafts_create = fake_service.users().drafts().create

    result = decide_and_act(
        fake_client, "gemini-2.5-flash", fake_service, "msg-4",
        "Jane Doe <jane@example.com>", "Lunch tomorrow?",
        "Are you free for lunch tomorrow at noon?",
        {"category": "Personal", "reason": "Direct question"}
    )

    assert not fake_drafts_create.called, "drafts().create() must NOT be called when a draft already exists"
    assert "existing_draft" in result
    assert "existing-draft-1" in result
    assert "OK:" in result
    print("PASS: existing draft on thread blocks creation of a duplicate draft")
    print("PASS: model is asked to review the existing draft text instead")


def test_missing_message_id_skips_draft_with_warning():
    """
    Regression test for the phantom-draft cause: if the original email
    has no Message-ID header, _create_draft must refuse to draft rather
    than silently creating an unanchored draft.
    """
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = make_fake_genai_response(
        "draft_reply", {"reply_body": "Hello!", "rationale": "Needs a reply."}
    )

    fake_service = make_fake_service({
        "threadId": "thread-no-msgid",
        "payload": {"headers": []},  # no Message-ID header present
    })
    fake_drafts_create = fake_service.users().drafts().create

    result = decide_and_act(
        fake_client, "gemini-2.5-flash", fake_service, "msg-5",
        "Someone <someone@example.com>", "Hi",
        "Just checking in.",
        {"category": "Personal", "reason": "Generic"}
    )

    assert not fake_drafts_create.called, "drafts().create() must NOT be called without a Message-ID header"
    assert "Skipped" in result
    print("PASS: missing Message-ID header prevents draft creation (no phantom draft)")


if __name__ == "__main__":
    test_extract_email_address()
    test_draft_reply_path()
    test_flag_for_scheduling_path_no_calendar_configured()
    test_flag_for_scheduling_path_with_calendar_slots()
    test_no_action_path()
    test_existing_draft_skips_new_draft()
    test_missing_message_id_skips_draft_with_warning()
    print("\nAll tests passed!")