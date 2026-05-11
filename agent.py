import os
import base64
import json
import time
from dotenv import load_dotenv

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google import genai
from google.genai import types

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
MODEL = "gemini-2.5-flash"
MAX_EMAILS = 5
HIGH_PRIORITY_THRESHOLD = 8
LOW_PRIORITY_THRESHOLD = 3
HIGH_PRIORITY_LABEL = "AI-Priority"
MED_PRIORITY_LABEL = "AI-Medium"
LOW_PRIORITY_LABEL = "AI-LowPriority"

ALL_AI_LABELS = [HIGH_PRIORITY_LABEL, MED_PRIORITY_LABEL, LOW_PRIORITY_LABEL]

G_API_KEY = os.getenv("G_API_KEY")
if not G_API_KEY:
    raise ValueError("G_API_KEY not found in .env file.")

client = genai.Client(api_key=G_API_KEY)

def get_gmail_service():
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def get_email_body(payload):
    body = ""
    if "parts" in payload:
        for part in payload["parts"]:
            if part["mimeType"] == "text/plain":
                data = part["body"].get("data")
                if data:
                    body += base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
            elif "parts" in part:
                body += get_email_body(part)
    else:
        data = payload.get("body", {}).get("data")
        if data:
            body = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    return body


def download_attachments(service, message_id, payload):
    if "parts" not in payload:
        return
    for part in payload["parts"]:
        filename = part.get("filename")
        if not filename:
            continue
        attachment_id = part["body"].get("attachmentId")
        if not attachment_id:
            continue
        attachment = service.users().messages().attachments().get(
            userId="me", messageId=message_id, id=attachment_id
        ).execute()
        file_data = base64.urlsafe_b64decode(attachment["data"].encode("UTF-8"))
        os.makedirs("downloads", exist_ok=True)
        path = os.path.join("downloads", filename)
        with open(path, "wb") as f:
            f.write(file_data)
        print(f"    Saved attachment: {filename}")


def get_or_create_label(service, label_name):
    labels_response = service.users().labels().list(userId="me").execute()
    for label in labels_response.get("labels", []):
        if label["name"] == label_name:
            return label["id"]
    # Label doesn't exist — create it
    new_label = service.users().labels().create(
        userId="me",
        body={"name": label_name, "labelListVisibility": "labelShow", "messageListVisibility": "show"}
    ).execute()
    print(f"    Created label: '{label_name}'")
    return new_label["id"]


def clear_ai_labels(service, message_id, all_label_ids):
    existing = service.users().messages().get(
        userId="me", id=message_id, format="minimal"
    ).execute().get("labelIds", [])

    to_remove = [lid for lid in all_label_ids if lid in existing]
    if to_remove:
        service.users().messages().modify(
            userId="me",
            id=message_id,
            body={"removeLabelIds": to_remove}
        ).execute()
        return True  # signals previous label was replaced
    return False


def apply_label(service, message_id, label_id):
    service.users().messages().modify(
        userId="me",
        id=message_id,
        body={"addLabelIds": [label_id]}
    ).execute()

def triage_email(subject, body, retries=3):
    prompt = f""" Analyze the following email and determine its priority on a scale of 1 to 10.
10 = Urgent / Action Required, 1 = Junk / Newsletter / Spam.

Subject: {subject}
Body: {body[:2000]}

Return ONLY a valid JSON object with no extra text:
{{
  "priority_score": <int 1-10>,
  "category": "<string e.g. Meeting Request, Newsletter, Urgent, Personal, Notification>",
  "reason": "<one sentence explaining the score>"
}}
"""
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)

        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < retries:
                wait = 30 * attempt  # 30s, 60s, 90s
                print(f"    [Rate limit] Waiting {wait}s before retrying {attempt}/{retries}...")
                time.sleep(wait)
            else:
                raise



def run_agent():
    service = get_gmail_service()
    print("Agent active. Checking for unread emails...\n")

    # Pre-fetch all label IDs once (avoids repeated API calls)
    high_label_id = get_or_create_label(service, HIGH_PRIORITY_LABEL)
    med_label_id  = get_or_create_label(service, MED_PRIORITY_LABEL)
    low_label_id  = get_or_create_label(service, LOW_PRIORITY_LABEL)
    all_label_ids = [high_label_id, med_label_id, low_label_id]

    results = service.users().messages().list(
        userId="me", q="is:unread", maxResults=MAX_EMAILS
    ).execute()
    messages = results.get("messages", [])

    if not messages:
        print("No unread emails.")
        return

    for msg in messages:
        full_msg = service.users().messages().get(userId="me", id=msg["id"]).execute()
        payload = full_msg["payload"]
        headers = payload.get("headers", [])

        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown")

        body = get_email_body(payload)
        download_attachments(service, msg["id"], payload)

        print(f"Processing: {subject}")
        print(f"  From: {sender}")

        try:
            analysis = triage_email(subject, body)

            score = analysis.get("priority_score", 0)
            category = analysis.get("category", "Unknown")
            reason = analysis.get("reason", "N/A")

            print(f"  Score: {score}/10 | Category: {category}")
            print(f"  Reason: {reason}")

            # Strip any previous AI labels before applying the fresh one
            was_relabeled = clear_ai_labels(service, msg["id"], all_label_ids)

            if score >= HIGH_PRIORITY_THRESHOLD:
                apply_label(service, msg["id"], high_label_id)
                tag = f"[UPDATED → '{HIGH_PRIORITY_LABEL}']" if was_relabeled else f"[ACTION] Labeled as '{HIGH_PRIORITY_LABEL}'"
            elif score <= LOW_PRIORITY_THRESHOLD:
                apply_label(service, msg["id"], low_label_id)
                tag = f"[UPDATED → '{LOW_PRIORITY_LABEL}']" if was_relabeled else f"[ACTION] Labeled as '{LOW_PRIORITY_LABEL}'"
            else:
                apply_label(service, msg["id"], med_label_id)
                tag = f"[UPDATED → '{MED_PRIORITY_LABEL}']" if was_relabeled else f"[ACTION] Labeled as '{MED_PRIORITY_LABEL}'"

            print(f"  {tag}")

        except Exception as e:
            print(f"  [ERROR] Failed to process: {e}")

        print()

    print("Batch complete.")


if __name__ == "__main__":
    run_agent()