# Gmail Agent

An autonomous email agent that connects to Gmail via OAuth 2.0, uses
Gemini to triage unread email by priority, and — for anything flagged
high-priority — reasons about what to do next: draft a reply, flag it
for scheduling, or take no action. Replies are always saved as **drafts**,
never sent automatically.

## What it does

1. **Authenticate** with Gmail using OAuth 2.0 (one-time browser consent,
   then a cached/refreshed token for subsequent runs).
2. **Fetch** the most recent unread messages.
3. **Parse** each message — extract the plain-text body and save any
   attachments to `downloads/`.
4. **Triage** each email with Gemini: a priority score (1–10), a category
   (e.g. "Meeting Request", "Newsletter", "Urgent"), and a one-sentence
   reason.
5. **Label** the email in Gmail (`AI-Priority`, `AI-Medium`, or
   `AI-LowPriority`) based on the score.
6. **Reason and act** on high-priority emails: a ReAct-style step gives
   Gemini a small toolset — `draft_reply`, `flag_for_scheduling`, or
   `no_action_needed` — and lets the model decide which one applies,
   then actually executes it against the Gmail API.

There's also a standalone search mode: describe what you're looking for
in plain English, and the agent translates it into Gmail's search syntax
and runs it.

## Why drafts, not sends

The agent never sends mail on its own. `draft_reply` and
`flag_for_scheduling` both call Gmail's `drafts().create()` endpoint —
the draft shows up in your Gmail account, threaded correctly under the
original message, waiting for you to review and hit send (or not).

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/onuva/Gmail-Agent.git
cd Gmail-Agent
pip install -r requirements.txt
```

### 2. Get a Gemini API key

Create a key at [Google AI Studio](https://aistudio.google.com/apikey),
then:

```bash
cp .env.example .env
# edit .env and paste your key into G_API_KEY=
```

### 3. Enable the Gmail API and get `credentials.json`

1. Go to the [Google Cloud Console](https://console.cloud.google.com/),
   create (or select) a project.
2. Enable the **Gmail API** for that project.
3. Under **APIs & Services → Credentials**, create an **OAuth client ID**
   of type **Desktop app**.
4. Download the resulting JSON and save it in the project root as
   `credentials.json`.

`credentials.json` and the `token.json` it generates on first run both
contain sensitive material — they're already excluded via `.gitignore`
and should never be committed.

## Running it

```bash
python agent.py
```

On the first run, a browser window opens for you to sign in and grant
permission. After that, `token.json` is cached and refreshed
automatically — no repeated logins.

### Natural-language search

```bash
python agent.py --search "unread emails from my manager about the budget"
```

## Example output

```
Agent active. Checking for unread emails...

Processing: Q3 Budget Review — need your sign-off by Friday
  From: Dana Lee <dana@company.com>
  Score: 9/10 | Category: Urgent
  Reason: Requires a decision with an explicit deadline this week.
  [ACTION] Labeled as 'AI-Priority'
  [REACT->draft_reply] Created draft reply. (Direct question needing a timely response.)

Processing: Are you free for a quick call next week?
  From: Alex Rivera <alex@partner.io>
  Score: 8/10 | Category: Meeting Request
  Reason: Sender is requesting a scheduling decision.
  [ACTION] Labeled as 'AI-Priority'
  [REACT->flag_for_scheduling] Created draft proposing scheduling follow-up.

Processing: Your receipt from Acme Hosting
  From: billing@acmehosting.com
  Score: 2/10 | Category: Notification
  Reason: Automated transactional receipt, no response needed.
  [ACTION] Labeled as 'AI-LowPriority'

Batch complete.
```

## Project structure

```
agent.py          # Main pipeline: auth, fetch, triage, label, ReAct dispatch
react_actions.py  # Tool definitions + execution for the ReAct decision step
test.py           # Quick Gemini API connectivity check
requirements.txt
.env.example
```

## Configuration

A few constants at the top of `agent.py` control behavior:

| Constant                  | Default | Meaning                                |
|----------------------------|---------|-----------------------------------------|
| `MAX_EMAILS`              | 5       | Unread messages fetched per run         |
| `HIGH_PRIORITY_THRESHOLD` | 8       | Score at/above which an email is "high" and triggers the ReAct step |
| `LOW_PRIORITY_THRESHOLD`  | 3       | Score at/below which an email is "low"  |
| `MODEL`                   | `gemini-2.5-flash` | Gemini model used for triage and reasoning |

## Limitations

- Drafts only — nothing is ever sent without a human clicking send.
- Scheduling support proposes time windows in a draft reply; it does not
  read or write to Google Calendar.
- Designed for single-user, local/CLI use — there's no multi-account or
  server deployment story here.
