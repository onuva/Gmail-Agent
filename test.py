"""
Quick connectivity smoke test — confirms the Gemini API key in .env
is valid and the model responds, before running the full agent.
"""
import os
import sys
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("G_API_KEY")
if not api_key:
    print("FAIL: G_API_KEY not set in .env")
    sys.exit(1)

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say a greeting in one sentence."
    )
    if response.text:
        print(f"PASS: Gemini responded: {response.text.strip()}")
    else:
        print("FAIL: Gemini returned an empty response.")
        sys.exit(1)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)