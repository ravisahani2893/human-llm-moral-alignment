from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY
from app.llm_logger import log_llm_call
import sys
import time
from google.genai.errors import ClientError
from google import genai



client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-2.5-flash"


def ask_gemini(prompt):
    print(f"[gemini_client] calling Gemini with model={GEMINI_MODEL!r}", file=sys.stderr)
    while True:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    http_options=types.HttpOptions(timeout=60000),
                ),
            )
            log_llm_call("gemini", GEMINI_MODEL, prompt, status="success", response=response.text)
            return response.text

        except ClientError as e:
            log_llm_call("gemini", GEMINI_MODEL, prompt, status="error", error=str(e))
            if "429" in str(e):
                print(e)
                print("Rate limit reached. Waiting 15 seconds...")
                time.sleep(10)
            else:
                raise
        except Exception as e:
            log_llm_call("gemini", GEMINI_MODEL, prompt, status="error", error=str(e))
            print(f"[gemini_client] call failed/timed out: {e}. Retrying in 10 seconds...", file=sys.stderr)
            time.sleep(10)