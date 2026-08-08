import sys
import time

from openai import OpenAI, RateLimitError

from app.config import GROQ_API_KEY

GROQ_ENDPOINT = "https://api.groq.com/openai/v1"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=GROQ_ENDPOINT, api_key=GROQ_API_KEY, timeout=60.0)
    return _client


def ask_groq(prompt: str, model: str = "llama-3.3-70b-versatile"):
    print(f"[groq_client] calling Groq with model={model!r}", file=sys.stderr)

    while True:
        try:
            response = _get_client().chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            if "insufficient_quota" in str(e) or "credit_balance_exhausted" in str(e):
                raise RuntimeError(f"Groq account is out of credits: {e}") from e
            print(e)
            print("Rate limit reached. Waiting 15 seconds...")
            time.sleep(15)
