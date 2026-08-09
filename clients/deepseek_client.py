import sys
import time

from openai import OpenAI, RateLimitError

from app.config import DEEPSEEK_API_KEY

DEEPSEEK_ENDPOINT = "https://api.deepseek.com"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=DEEPSEEK_ENDPOINT, api_key=DEEPSEEK_API_KEY, timeout=60.0)
    return _client


def ask_deepseek(prompt: str, model: str = "deepseek-v4-pro"):
    print(f"[deepseek_client] calling DeepSeek with model={model!r}", file=sys.stderr)

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
                raise RuntimeError(f"DeepSeek account is out of credits: {e}") from e
            print(e)
            print("Rate limit reached. Waiting 15 seconds...")
            time.sleep(15)
