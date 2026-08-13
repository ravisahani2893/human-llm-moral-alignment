import sys
import time

from openai import OpenAI, RateLimitError

from app.config import OPENAI_API_KEY
from app.llm_logger import log_llm_call

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)
    return _client


def ask_openai(prompt: str, model: str = "gpt-4.1"):
    print(f"[openai_client] calling OpenAI with model={model!r}", file=sys.stderr)

    while True:
        try:
            response = _get_client().chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content
            log_llm_call("openai", model, prompt, status="success", response=text)
            return text

        except RateLimitError as e:
            if "insufficient_quota" in str(e) or "credit_balance_exhausted" in str(e):
                log_llm_call("openai", model, prompt, status="error", error=str(e))
                raise RuntimeError(f"OpenAI account is out of credits: {e}") from e
            log_llm_call("openai", model, prompt, status="error", error=str(e))
            print(e)
            print("Rate limit reached. Waiting 15 seconds...")
            time.sleep(15)
