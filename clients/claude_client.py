import sys
import time

import anthropic

from app.config import ANTHROPIC_API_KEY
from app.llm_logger import log_llm_call

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=60.0)

CLAUDE_MODEL = "claude-sonnet-5"


def ask_claude(prompt):
    print(f"[claude_client] calling Claude with model={CLAUDE_MODEL!r}", file=sys.stderr)
    while True:
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            for block in response.content:
                if block.type == "text":
                    log_llm_call("claude", CLAUDE_MODEL, prompt, status="success", response=block.text)
                    return block.text
            raise ValueError(f"No text block in Claude response: {response.content!r}")

        except anthropic.RateLimitError as e:
            log_llm_call("claude", CLAUDE_MODEL, prompt, status="error", error=str(e))
            print(e)
            print("Rate limit reached. Waiting 15 seconds...")
            time.sleep(15)
        except Exception as e:
            log_llm_call("claude", CLAUDE_MODEL, prompt, status="error", error=str(e))
            print(f"[claude_client] call failed/timed out: {e}. Retrying in 10 seconds...", file=sys.stderr)
            time.sleep(10)
