"""
One plain-text log file per process session, in logs/, named by the
timestamp the session started — every LLM request/response/error made by
that process gets appended to the same file for its whole lifetime.
Every attempt is logged separately, including retried attempts after a
rate limit or transient error, not just the final outcome — so a retry
storm (like the Groq rate-limit pile-up that stalled a job earlier in this
project) is fully visible after the fact just by reading the log, without
needing to have been watching stderr live when it happened.
"""
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Determined once, at first import in this process — every call from this
# process (api.py, mcp_server.server, a CLI script, ...) appends to the
# same file for as long as the process lives.
_SESSION_STARTED_AT = datetime.now(timezone.utc)
SESSION_LOG_PATH = LOG_DIR / f"{_SESSION_STARTED_AT.strftime('%Y%m%dT%H%M%S%f')}.log"


def log_llm_call(provider: str, model: str, prompt: str, status: str, response: str = None, error: str = None):
    """
    status must be "success" or "error". response is the raw model output
    text (success only); error is the exception message (error only).
    """
    now = datetime.now(timezone.utc)
    lines = [
        f"[{now.isoformat()}] {status.upper()} {provider}/{model}",
        f"PROMPT: {prompt}",
    ]
    if response is not None:
        lines.append(f"RESPONSE: {response}")
    if error is not None:
        lines.append(f"ERROR: {error}")
    lines.append("-" * 80)

    with open(SESSION_LOG_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")
