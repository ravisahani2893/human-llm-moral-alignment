"""
Persistent MCP client connection for api.py, so UI-triggered actions (like
starting a dataset export) go through the real MCP protocol — the same
mcp_server.server tools an MCP client like Claude Desktop or
tools/trigger_export_via_mcp.py would call — instead of importing the
underlying app modules directly.

The MCP server subprocess is spawned once at API startup and kept alive for
the API server's lifetime (mirrors how Claude Desktop holds one persistent
connection). A background thread runs its own asyncio event loop so a sync
FastAPI endpoint can call call_tool() and block for the result without
needing to become async itself.
"""
import asyncio
import json
import sys
import threading
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_loop: asyncio.AbstractEventLoop | None = None
_session: ClientSession | None = None
_ready = threading.Event()


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=str(PROJECT_ROOT / ".venv" / "bin" / "python3"),
        args=["-m", "mcp_server.server"],
        cwd=str(PROJECT_ROOT),
    )


def _extract_text(content_items) -> str:
    return "\n".join(c.text for c in content_items if hasattr(c, "text"))


async def _run():
    global _session
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            _session = session
            print("[mcp_client] persistent MCP connection ready", file=sys.stderr)
            _ready.set()
            await asyncio.Event().wait()  # keep the connection open until process exit


def _thread_main():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_until_complete(_run())


def start():
    """Start the persistent MCP client connection in a background thread. Call once at API startup."""
    thread = threading.Thread(target=_thread_main, daemon=True)
    thread.start()
    if not _ready.wait(timeout=15):
        raise RuntimeError("MCP client failed to connect within 15s")


def call_tool(name: str, arguments: dict) -> dict:
    """Call an MCP tool on the persistent connection, blocking until the result arrives."""
    if _session is None or _loop is None:
        raise RuntimeError("MCP client not started — call mcp_client.start() first")
    future = asyncio.run_coroutine_threadsafe(_session.call_tool(name, arguments=arguments), _loop)
    result = future.result(timeout=120)
    return json.loads(_extract_text(result.content))
