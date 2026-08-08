"""
CLI tool: trigger a per-model dataset export as a real MCP client — spawns
mcp_server/server.py as a subprocess and calls its start_dataset_export /
get_export_status tools over the actual MCP protocol (stdio), instead of
importing app.export_jobs directly. This is the "I run my experimentation
via MCP" path for the demo.

Usage:
    python -m tools.trigger_export_via_mcp --model gpt-github
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLL_INTERVAL_SECONDS = 5


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=str(PROJECT_ROOT / ".venv" / "bin" / "python3"),
        args=["-m", "mcp_server.server"],
        cwd=str(PROJECT_ROOT),
    )


def _extract_text(content_items) -> str:
    return "\n".join(c.text for c in content_items if hasattr(c, "text"))


async def trigger_export(model: str):
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print(f"[mcp] calling start_dataset_export(model={model!r})", file=sys.stderr)
            result = await session.call_tool("start_dataset_export", arguments={"model": model})
            snapshot = json.loads(_extract_text(result.content))
            job_id = snapshot["id"]
            print(f"[mcp] job started: {job_id}", file=sys.stderr)

            while snapshot["status"] == "running":
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                result = await session.call_tool("get_export_status", arguments={"job_id": job_id})
                snapshot = json.loads(_extract_text(result.content))
                print(f"[mcp] {snapshot['completed']}/{snapshot['total']} ({snapshot['status']})", file=sys.stderr)

            if snapshot["status"] == "completed":
                print(f"[mcp] done: {snapshot['csv_path']}", file=sys.stderr)
            else:
                print(f"[mcp] failed: {snapshot.get('error')}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Trigger a dataset export via the MCP server (real MCP protocol call).")
    parser.add_argument("--model", required=True, help="Model key: gemini, lama, deep-seek, gpt-github, claude")
    args = parser.parse_args()
    asyncio.run(trigger_export(args.model))


if __name__ == "__main__":
    main()
