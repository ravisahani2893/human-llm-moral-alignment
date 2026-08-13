"""
General-purpose MCP agent.

A real LLM (Gemini) acts as the orchestrator: given a free-form natural
language instruction and the full live tool set discovered from this
project's MCP server (mcp_server/server.py) over the actual MCP protocol
(stdio, JSON-RPC) — not by importing the underlying Python functions
directly — the model decides which tools to call, in what order, and when
it has enough information to stop and respond. Unlike a fixed UI button or
a scripted task template, nothing here prescribes the sequence: the
instruction is handed to the model as-is.

Usage:
    python -m agents.general_agent --instruction "Export the entire dataset for deep-seek, then compute its alignment metrics."
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

from app.config import GEMINI_API_KEY

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GEMINI_MODEL = "gemini-2.5-flash"
MAX_TOOL_TURNS = 15
POLL_INTERVAL_SECONDS = 3
POLL_TIMEOUT_SECONDS = 7200  # full-dataset runs across several models can take a long time

SYSTEM_INSTRUCTION = """
You are an assistant for a moral-alignment research project, with access to
every tool, resource, and prompt exposed by this project's MCP server:
running model evaluations (single scenario, random sample, or the entire
dataset), starting and polling per-model dataset exports across multiple
prompt versions, computing alignment metrics, listing prompt versions, and
reading dataset/job/export resources.

This project asks two DIFFERENT questions — do not conflate them:
- Human-LLM Alignment (compute_alignment_metrics): how well does ONE
  model's predictions match human annotations? Human vs. model.
- Cross-Model Agreement (compute_cross_model_agreement): how similarly do
  TWO OR MORE models score the SAME scenarios COMPARED TO EACH OTHER? Model
  vs. model — it never reads or reports human annotations, only pairwise
  correlation between models' own predictions. If asked to compare models
  "against each other" or measure their "agreement"/"convergence"/
  "similarity," that is compute_alignment_metrics's twin, not a substitute
  for it — use compute_cross_model_agreement, and do not describe its
  output as measuring alignment with humans.

Given the user's instruction, decide for yourself which tools to call and
in what order — the instruction will not tell you the exact sequence.
Chain multiple tool calls when the task requires it (e.g. "export gemini
then show me its alignment metrics" means: call start_dataset_export,
wait for it to finish, then call compute_alignment_metrics). Read a tool's
description carefully before calling it — several tools return
immediately with a job id rather than waiting for completion; poll the
matching status tool (get_job_status / get_export_status) until it stops
running, unless the instruction only asked you to start something.

Use read_mcp_resource for any URI-addressed resource (e.g.
'jobs://{job_id}/csv' for a completed job's full results CSV including
model reasoning text, or 'dataset://scenarios/{id}' for one scenario's
text and human labels) — these are not regular tools.

If the instruction is ambiguous about which model(s) or prompt version to
use, make a reasonable choice and say so in your final response rather
than guessing silently.

When you have completed the instruction (or determined it can't be
completed), stop calling tools and write a clear final response: what you
did, the actual results/numbers (not just "done"), and anything the user
should know (errors, partial completion, assumptions you made). Producing
plain text with no further function calls signals you are done.
""".strip()


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=str(PROJECT_ROOT / ".venv" / "bin" / "python3"),
        args=["-m", "mcp_server.server"],
        cwd=str(PROJECT_ROOT),
    )


def _extract_text(content_items) -> str:
    return "\n".join(c.text for c in content_items if hasattr(c, "text"))


async def _wait_for_job(session: ClientSession, status_tool: str, id_field: str, start_result_text: str, on_step=None) -> str:
    """
    Poll a status tool (get_job_status / get_export_status) via real MCP
    tool calls until the job leaves "running", without spending the agent's
    own reasoning turns on it — waiting is mechanical, not a decision worth
    an LLM round trip.
    """
    snapshot = json.loads(start_result_text)
    job_id = snapshot.get("id")
    if not job_id:
        return start_result_text

    elapsed = 0
    while snapshot.get("status") == "running" and elapsed < POLL_TIMEOUT_SECONDS:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS
        result = await session.call_tool(status_tool, arguments={id_field: job_id})
        snapshot = json.loads(_extract_text(result.content))
        print(f"  [agent] (auto-poll) {status_tool} {job_id}: {snapshot.get('status')}", file=sys.stderr)
        if on_step:
            on_step({
                "type": "job_progress",
                "job_id": job_id,
                "status": snapshot.get("status"),
                "completed": snapshot.get("completed") or snapshot.get("completed_per_model"),
                "total": snapshot.get("total"),
            })

    return json.dumps(snapshot)


# Tools that return immediately with a job id rather than waiting for
# completion — auto-polled so the agent doesn't burn turns on waiting.
_AUTO_POLL_TOOLS = {
    "start_multi_model_evaluation": ("get_job_status", "job_id"),
    "start_dataset_export": ("get_export_status", "job_id"),
}


async def run_agent(instruction: str, max_turns: int = MAX_TOOL_TURNS, on_step=None):
    """
    on_step, if given, is called synchronously with each transcript entry
    as soon as it's produced (task, tool_call, tool_result, job_progress,
    final_report, max_turns_reached) — lets a caller (e.g. the web UI) show
    a live activity log instead of only seeing the result once done.
    """
    def emit(entry: dict):
        transcript.append(entry)
        if on_step:
            on_step(entry)

    client = genai.Client(api_key=GEMINI_API_KEY)
    transcript: list[dict] = []
    emit({"type": "task", "text": instruction})

    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools

            function_declarations = [
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description or "",
                    parametersJsonSchema=t.inputSchema,
                )
                for t in mcp_tools
            ]
            # MCP resources aren't tools, so give the agent one generic
            # wrapper tool to fetch any resource by URI.
            function_declarations.append(
                types.FunctionDeclaration(
                    name="read_mcp_resource",
                    description=(
                        "Read an MCP resource by URI, e.g. 'jobs://{job_id}/csv' for a completed "
                        "job's full results CSV (includes each model's reasoning text per scenario), "
                        "or 'dataset://scenarios/{id}' for one scenario's text and human labels."
                    ),
                    parametersJsonSchema={
                        "type": "object",
                        "properties": {"uri": {"type": "string"}},
                        "required": ["uri"],
                    },
                )
            )
            gemini_tools = [types.Tool(functionDeclarations=function_declarations)]

            contents = [types.Content(role="user", parts=[types.Part(text=instruction)])]

            for turn in range(max_turns):
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        tools=gemini_tools,
                    ),
                )
                candidate = response.candidates[0]
                contents.append(candidate.content)

                function_calls = [
                    part.function_call for part in candidate.content.parts
                    if part.function_call is not None
                ]

                if not function_calls:
                    final_text = "".join(
                        part.text for part in candidate.content.parts if part.text
                    )
                    emit({"type": "final_report", "turn": turn, "text": final_text})
                    return final_text, transcript

                response_parts = []
                for fc in function_calls:
                    args = dict(fc.args) if fc.args else {}
                    print(f"  [agent] turn {turn}: calling {fc.name}({args})", file=sys.stderr)
                    emit({"type": "tool_call", "turn": turn, "name": fc.name, "args": args})

                    try:
                        if fc.name == "read_mcp_resource":
                            resource_result = await session.read_resource(AnyUrl(args["uri"]))
                            result_text = _extract_text(resource_result.contents)
                        else:
                            result = await session.call_tool(fc.name, arguments=args)
                            result_text = _extract_text(result.content)
                            if result.isError:
                                result_text = json.dumps({"error": result_text})
                            elif fc.name in _AUTO_POLL_TOOLS:
                                status_tool, id_field = _AUTO_POLL_TOOLS[fc.name]
                                result_text = await _wait_for_job(session, status_tool, id_field, result_text, on_step=emit)
                    except Exception as exc:
                        result_text = json.dumps({"error": str(exc)})

                    print(f"  [agent] turn {turn}: {fc.name} -> {result_text[:200]}", file=sys.stderr)
                    emit({"type": "tool_result", "turn": turn, "name": fc.name, "result": result_text[:4000]})

                    response_parts.append(
                        types.Part.from_function_response(name=fc.name, response={"result": result_text})
                    )

                contents.append(types.Content(role="user", parts=response_parts))

            emit({"type": "max_turns_reached", "turns": max_turns})
            return "(Agent stopped: reached max tool-call turns without producing a final response.)", transcript


def extract_csv_path(transcript: list[dict]) -> str | None:
    """
    Pull the most recent CSV path mentioned anywhere in a tool result (job
    or export status), so the caller can offer a download even though the
    agent's own response is prose, not raw data.
    """
    csv_path = None
    for entry in transcript:
        if entry.get("type") != "tool_result":
            continue
        try:
            data = json.loads(entry["result"])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and data.get("csv_path"):
            csv_path = data["csv_path"]
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Run the general-purpose MCP agent with a free-form instruction.")
    parser.add_argument("--instruction", required=True, help="Free-form natural language instruction for the agent")
    parser.add_argument("--max-turns", type=int, default=MAX_TOOL_TURNS, help="Max tool-call turns before giving up")
    args = parser.parse_args()

    print(f"[agent] instruction: {args.instruction}\n", file=sys.stderr)
    report, transcript = asyncio.run(run_agent(args.instruction, max_turns=args.max_turns))
    print(report)


if __name__ == "__main__":
    main()
