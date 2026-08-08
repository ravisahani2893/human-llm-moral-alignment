import argparse
import asyncio
import json
import os
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _server_params():
    return StdioServerParameters(
        command="python3",
        args=["-m", "mcp_server.server"],
        cwd=PROJECT_ROOT,
    )


async def run(scenario: str, models: list[str]):
    async def log_handler(params):
        print(f"  [log:{params.level}] {params.data}")

    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write, logging_callback=log_handler) as session:
            await session.initialize()

            for model in models:
                print(f"\n=== {model} ===")
                result = await session.call_tool(
                    "evaluate_moral_scenario",
                    arguments={"scenario": scenario, "model": model},
                )
                for content in result.content:
                    print(content.text)


def _check(label: str, condition: bool, detail: str = ""):
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {label}" + (f" — {detail}" if detail and not condition else ""))
    return condition


async def smoke_test(sample_size: int = 2, models: list[str] = None):
    """
    Exercises the real MCP protocol (over stdio, actual JSON-RPC — not a
    direct Python import) against every tool/resource/prompt added in the
    latest round of fixes: non-blocking job start, cross-process-style
    status polling, typed metrics output, resources, and the rubric prompt.
    """
    models = models or ["gemini"]
    all_ok = True

    async def log_handler(params):
        pass

    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write, logging_callback=log_handler) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            tool_names = {t.name for t in tools}
            all_ok &= _check(
                "expected tools registered",
                {"start_multi_model_evaluation", "get_job_status", "list_recent_jobs",
                 "start_dataset_export", "get_export_status"} <= tool_names,
                f"found {sorted(tool_names)}",
            )

            resources = (await session.list_resources()).resources
            templates = (await session.list_resource_templates()).resourceTemplates
            resource_uris = {str(r.uri) for r in resources} | {t.uriTemplate for t in templates}
            all_ok &= _check(
                "expected resources registered",
                {"dataset://scenarios/count", "jobs://recent", "dataset://scenarios/{scenario_id}"} <= resource_uris,
                f"found {sorted(resource_uris)}",
            )

            prompts = (await session.list_prompts()).prompts
            all_ok &= _check(
                "rubric prompt registered",
                any(p.name == "moral_valence_scoring_rubric" for p in prompts),
            )

            print("\n--- starting job (should return immediately, not block) ---")
            t0 = time.time()
            result = await session.call_tool(
                "start_multi_model_evaluation",
                arguments={"models": models, "sample_size": sample_size},
            )
            elapsed = time.time() - t0
            snapshot = json.loads(result.content[0].text)
            job_id = snapshot["id"]
            all_ok &= _check("start_multi_model_evaluation returned quickly", elapsed < 5, f"took {elapsed:.1f}s")
            all_ok &= _check("job status is 'running' right after start", snapshot["status"] == "running")
            print(f"  job_id={job_id}")

            print("\n--- polling get_job_status until completion ---")
            status = "running"
            for _ in range(60):
                result = await session.call_tool("get_job_status", arguments={"job_id": job_id})
                snapshot = json.loads(result.content[0].text)
                status = snapshot["status"]
                print(f"  status={status} completed_per_model={snapshot['completed_per_model']}")
                if status != "running":
                    break
                await asyncio.sleep(3)
            all_ok &= _check("job reached 'completed'", status == "completed", f"ended as {status!r}")

            print("\n--- list_recent_jobs shows this job ---")
            result = await session.call_tool("list_recent_jobs", arguments={})
            job_ids = {json.loads(c.text)["id"] for c in result.content}
            all_ok &= _check("job visible in list_recent_jobs", job_id in job_ids)

            print("\n--- reading dataset resource ---")
            from pydantic import AnyUrl
            resource_result = await session.read_resource(AnyUrl("dataset://scenarios/count"))
            count_text = resource_result.contents[0].text if resource_result.contents else ""
            all_ok &= _check("dataset scenario count resource readable", count_text.isdigit(), count_text)

            print("\n--- reading rubric prompt ---")
            prompt_result = await session.get_prompt("moral_valence_scoring_rubric", arguments={"scenario": "test"})
            all_ok &= _check("rubric prompt returns content", len(prompt_result.messages) > 0)

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Test the MCP server: either evaluate one scenario across models, "
                     "or run a full smoke test of tools/resources/prompts with --smoke-test."
    )
    parser.add_argument("--scenario", help="Moral scenario text to evaluate")
    parser.add_argument(
        "--models",
        default="gemini,lama,deep-seek,gpt-github",
        help="Comma-separated model keys (default: gemini,lama,deep-seek,gpt-github)",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run the full smoke test instead")
    parser.add_argument("--sample-size", type=int, default=2, help="Scenarios to use in the smoke test job")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.smoke_test:
        ok = asyncio.run(smoke_test(sample_size=args.sample_size, models=models))
        raise SystemExit(0 if ok else 1)

    if not args.scenario:
        parser.error("--scenario is required unless --smoke-test is passed")
    asyncio.run(run(args.scenario, models))


if __name__ == "__main__":
    main()
