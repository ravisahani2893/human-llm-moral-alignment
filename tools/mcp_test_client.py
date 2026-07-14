import argparse
import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def run(scenario: str, models: list[str]):
    server_params = StdioServerParameters(
        command="python3",
        args=["-m", "mcp_server.server"],
        cwd=PROJECT_ROOT,
    )

    async def log_handler(params):
        print(f"  [log:{params.level}] {params.data}")

    async with stdio_client(server_params) as (read, write):
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


def main():
    parser = argparse.ArgumentParser(
        description="Call evaluate_moral_scenario on the MCP server for one or more models."
    )
    parser.add_argument("--scenario", required=True, help="Moral scenario text to evaluate")
    parser.add_argument(
        "--models",
        default="gemini,lama,deep-seek,gpt-github",
        help="Comma-separated model keys (default: gemini,lama,deep-seek,gpt-github)",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    asyncio.run(run(args.scenario, models))


if __name__ == "__main__":
    main()
