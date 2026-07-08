from mcp.server.fastmcp import FastMCP
from app.prompts import build_prompt
from clients.gemini_client import ask_gemini
from clients.openai_client import ask_gpt
from app.models import MoralValenceResponse
import json

from app.evaluator import (
    evaluate_single,
    evaluate_random,
    evaluate_dataset as evaluate_complete_dataset
)

mcp = FastMCP("Human LLM Moral Alignment")

@mcp.tool()
def evaluate_moral_scenario(
    scenario: str,
    model: str = "gemini"
):
    print("=" * 50)
    print(f"Model parameter received: {model}")
    print("=" * 50)

    prompt = build_prompt(scenario)

    if model.lower() == "gemini":
        print(">>> Calling Gemini")
        response = ask_gemini(prompt)

    elif model.lower() == "gpt":
        print(">>> Calling GPT")
        response = ask_gpt(prompt)

    else:
        raise ValueError(f"Unsupported model: {model}")

    prediction= json.loads(response)
    return MoralValenceResponse(
        action_valence=prediction["action_valence"],
        consequence_valence=prediction["consequence_valence"],
    )

@mcp.tool()
def evaluate_random_scenarios(sample_size: int = 10):
    """
    Evaluate random moral scenarios from the dataset using Gemini.
    """
    return evaluate_random(sample_size)


@mcp.tool()
def evaluate_dataset():
    """
    Evaluate the complete dataset using Gemini.
    """
    return evaluate_complete_dataset()

if __name__ == "__main__":
    print("FastMCP server started...")
    mcp.run()