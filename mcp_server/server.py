from mcp.server.fastmcp import FastMCP
from app.prompts import build_prompt
from app.gemini_client import ask_gemini
from app.models import MoralValenceResponse
import json

from app.evaluator import (
    evaluate_single,
    evaluate_random,
    evaluate_dataset as evaluate_complete_dataset
)

mcp = FastMCP("Human LLM Moral Alignment")

@mcp.tool()
def evaluate_moral_scenario(scenario: str) -> MoralValenceResponse:
    """
    Evaluate Action and Consequence Moral Valence.
    """
    print(f"Evaluating scenario: {scenario}")
    prompt = build_prompt(scenario)

    response = ask_gemini(prompt)

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