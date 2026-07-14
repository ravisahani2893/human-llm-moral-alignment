import asyncio
import sys

import pandas as pd
from mcp.server.fastmcp import FastMCP, Context
from app.models import MoralValenceResponse

from app.evaluator import (
    MODEL_LABELS,
    evaluate_single,
    evaluate_random,
    evaluate_dataset as evaluate_complete_dataset
)
from app.jobs import get_job, start_job
from app.metrics import compute_report

mcp = FastMCP("Human LLM Moral Alignment")

@mcp.tool()
async def evaluate_moral_scenario(
    scenario: str,
    model: str = "gemini",
    ctx: Context = None
):
    """
    Evaluate a single moral scenario.

    model must be one of: "gemini", "lama", "deep-seek", "gpt-github".
    """
    await ctx.info(f"evaluate_moral_scenario called with model={model!r}")
    prediction = evaluate_single(scenario, model=model)
    return MoralValenceResponse(
        action_valence=prediction["action_valence"],
        action_reasoning=prediction.get("action_reasoning", ""),
        action_factors=prediction.get("action_factors", []),
        consequence_valence=prediction["consequence_valence"],
        consequence_reasoning=prediction.get("consequence_reasoning", ""),
        consequence_factors=prediction.get("consequence_factors", []),
    )

@mcp.tool()
async def evaluate_random_scenarios(sample_size: int = 10, model: str = "gemini", ctx: Context = None):
    """
    Evaluate random moral scenarios from the dataset using the given model.
    """
    await ctx.info(f"evaluate_random_scenarios called with model={model!r}, sample_size={sample_size}")
    return evaluate_random(sample_size, model=model)


@mcp.tool()
async def evaluate_dataset(model: str = "gemini", ctx: Context = None):
    """
    Evaluate the complete dataset using the given model.
    """
    await ctx.info(f"evaluate_dataset called with model={model!r}")
    return evaluate_complete_dataset(model=model)


@mcp.tool()
async def run_multi_model_evaluation(
    models: list[str],
    sample_size: int | None = None,
    ctx: Context = None,
):
    """
    Run several models against the dataset (or a sample of it) in parallel
    and wait for the run to finish.

    models must be a subset of: "gemini", "lama", "deep-seek", "gpt-github".
    sample_size limits the run to the first N scenarios; omit it to run the
    full dataset (slow, and consumes real API quota for every model).

    Returns the job id, the CSV path it was written to, and per-model
    progress/error counts. Pass the CSV path to compute_alignment_metrics
    to analyze the results, or read the CSV directly to inspect individual
    scenario reasoning.
    """
    await ctx.info(f"run_multi_model_evaluation called with models={models}, sample_size={sample_size}")
    job = start_job(models=models, sample_size=sample_size)

    while job.status == "running":
        await asyncio.sleep(1)

    snapshot = job.snapshot()
    snapshot["csv_path"] = str(job.csv_path)
    return snapshot


@mcp.tool()
async def compute_alignment_metrics(
    csv_path: str,
    models: list[str],
    ctx: Context = None,
):
    """
    Compute human-vs-model alignment metrics from a multi-model results CSV
    (produced by run_multi_model_evaluation, or downloaded from the web app's
    batch-run CSV export).

    Returns, per model: Pearson/Spearman correlation, MAE, RMSE, sign
    agreement, and mean bias for the action axis, consequence axis, and both
    combined; pairwise cross-model correlation; and a breakdown by dataset
    metadata (pattern/source/input_type) showing where each model diverges
    most from human labels.

    models must be a subset of: "gemini", "lama", "deep-seek", "gpt-github".
    """
    await ctx.info(f"compute_alignment_metrics called with csv_path={csv_path!r}, models={models}")
    df = pd.read_csv(csv_path)
    model_labels = [MODEL_LABELS.get(m, m) for m in models]
    return compute_report(df, model_labels)


@mcp.tool()
async def get_job_status(job_id: str, ctx: Context = None):
    """
    Check the status/progress of a job started by run_multi_model_evaluation
    in this server process (job ids from the separate web app are not visible
    here — use its own /api/jobs/{id} endpoint for those).
    """
    await ctx.info(f"get_job_status called with job_id={job_id!r}")
    job = get_job(job_id)
    if job is None:
        return {"error": f"No job found with id {job_id!r} in this server process."}
    snapshot = job.snapshot()
    snapshot["csv_path"] = str(job.csv_path)
    return snapshot


if __name__ == "__main__":
    print("FastMCP server started...", file=sys.stderr)
    mcp.run()
