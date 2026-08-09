import sys

from mcp.server.fastmcp import FastMCP, Context

from app.dataset import load_dataset
from app import evals
from app.jobs import OUTPUT_DIR, get_job, list_jobs, start_job
from app.export_jobs import get_export_job, list_export_jobs, start_export_job
from app.models import ExportJobSnapshot, JobSnapshot, ModelAlignmentReport, MoralValenceResponse
from app.prompt_versions import available_versions
from app.prompts import build_prompt

from app.evaluator import (
    MODEL_LABELS,
    evaluate_alignment,
    evaluate_single,
    evaluate_random,
    evaluate_dataset as evaluate_complete_dataset,
)

mcp = FastMCP("Human LLM Moral Alignment")


# ---- Tools ----

@mcp.tool()
async def evaluate_moral_scenario(
    scenario: str,
    model: str = "gemini",
    ctx: Context = None,
) -> MoralValenceResponse:
    """
    Evaluate a single moral scenario with one model.

    model must be one of: "gemini", "lama", "deep-seek", "gpt-github", "claude".
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
    Evaluate random moral scenarios from the dataset using one model
    (blocking, single-model only). For multi-model runs, use
    start_multi_model_evaluation instead.
    """
    await ctx.info(f"evaluate_random_scenarios called with model={model!r}, sample_size={sample_size}")
    return evaluate_random(sample_size, model=model)


@mcp.tool()
async def evaluate_dataset(model: str = "gemini", ctx: Context = None):
    """
    Evaluate the complete dataset using one model (blocking, single-model
    only, slow for the full ~500 scenarios). For multi-model runs, use
    start_multi_model_evaluation instead.
    """
    await ctx.info(f"evaluate_dataset called with model={model!r}")
    return evaluate_complete_dataset(model=model)


@mcp.tool()
async def start_multi_model_evaluation(
    models: list[str],
    sample_size: int | None = None,
    ctx: Context = None,
) -> JobSnapshot:
    """
    Start a background run of several models against the dataset (or a
    sample of it) in parallel. Returns immediately with a job id — it does
    NOT wait for the run to finish, since a full run can take many minutes
    and would otherwise block this tool call for that entire time.

    models must be a subset of: "gemini", "lama", "deep-seek", "gpt-github", "claude".
    sample_size limits the run to a random sample of N scenarios; omit it to
    run the full dataset (slow, and consumes real API quota for every model).

    After calling this, poll get_job_status(job_id) until status is
    "completed", then read the CSV directly to inspect scenario reasoning
    or compute your own alignment metrics against it. Jobs are visible from
    any process (web app or this MCP server) since status is persisted to
    disk, not just kept in memory.
    """
    await ctx.info(f"start_multi_model_evaluation called with models={models}, sample_size={sample_size}")
    job = start_job(models=models, sample_size=sample_size)
    return JobSnapshot(**job.snapshot())


@mcp.tool()
async def start_dataset_export(model: str, prompt_version: str = "current", ctx: Context = None) -> ExportJobSnapshot:
    """
    Start a background export of one model's raw predictions on the ENTIRE
    dataset to outputs/output_<model>_entire_dataset.csv (or
    outputs/output_<model>_<prompt_version>_entire_dataset.csv when
    prompt_version isn't "current") — no human comparison columns, just ID,
    Scenario, {model}_action, {model}_consequences, action_reasoning,
    consequences_reasoning. Meant as raw input for metrics computed
    separately later, once paired with human labels.

    Returns immediately with a job id — does NOT wait for the run to finish
    (a full 500-scenario run takes many minutes). Progressive and resumable:
    if a prior export for this model+prompt_version already exists, only the
    remaining scenarios are processed, so this is always safe to call again
    after an interruption.

    model must be one of: "gemini", "lama", "deep-seek", "gpt-github", "claude".
    prompt_version must be one of the values returned by list_prompt_versions
    (typically "v1", "v2", "few_shot", "current"). Using a non-"current"
    version writes to a separate file, so it never overwrites/mixes with an
    existing "current" export for the same model — this is how few-shot vs.
    zero-shot runs for the same model are kept directly comparable.
    Poll get_export_status(job_id) until status is "completed".
    """
    await ctx.info(f"start_dataset_export called with model={model!r}, prompt_version={prompt_version!r}")
    job = start_export_job(model=model, prompt_version=prompt_version)
    return ExportJobSnapshot(**job.snapshot())


@mcp.tool()
async def get_export_status(job_id: str, ctx: Context = None) -> ExportJobSnapshot:
    """
    Check progress of a dataset export job started by start_dataset_export,
    including jobs started from another process (status is persisted to
    disk, not process-local).
    """
    await ctx.info(f"get_export_status called with job_id={job_id!r}")
    job = get_export_job(job_id)
    if job is None:
        raise ValueError(f"No export job found with id {job_id!r}.")
    return ExportJobSnapshot(**job.snapshot())


@mcp.tool()
async def list_recent_exports(ctx: Context = None) -> list[ExportJobSnapshot]:
    """List all known dataset export jobs (most recent first), across processes."""
    await ctx.info("list_recent_exports called")
    return [ExportJobSnapshot(**snap) for snap in list_export_jobs()]


@mcp.tool()
async def compute_alignment_metrics(
    model: str,
    prompt_version: str = "current",
    ctx: Context = None,
) -> ModelAlignmentReport:
    """
    Compute how well one model's predictions align with your human
    annotations — Lin's CCC (primary metric, matching the source paper),
    MAE, RMSE, Pearson, and Spearman correlation, computed independently
    for action and consequence valence. Reads the raw per-model export CSV
    (outputs/output_<model>[_<prompt_version>]_entire_dataset.csv, produced
    by start_dataset_export) and merges it against the human-annotated
    dataset by scenario ID.

    model must be one of: "gemini", "lama", "deep-seek", "gpt-github", "claude".
    prompt_version must match an export that has actually been run for this
    model (see list_prompt_versions / start_dataset_export / list_recent_exports)
    — typically "current" or "few_shot". Raises if that export doesn't exist yet.
    """
    await ctx.info(f"compute_alignment_metrics called with model={model!r}, prompt_version={prompt_version!r}")
    report = evaluate_alignment(model, prompt_version=prompt_version)
    return ModelAlignmentReport(**report)


@mcp.tool()
async def get_job_status(job_id: str, ctx: Context = None) -> JobSnapshot:
    """
    Check the status/progress of a job by id, including jobs started from
    the web app (job status is persisted to disk, not process-local).
    """
    await ctx.info(f"get_job_status called with job_id={job_id!r}")
    job = get_job(job_id)
    if job is None:
        raise ValueError(f"No job found with id {job_id!r}.")
    return JobSnapshot(**job.snapshot())


@mcp.tool()
async def list_recent_jobs(ctx: Context = None) -> list[JobSnapshot]:
    """
    List all known evaluation jobs (most recent first), across both this
    MCP server and the web app, so an agent can discover past/ongoing runs
    without already having a job id in hand.
    """
    await ctx.info("list_recent_jobs called")
    return [JobSnapshot(**snap) for snap in list_jobs()]


# ---- Evals tools ----

@mcp.tool()
async def run_fixed_sample_eval(
    model: str,
    version: str = "current",
    size: int = 30,
    ctx: Context = None,
) -> dict:
    """
    Evaluate a fixed, reproducible sample of scenarios (same scenario IDs
    every time for a given size, via a fixed random seed) with one
    model/prompt-version combination, and report alignment metrics against
    the human-annotated labels.

    model must be one of: "gemini", "lama", "deep-seek", "gpt-github", "claude".
    version must be one of the values returned by list_prompt_versions
    (typically "v1", "v2", "current").

    Use this instead of start_multi_model_evaluation when you want a
    repeatable regression check rather than a fresh random sample.
    """
    await ctx.info(f"run_fixed_sample_eval called with model={model!r}, version={version!r}, size={size}")
    return evals.evaluate_fixed_sample(model=model, version=version, size=size)


@mcp.tool(name="compare_prompt_versions")
async def compare_prompt_versions_tool(
    model: str,
    size: int = 30,
    versions: list[str] | None = None,
    ctx: Context = None,
) -> list[dict]:
    """
    Run the same fixed evaluation sample through every prompt version (v1,
    v2, current by default) for one model, so a prompt change can be judged
    by measured alignment against human labels instead of by unrecorded
    impression. Returns one alignment-metrics report per version, in the
    order given.
    """
    await ctx.info(f"compare_prompt_versions called with model={model!r}, size={size}, versions={versions}")
    return evals.compare_prompt_versions(model=model, size=size, versions=versions)


@mcp.tool()
async def check_stability(
    model: str,
    sample_size: int = 5,
    repeats: int = 3,
    ctx: Context = None,
) -> list[dict]:
    """
    Re-run the same scenarios multiple times with one model to quantify
    run-to-run sampling noise (mean/stdev of action and consequence
    valence per scenario). High stdev on a scenario means some of that
    scenario's apparent divergence from humans may be noise, not a stable
    disagreement — useful context before over-interpreting a single run.
    """
    await ctx.info(f"check_stability called with model={model!r}, sample_size={sample_size}, repeats={repeats}")
    return evals.stability_check(model=model, sample_size=sample_size, repeats=repeats)


@mcp.tool()
async def list_prompt_versions(ctx: Context = None) -> list[str]:
    """List available prompt versions that can be passed to run_fixed_sample_eval / compare_prompt_versions_tool."""
    await ctx.info("list_prompt_versions called")
    return available_versions()


# ---- Resources ----

@mcp.resource("dataset://scenarios/count")
def dataset_scenario_count() -> str:
    """Number of scenarios available in the moral-alignment dataset."""
    return str(len(load_dataset()))


@mcp.resource("dataset://scenarios/{scenario_id}")
def dataset_scenario(scenario_id: str) -> str:
    """
    A single scenario from the dataset by ID, including its human gold-
    standard action/consequence valence and metadata (pattern/source/
    input_type), as JSON. Lets an agent inspect a specific scenario without
    needing to have run an evaluation first.
    """
    df = load_dataset()
    row = df[df["ID"] == int(scenario_id)]
    if row.empty:
        return f'{{"error": "No scenario with ID {scenario_id}"}}'
    return row.iloc[0].to_json()


@mcp.resource("jobs://recent")
def jobs_recent() -> str:
    """
    All known evaluation jobs (most recent first) as JSON, across both the
    web app and this MCP server, for browsing without a specific job id.
    """
    import json

    return json.dumps(list_jobs())


@mcp.resource("jobs://{job_id}/csv")
def job_csv_contents(job_id: str) -> str:
    """The raw results CSV for a completed job, by id."""
    csv_path = OUTPUT_DIR / f"job_{job_id}.csv"
    if not csv_path.exists():
        return f"No CSV found for job {job_id!r}."
    return csv_path.read_text()


# ---- Prompts ----

@mcp.prompt()
def moral_valence_scoring_rubric(scenario: str = "") -> str:
    """
    The exact rubric/prompt used to score action and consequence moral
    valence for a scenario, so another client can reproduce this project's
    methodology exactly rather than approximating it.
    """
    return build_prompt(scenario or "{scenario text goes here}")


@mcp.prompt()
def few_shot_scoring_rubric(scenario: str = "") -> str:
    """
    The exact few-shot rubric/prompt (10 worked examples of real
    human-labeled scenarios, valence scores only — no reasoning) used to
    score action and consequence moral valence, for the same reproducibility
    reason as moral_valence_scoring_rubric. Corresponds to prompt_version
    "few_shot" in start_dataset_export / compute_alignment_metrics.
    """
    from app.prompt_versions import get_prompt_builder

    builder = get_prompt_builder("few_shot")
    return builder(scenario or "{scenario text goes here}")


if __name__ == "__main__":
    print("FastMCP server started...", file=sys.stderr)
    mcp.run()
