# MCP Server Reference — `mcp_server/server.py`

This server exposes the project's evaluation pipeline over the Model Context Protocol (MCP), so any MCP-speaking client (Claude Desktop, or the project's own `agents/divergence_analyst.py`) can call it. It has three kinds of surface: **Tools** (do something), **Resources** (read something), **Prompts** (retrieve a reusable instruction template).

## Tools

### Core evaluation

| Tool | What it does | Key args | Notes |
|---|---|---|---|
| `evaluate_moral_scenario` | Scores one scenario with one model — action valence, action reasoning, consequence valence, consequence reasoning. | `scenario`, `model` | Fast, single call. Used for the "single scenario" mode everywhere in the project. |
| `evaluate_random_scenarios` | Scores a random sample from the dataset with **one** model, blocking until done. | `sample_size`, `model` | Legacy/simple path — single-model only. Superseded by `start_multi_model_evaluation` for anything multi-model. |
| `evaluate_dataset` | Scores the **entire** dataset with one model, blocking until done. | `model` | Same caveat — single-model, blocking, slow for ~500 rows. |
| `start_multi_model_evaluation` | Starts a background job scoring several models against a sample or the full dataset **in parallel**, and returns immediately (does not wait). | `models`, `sample_size` (omit for full dataset) | The main entry point for batch runs. Non-blocking by design — a full run can take many minutes, and a blocking call would tie up the caller for that whole time. |
| `get_job_status` | Polls a job's progress by ID. | `job_id` | Works even for jobs started by the web app, not just this MCP server — job status is persisted to disk, not held only in memory. |
| `list_recent_jobs` | Lists all known jobs (most recent first), across the web app and MCP server. | — | Lets an agent discover past runs without already knowing a job ID. |
| `compute_alignment_metrics` | Computes Pearson/Spearman/CCC correlation, MAE, RMSE, sign agreement, mean bias, cross-model agreement, and a breakdown by scenario metadata (pattern/source/input_type) — from a completed job's results CSV. | `csv_path`, `models` | The statistics engine. Deterministic — same input always gives the same numbers. **Human-LLM Alignment**: compares one model's predictions against human annotations. |
| `compute_cross_model_agreement` | Computes pairwise Pearson and Spearman correlation between models' own predictions, for Action and Consequence Valence separately, across per-model export CSVs. Returns 4 correlation matrices (Action×Pearson, Action×Spearman, Consequence×Pearson, Consequence×Spearman) plus a flat pairwise list. | `models` (optional, defaults to all 5), `prompt_version` (default `"current"`) | **Cross-Model Agreement**: model-vs-model only — never reads or reports human annotations (see [EVALUATION_METRICS.md §3a](EVALUATION_METRICS.md#3a-cross-model-agreement-current-implementation)). Requires per-model export CSVs to already exist (`start_dataset_export`); merges by scenario ID, not row order, and only includes scenarios every selected model has a prediction for. |

### Evaluation-quality tools ("evals")

| Tool | What it does | Key args | Notes |
|---|---|---|---|
| `run_fixed_sample_eval` | Scores a **fixed, reproducible** sample (same scenario IDs every time, via a fixed random seed) with one model/prompt-version combo. | `model`, `version`, `size` | Use this instead of a fresh random sample when you want a repeatable regression check — the point of a fixed sample is that results are comparable run over run. |
| `compare_prompt_versions` | Runs the same fixed evaluation sample through every prompt version (`v1`, `v2`, `current`) for one model and reports metrics for each. | `model`, `size`, `versions` | Turns "we think the newer prompt is better" into a measured claim. |
| `check_stability` | Re-runs the same scenarios multiple times with one model to measure run-to-run variance (mean/stdev of valence per scenario). | `model`, `sample_size`, `repeats` | Tells you how much of a model's "divergence from humans" is genuine disagreement versus sampling noise. |
| `list_prompt_versions` | Lists which prompt versions exist (`v1`, `v2`, `current`). | — | Small utility, used by the other two eval tools. |

## Resources (read-only, no side effects)

| URI | What it returns |
|---|---|
| `dataset://scenarios/count` | The number of scenarios in the dataset (currently 500). |
| `dataset://scenarios/{scenario_id}` | One scenario's full record by ID — text, human gold-standard action/consequence valence, and metadata (pattern/source/input_type). |
| `jobs://recent` | All known jobs as JSON, same data as `list_recent_jobs` but as a browsable resource instead of a tool call. |
| `jobs://{job_id}/csv` | The full raw results CSV for a completed job — every scenario, every model's scores **and reasoning text**. This is how the divergence-analyst agent reads actual model reasoning to explain *why* a divergence happened. |

## Prompts

| Name | What it does |
|---|---|
| `moral_valence_scoring_rubric` | Returns the exact prompt template (`app/prompts.py`) used to score every scenario, so another researcher/client can reproduce the methodology precisely instead of guessing at it from your code. |

## How these fit together (a typical chain)

1. `start_multi_model_evaluation(models=["gemini","lama"], sample_size=15)` → job starts, returns a `job_id` immediately.
2. `get_job_status(job_id)` polled until `status == "completed"` (the agent does this automatically without spending its own reasoning turns on it).
3. `compute_alignment_metrics(csv_path, models)` → correlation/MAE/CCC numbers.
4. `jobs://{job_id}/csv` (resource, not a tool) → raw per-scenario reasoning text, read to explain *why* the worst-divergence cases diverged.
5. Final report written, quoting the actual reasoning from step 4, grounded in the numbers from step 3.

No single tool does step 5 — that synthesis only happens because something (the agent) decided to call these four things in this order for this specific question.
