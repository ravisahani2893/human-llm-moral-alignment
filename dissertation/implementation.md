# Implementation

The Methodology chapter explains *what* each part of the system does and *why* it was built that way. This chapter explains *how* it was actually built — the technology choices, the repository layout, and the implementation patterns used in each layer, in enough detail that another developer could reproduce the system from this description.

## 1. Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.13 | Single language across every layer (backend, MCP server, data/metrics, LLM clients) keeps the whole system easy to reason about end to end. |
| Web API | FastAPI + Uvicorn | Async-friendly, minimal boilerplate, automatic request validation via Pydantic models. |
| MCP server | `mcp` / FastMCP | The official Python MCP SDK's high-level server framework — turns a plain Python function into an MCP tool with one decorator, using the function's docstring as the tool's description. |
| Data / metrics | pandas, numpy, scipy | Standard scientific Python stack — pandas for CSV/dataframe handling, numpy/scipy for the statistical calculations in `app/metric.py`. |
| LLM SDKs | `google-genai` (Gemini), `anthropic` (Claude), `openai` (used for GPT-4.1 **and**, since Groq and DeepSeek both expose an OpenAI-compatible API, also used for Llama-via-Groq and DeepSeek by pointing the same client at a different `base_url`) | Reusing the OpenAI SDK for three of the five providers, instead of writing three more bespoke clients, kept the client layer smaller without losing any capability. |
| Frontend | Plain HTML/CSS/JavaScript, no framework | The UI is a thin, mostly declarative layer over the REST API — a framework would have added build tooling for very little benefit at this scale (7 pages, no client-side state beyond one page's form). |

## 2. Repository Layout

```
api.py                     REST API entrypoint (FastAPI app)
mcp_server/server.py       MCP server entrypoint (FastMCP app) — the capability boundary
agents/general_agent.py    Natural-language orchestrating agent

app/
  dataset.py                Loads the human-annotated dataset (data/processed/moralalign_dataset.csv)
  prompts.py, prompt_versions.py   Prompt templates and the versioning system (current/v1/v2/few_shot)
  evaluator.py               Core evaluation logic: MODEL_CLIENTS dispatch, evaluate_single,
                              evaluate_alignment, calculate_cross_model_agreement,
                              calculate_variant_bias / calculate_all_variant_bias
  metric.py                  Pure statistical functions: CCC, MAE, RMSE, Pearson, Spearman, Wilcoxon
  models.py                  Pydantic response models shared by the MCP server and the REST API
  jobs.py, export_jobs.py, bias_jobs.py, agent_runs.py   Background job runners (one per job type,
                              identical threading + disk-JSON-status pattern in each)
  mcp_client.py               The persistent MCP client used by api.py (see §5)
  llm_logger.py                Per-process-session plain-text request/response logging

clients/                    One thin wrapper per LLM provider (gemini, claude, openai, groq, deepseek)
tools/                      CLI scripts: dataset export, bias-variant export, prompt-version
                              comparison, and the standalone test scripts
web/                        Static HTML/JS/CSS pages, served directly by FastAPI's StaticFiles
data/
  processed/moralalign_dataset.csv     The 500-scenario human-annotated dataset
  variants/variants_GENDER.csv, variants_ETHNICITY.csv   Counterfactual bias-testing datasets
outputs/                    All generated CSVs and job-status JSON files (gitignored)
logs/                       Per-session LLM request logs (gitignored)
```

The separation between `app/evaluator.py` (what to compute) and `mcp_server/server.py` (how to expose it) is intentional: no evaluation logic lives inside the MCP tool functions themselves, only argument handling and calls out to `app/evaluator.py`. This is what let the REST API and the agent share the exact same computation without duplicating it (Methodology §4).

## 3. Data Layer

`app/dataset.py` is a thin wrapper around `pandas.read_csv` for the human-annotated dataset, with one helper (`sample_random`) for drawing a reproducible random sample using a fixed seed — used by the "evals" tools (§6 below) so a stability or prompt-comparison run always exercises the same scenarios.

The bias-testing variant datasets (`data/variants/variants_GENDER.csv`, `variants_ETHNICITY.csv`) follow a wide format: one row per original scenario, with one column per demographic variant (e.g. `Original`, `Male`, `Female`) holding that variant's rewritten scenario text, plus the *unchanged* original human `Action_Valence`/`Consequence_Valence` columns. `tools/bias_variant_eval.py` reshapes this into long format internally (one row per scenario-variant pair) before evaluation, auto-detecting which columns are variants versus metadata by excluding a fixed set of known non-variant column names — so adding a third variant dataset later requires no code change, only a new CSV following the same shape.

## 4. LLM Client Layer

Every client in `clients/` exposes the same shape of function: take a fully-built prompt string, return the provider's raw text response. `app/evaluator.py` wires all five into one dispatch table:

```python
MODEL_CLIENTS = {
    "gemini": lambda prompt: ask_gemini(prompt),
    "lama": lambda prompt: ask_groq(prompt, model="llama-3.3-70b-versatile"),
    "deep-seek": lambda prompt: ask_deepseek(prompt, model="deepseek-v4-pro"),
    "gpt-github": lambda prompt: ask_openai(prompt, model="gpt-4.1"),
    "claude": lambda prompt: ask_claude(prompt),
}
```

Everything above and below this table — prompt building, response parsing, metric calculation, job orchestration — is written against this one dictionary, never against an individual provider's SDK directly. Adding a sixth model means writing one more small client file and one more dictionary entry; nothing else in the codebase needs to know a new provider exists.

Each client also calls `app/llm_logger.py`'s `log_llm_call(...)` on every attempt — success, failure, and retry — writing a plain-text line to one log file per process run. This was added specifically to make provider-side incidents (rate limits, outages) diagnosable after the fact without needing to reproduce them live; it directly enabled diagnosing a real incident during the project where an LLM provider (GitHub Models, used originally for three of the five models) entered an undocumented "retirement brownout" — intermittent HTTP 410 errors — which the logs made visible as a pattern rather than isolated, confusing failures. That incident led to three of the five clients being migrated to call the providers' own APIs directly instead of routing through GitHub Models.

`response.get("action_valence")`-style parsing after each call strips Markdown code fences (some models wrap JSON output in ` ```json ` blocks) before calling `json.loads` — a small robustness step needed because not every provider's structured-output guarantees behave identically.

## 5. MCP Server and the Persistent Client

`mcp_server/server.py` is a single FastMCP application. Each tool is a Python `async def` decorated with `@mcp.tool()`; FastMCP inspects the function's type-annotated parameters to build its JSON input schema automatically, and uses the docstring verbatim as the tool's description — the same text a human reader sees in the source code is what the agent sees when deciding whether to call it. This double duty (documentation for humans, description for the agent) is why every tool's docstring in this project is written to be precise about arguments, return shape, and any preconditions (e.g. "requires an export to already exist for this model/prompt_version").

The MCP server is run as a **subprocess**, communicating over stdio using the JSON-RPC-based MCP protocol — the standard transport for a locally-run MCP server. Both consumers spawn or connect to this subprocess differently:

- The **agent** (`agents/general_agent.py`) spawns a fresh subprocess for each run, using `mcp.client.stdio.stdio_client`, and tears it down when the run finishes.
- The **REST API** (`api.py`) instead keeps **one persistent connection alive for the lifetime of the API process** (`app/mcp_client.py`). This exists because spawning a new subprocess and re-initializing an MCP session on every single HTTP request would add noticeable latency to every API call. The persistent client runs its own background thread with a dedicated asyncio event loop; `call_tool()` is a synchronous-looking function that internally does `asyncio.run_coroutine_threadsafe(...)` onto that loop and blocks for the result, letting FastAPI's own synchronous request handlers call it without needing to become async themselves.

A defect in this client layer was found and fixed during development: `call_tool()` originally never checked `result.isError` before attempting `json.loads()` on the tool's response text. Since a tool's *error* text is plain prose, not JSON, this produced a confusing `json.JSONDecodeError` instead of the tool's actual error message whenever a tool failed validation (e.g. requesting metrics for an export that doesn't exist yet). The fix checks `result.isError` first and raises a `RuntimeError` carrying the tool's real message, which then surfaces correctly all the way up through the REST layer to the browser.

## 6. Metric and Evaluation Logic

`app/metric.py` contains five pure functions with no dependency on the rest of the project — `calculate_ccc`, `calculate_mae`, `calculate_rmse`, `calculate_pearson`, `calculate_spearman`, `calculate_wilcoxon` — each taking two equal-length numeric arrays and returning a number (or, for Wilcoxon, a small dict of statistic/p-value/mean-delta/median-delta/n). Keeping these free of any file I/O or business logic is what made independent verification straightforward (Methodology §7): they could be re-derived from scratch and compared number-for-number without needing to reproduce any of the surrounding pipeline.

`app/evaluator.py` builds the three research-question-specific pipelines on top of these primitives:

- `evaluate_alignment(model, prompt_version)` — loads the human dataset and one model's export CSV, inner-joins on scenario `ID`, drops any row where the model's prediction is missing, and calls the five metric functions once for the Action axis and once for the Consequence axis.
- `calculate_cross_model_agreement(models, prompt_version)` — loads every selected model's export CSV, inner-joins all of them together on `ID` (so a scenario only counts if *every* selected model has a prediction for it), and computes Pearson/Spearman for every unique pair of models, assembling the results into four matrices (Action×Pearson, Action×Spearman, Consequence×Pearson, Consequence×Spearman).
- `calculate_variant_bias(model, dataset, variant_a, variant_b)` / `calculate_all_variant_bias(model, dataset)` — loads one model's bias-testing CSV, inner-joins two variants' rows on `ID`, and runs `calculate_wilcoxon` on the paired Action and Consequence scores; the `_all_` variant runs this once for every unique pair of variants found in the file.

All three explicitly avoid filling a missing value with zero — a scenario missing from either side of a join is simply excluded from that particular calculation, and the resulting count (`n_scenarios`) is always reported alongside the numbers so a reader can see exactly how much data a given metric is based on.

## 7. Background Jobs

Every long-running operation (full-dataset export, bias-variant evaluation, multi-model batch run, agent run) follows the same implementation pattern, duplicated across `app/export_jobs.py`, `app/bias_jobs.py`, `app/jobs.py`, and `app/agent_runs.py` rather than abstracted into one shared base class — a deliberate choice, since each job type's progress data (`completed`/`total` scenarios vs. `completed_per_model`, for instance) differs enough that a shared abstraction would have added indirection for little real code reduction.

The shared pattern itself:

```python
class SomeJob:
    def __init__(self, ...):
        self.id = uuid.uuid4().hex[:12]
        self.status = "running"
        ...

    def write_status(self):
        with open(f"outputs/somejob_{self.id}.status.json", "w") as f:
            json.dump(self.snapshot(), f)

def _run_job(job):
    def on_progress(completed, total):
        job.completed, job.total = completed, total
        job.write_status()
    try:
        do_the_actual_work(..., on_progress=on_progress)
        job.status = "completed"
    except Exception as exc:
        job.status, job.error = "error", str(exc)
    finally:
        job.write_status()

def start_job(...):
    job = SomeJob(...)
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return job
```

Writing the full status snapshot to a JSON file after every single unit of progress (not just at the end) is what makes status visible **across processes** — the API server, the MCP server subprocess, and a browser polling every few seconds are all just reading the same file, with no shared memory or message queue required. This is a deliberately low-tech solution, chosen because the project only ever runs on a single machine; a distributed deployment would need a real job queue, but that complexity wasn't needed here.

Export jobs (`tools/export_model_dataset.py`, `tools/bias_variant_eval.py`) are additionally resumable: on start, they read back any existing partial output CSV, and only re-process rows that are either missing entirely or present with a null prediction (a failed attempt from a previous run). This second condition was itself a bug fix made during the project — the original resume logic only checked whether a scenario's ID was present in the file at all, so a scenario that had failed and been written with a null value was incorrectly treated as "done" and silently never retried on subsequent runs.

## 8. REST API and Web UI

`api.py` defines one FastAPI route per operation, each doing three things: validate the request (model names against `MODEL_CLIENTS`, dataset names against files actually present in `data/variants/`, prompt versions against `list_prompt_versions`), call the matching MCP tool through the persistent client, and translate any resulting `RuntimeError` into an HTTP 400 with the tool's own error message attached. No route computes anything itself.

Each web page follows the same small pattern in its own `<page>.js` file: on load, fetch reference data (model list, prompt versions, available datasets) to populate selectors; on a button click, `POST` to start an operation; if the operation is a background job, poll its status endpoint every 2–3 seconds and update a progress bar until it leaves the `"running"` state; render the final result into a table or matrix using a small set of shared formatting helpers in `common.js` (`fmtValence`, `modelLabel`, `escapeHtml`, `buildModelSelect`). No page holds meaningful client-side state beyond the current job/poll-timer id — refreshing the page loses in-progress UI state but not the underlying job, since that lives in the disk-persisted status file, not in the browser.

## 9. Agent

`agents/general_agent.py` implements a standard LLM tool-calling loop: send the conversation so far (starting with just the user's instruction) plus the full tool list to Gemini; if the response contains function calls, execute each one against the MCP session, append the results back into the conversation, and loop; if the response is plain text with no function calls, that text is the agent's final answer and the loop stops. A hard cap (`MAX_TOOL_TURNS = 15`) prevents a runaway loop.

One piece of orchestration logic *is* hardcoded, deliberately, as a narrow exception to "the agent decides everything": tools that start a background job (`start_multi_model_evaluation`, `start_dataset_export`) are automatically polled to completion by the agent's own code (`_wait_for_job`) rather than requiring Gemini to spend its own reasoning turns repeatedly calling a status-check tool and deciding to wait. This was a practical efficiency choice — waiting is mechanical, not a decision — and does not compromise the "agent decides the sequence of operations" property, since the agent still decides *whether* to start such a job in the first place.

## 10. Testing

Three kinds of check exist in the codebase, matching the levels described in Methodology §7: `tools/test_metric.py` and `tools/test_evaluator.py` exercise the pure metric functions and the alignment pipeline directly; `tools/test_cross_model_agreement.py` implements the seven-case suite described in Methodology §7, using synthetic CSVs written under a dedicated test prompt-version suffix so a test run can never touch or overwrite real project data. All three are plain Python scripts runnable directly (`python -m tools.test_cross_model_agreement`), not wired into a CI system, since the project runs on a single development machine rather than a deployed service.
