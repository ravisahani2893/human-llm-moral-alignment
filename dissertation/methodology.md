# Methodology

## 1. Research Design

This project is a **computational, comparative evaluation study**. It does not train or fine-tune any model; it measures how five existing frontier LLMs (Gemini 2.5 Flash, Llama 3.3 70B, DeepSeek V4 Pro, GPT-4.1, and Claude Sonnet 5) score the same set of moral scenarios for two dimensions — **Action Valence** (was the action itself right or wrong) and **Consequence Valence** (was the outcome good or bad) — and compares those scores against a human-annotated ground truth.

The design has three layers, each answering a different research question:

1. **Human-LLM Alignment** — how closely does each model's judgment track human moral judgment? This is the core question the dissertation title asks.
2. **Cross-Model Agreement** — independent of humans entirely, how closely do the five models track *each other*? This is a secondary, exploratory question that turned out to produce one of the more interesting findings in the project (see Results/Evaluation): models agree with each other considerably more than any of them agree with the human annotator.
3. **Demographic Bias / Robustness** — for a fixed scenario, does a model's own score shift when only a surface demographic marker (a name signalling gender or ethnicity) changes, with the moral content of the scenario held constant? This is a counterfactual perturbation study, methodologically distinct from (1) and (2): it never involves human annotations or other models at all, only one model compared against itself across variants.

Each of these three questions is implemented as its own metric pipeline, deliberately kept separate in code so that a change to one can never silently affect another (see §6, ID-based merging).

### 1.1 Dataset

The 500 scenarios and their human Action/Consequence Valence labels were personally annotated by the author using the R Shiny annotation tool from the source paper's methodology. This is **not** the source paper's own published dataset — it is an independent, single-annotator replication of their annotation *process* over the same scenario pool. This distinction matters for the limitations discussion (§7 in Evaluation) and should be stated plainly wherever the dataset is described, to pre-empt an examiner's obvious question about provenance.

Two smaller counterfactual datasets were constructed for the bias-testing arm of the study: a **gender** variant set (35 scenarios rewritten with Male/Female/Original name variants) and an **ethnicity** variant set (38 scenarios rewritten with Indian/European/American/Original name variants). Each variant preserves the original human Action/Consequence Valence labels unchanged, because the moral content of the scenario has not changed — only a name has. These datasets are intentionally small; the sample-size implications are discussed as a limitation in §7.

---

## 2. System Architecture

The system is built around a single idea: **all evaluation logic lives behind a Model Context Protocol (MCP) server, and every other component — the web UI, the REST API, and the natural-language agent — is a client of that server, not a reimplementation of its logic.** This is the architectural decision most worth defending to an examiner, and §4 below does so explicitly.

```
Human-annotated dataset (single-annotator, 500 scenarios)
Bias variant datasets (gender/ethnicity counterfactuals)
         │
         ▼
Prompt Builder (zero-shot "current" / few-shot / v1 / v2)
         │
         ▼
┌─────────────────────────────────────────────────┐
│              MCP Server (FastMCP)                 │
│  Tools:   evaluate_moral_scenario,                 │
│           start_dataset_export,                    │
│           compute_alignment_metrics,                │
│           compute_cross_model_agreement,             │
│           start_bias_variant_eval,                    │
│           compute_variant_bias, evals tools             │
│  Resources: dataset / job / export data (read-only)      │
│  Prompts:   scoring-rubric templates (reproducibility)     │
└──────────────────────┬──────────────────────────────┘
                        │ (each tool calls one of 5 model clients)
        Gemini · Llama (via Groq) · DeepSeek · GPT-4.1 · Claude
                        │
                        ▼
      Background job layer (threading.Thread + disk-persisted
      JSON status files — resumable, visible across processes)
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
  FastAPI REST layer          General-purpose Agent
  (thin proxy over the        (Gemini-orchestrated;
   persistent MCP client)      discovers MCP tools at
          │                    runtime via list_tools();
          ▼                    no hardcoded workflow)
  Web UI — Single Scenario / Full Dataset / Per-Model Export /
  Human-LLM Alignment / Cross-Model Agreement / Bias Testing / Agent
```

### 2.1 Why this shape

The system was not designed top-down from this diagram; it grew from a working single-model evaluator into this shape as three separate requirements emerged, and it is worth narrating that evolution briefly in the dissertation because it justifies the design better than presenting it as pre-planned:

1. First, a way to score one scenario with one model (`evaluate_moral_scenario`) and compute alignment against human labels — this alone could have been plain REST.
2. Then, a requirement to run the same evaluation across the *entire* dataset for *all five* models without blocking a web request for the many minutes that takes — this produced the background job layer.
3. Then, a requirement for a natural-language agent that could *decide for itself* which of these operations to run, in what order, from a free-form instruction — this is the requirement that actually necessitates MCP over plain REST, and is explained fully in §4.

---

## 3. Component Walkthrough

### 3.1 Web UI
Seven static HTML pages (no frontend framework — vanilla JS/HTML/CSS, served directly by FastAPI's `StaticFiles`), each corresponding to one operation: running a single scenario, running the full dataset, exporting one model's raw predictions, computing Human-LLM Alignment, computing Cross-Model Agreement, running the Bias Testing pipeline, and driving the general-purpose agent. Long-running operations (exports, bias evaluations, batch runs, agent runs) show a live progress bar polling a status endpoint every 2–3 seconds, rather than blocking the browser.

### 3.2 FastAPI REST layer (`api.py`)
A conventional REST API — but its role is narrower than it looks. For anything that touches evaluation logic, its handlers do not call Python functions directly; they call `mcp_client.call_tool(...)` over a **persistent** MCP connection held for the lifetime of the API process. This is explained fully in §4.

### 3.3 MCP Server (`mcp_server/server.py`)
Built with FastMCP, exposing three kinds of surface:
- **Tools** — the callable operations (evaluate, export, compute metrics). Each tool's Python docstring *is* its description in the MCP protocol, which becomes directly relevant when the agent uses this text to decide which tool to call (§3.6).
- **Resources** — read-only, URI-addressed data (e.g. `dataset://scenarios/{id}`, `jobs://{job_id}/csv`), for retrieving results without a "compute" side effect.
- **Prompts** — the exact scoring-rubric templates used to query each LLM, exposed so the methodology is reproducible by another party without reading the source code.

### 3.4 Model integration layer
Five thin client wrappers (`clients/*.py`), one per provider, each taking a built prompt string and returning the provider's raw text response, which is then parsed as JSON into `{action_valence, action_reasoning, consequence_valence, consequence_reasoning}`. All five are called through a uniform `MODEL_CLIENTS` dispatch dictionary (`app/evaluator.py`), so every downstream tool is written against one interface regardless of provider.

### 3.5 Metric layer (`app/metric.py`, `app/evaluator.py`)
Five metric primitives — CCC, MAE, RMSE, Pearson, Spearman — plus a Wilcoxon signed-rank test added specifically for bias testing. Three higher-level functions compose these primitives for the three research questions in §1: `evaluate_alignment` (human vs. one model), `calculate_cross_model_agreement` (model vs. model, pairwise), and `calculate_variant_bias`/`calculate_all_variant_bias` (one model vs. itself across demographic variants). None of the three duplicate the underlying math — they only differ in which two columns of scores they hand to the same primitive functions. §6 explains why this matters more than it sounds.

### 3.6 Agent (`agents/general_agent.py`)
A Gemini-orchestrated agent that receives a free-form natural-language instruction and the **live** tool list from the MCP server (via `list_tools()`), and decides for itself which tools to call, in what order, stopping when it has enough information to answer. Nothing about the sequence of operations is hardcoded — this is what distinguishes it from a scripted pipeline and is the central justification for using MCP at all (§4).

### 3.7 Background job infrastructure
Long-running operations use `threading.Thread` plus a JSON status file written to disk after every unit of progress. This solves two problems at once: (a) the API request that starts a job returns immediately instead of blocking, and (b) any other process (the API server, the MCP server subprocess, a later poll from the browser) can read the same status by reading the file, not by holding shared memory. A client-side heuristic (`effectiveStatus()` in `common.js`) flags a job as "stalled" if its status file hasn't been touched in over 90 seconds, since a genuinely running job updates continuously — this caught a real incident during development where a server restart orphaned several background threads without marking their jobs as failed.

Exports are additionally **resumable**: a partial CSV is read back in on restart, and only the scenarios not already present (with valid, non-null predictions — a bug where failed rows were incorrectly treated as "done" was found and fixed during the project) are re-processed.

---

## 4. Design Justification: MCP instead of plain REST

An examiner is very likely to ask: *"You already have a REST API — why introduce MCP at all? Isn't this unnecessary complexity?"* The honest answer is not "MCP is categorically better than REST." It is: **plain REST does not solve the problem that the agent component actually has**, and the added complexity is a cost paid specifically for that requirement, not a general architectural preference.

**The actual problem**: the general-purpose agent (§3.6) must take a free-form instruction like *"Export the entire dataset for claude, then show me its alignment metrics"* and decide, itself, which backend operations to call and in what order. It cannot have that sequence hardcoded, because the whole point is that a researcher can ask it something the author never anticipated. This requires the agent to discover, at runtime, *what operations exist and what each one needs as input*.

Plain REST does not give you this for free. An OpenAPI/Swagger spec can describe endpoints, but nothing in a typical REST stack hands an LLM orchestrator a structured, machine-readable "here are your available functions" list at runtime — that glue would have to be hand-written and kept in sync with the API by hand. MCP's `list_tools()` call does exactly this natively: it returns every tool's name, natural-language description (taken directly from the Python docstring), and JSON input schema, in the exact shape an LLM function-calling API expects. The agent in this project calls `list_tools()` once at startup and passes the result straight into Gemini's function-calling configuration — there is no manually maintained mapping between "things the agent can do" and "things the backend exposes."

**The REST API was not replaced — it was demoted to one client among several.** `api.py`'s handlers call the same MCP tools over `app/mcp_client.py`'s persistent connection that the agent uses, rather than reimplementing evaluation logic. This was a deliberate choice: it guarantees the web UI and the agent can never see different numbers for the same underlying computation, because there is only one implementation for either to call.

**The honest cost, worth stating explicitly rather than waiting for it to be raised**: MCP's stdio/JSON-RPC transport requires managing a subprocess and its lifecycle (the MCP server runs as a child process of the API server), which a browser cannot speak directly — hence the REST proxy layer still has to exist. This added real engineering overhead during development: a persistent connection with its own background thread and event loop (`app/mcp_client.py`), and a genuine bug where tool errors were not correctly distinguished from successful JSON responses (`result.isError` was not checked), which produced confusing parse errors until it was found and fixed. If the project had never needed the agent — if it were only ever going to be a web UI over a fixed set of buttons — plain REST would have been simpler, with none of this overhead. The complexity is justified by, and scoped to, the agent-orchestration requirement.

---

## 5. Prompt Design Methodology

Every model is asked the same question, in the same format, so that any difference in scores can be attributed to the model itself and not to one model getting an easier or clearer question than another.

Two prompt styles were used:

- **Zero-shot ("current")** — the model is given the scenario and asked to rate Action Valence and Consequence Valence, with no worked examples. This is the simplest and fastest condition, and is the default used everywhere unless stated otherwise.
- **Few-shot** — the model is first shown 10 worked examples: real scenarios from the dataset, together with their *real* human-given valence scores (no invented reasoning was added to these examples, to avoid teaching the model a style of explanation that isn't grounded in the actual annotation). The idea is simple: showing a few labelled examples first often helps a model calibrate its own scale before it has to label anything itself.

Both prompt styles were kept as separate, versioned outputs (`current` vs `few_shot`, plus two earlier iterations `v1`/`v2` kept for the historical record), so that a "does few-shot actually help?" comparison is a simple matter of comparing two already-computed sets of metrics, rather than something that has to be re-run from scratch.

Two smaller design choices matter enough to state explicitly:

- The prompt always asks for a **reasoning string** alongside the numeric score, not just the number. This wasn't required for the metrics themselves, but it means every disagreement between a model and the human annotator can be read and understood afterwards, instead of being just an unexplained number.
- Temperature was set to 0 for every model that supports it, to keep answers as repeatable as possible run to run. One exception is worth naming honestly: Claude's API (the version used here) does not expose a temperature parameter at all, so this one model was evaluated at whatever its default sampling behaviour is. This is a real, small asymmetry between models and is flagged again in the Evaluation chapter's limitations.

---

## 6. Metric Methodology

### 6.1 Why CCC is the primary metric

Plain correlation (Pearson) only checks whether two sets of numbers move together — it doesn't check whether they're actually close to each other in value. A model could always score everything 0.3 higher than the human, or squeeze all its answers into a narrow band near zero while the human uses the full range from -1 to 1, and Pearson would still report a high number as long as the model's *ranking* of scenarios matched the human's ranking.

For this study, that's not good enough. The question being asked is "does the model's judgment actually match the human's judgment," not just "does the model rank scenarios in a similar order." **Lin's Concordance Correlation Coefficient (CCC)** answers the stricter, more useful question: it starts from Pearson's correlation and then multiplies it by a penalty term that shrinks whenever the two sets of scores differ in their average value or in how spread out they are. A model that is consistently "in the right direction but too mild" gets penalised by CCC even though Pearson would have looked fine.

Because of this, CCC is used as the **primary** metric throughout the project — and because it is also the metric the source paper (whose annotation method this project's dataset follows) reports, using it keeps this project's numbers comparable to that prior work, which is itself a reason worth stating.

### 6.2 Why CCC alone is not enough — the supporting metrics

CCC has one real weakness: because its formula is built from variances, a single scenario where a model's answer is wildly different from the human's can pull the whole number a long way, especially when there aren't many scenarios to average over. This was demonstrated directly during the project: with a toy example of only 4 scenarios, one unusually large disagreement was enough to flip the sign of the entire CCC score from positive to negative, even though the other three scenarios agreed reasonably well. At the full dataset size (500 scenarios) this effect is much smaller, but it is still a reason not to trust one single number in isolation.

For that reason, four supporting metrics are always reported alongside CCC:

- **MAE / RMSE** — the plain average size of the error, in the same units as the valence scores. Useful because, unlike CCC, these aren't affected by a handful of outliers dominating the picture.
- **Pearson** — kept alongside CCC specifically so the *gap* between the two is visible. If Pearson is much higher than CCC, that tells you the disagreement is mostly about scale/bias (the model tracks the human's pattern but with an offset). If the two are close together, as they generally were in this project's results, that tells you the disagreement is more fundamental — the model and the human are ranking scenarios differently, not just scoring them on different scales.
- **Spearman** — a rank-only correlation, unaffected by scale entirely. Useful as a sanity check on Pearson.

### 6.3 Why the bias-testing study uses a different metric (Wilcoxon)

The Human-LLM Alignment question and the Bias Testing question are different shapes of problem, so they use different statistics. Alignment compares 500 independent scenarios against a human score — a correlation-style question. Bias testing compares the *same* scenario to itself, scored twice, once per demographic variant — a **paired** question: "for this one scenario, did the score go up or down when the name changed?"

The **Wilcoxon signed-rank test** is the right tool for a paired question like this, for two reasons. First, it works directly on the paired differences (Female score minus Male score, for the same scenario), rather than treating the two sets of scores as independent. Second, it does not assume those differences are shaped like a normal distribution — which matters here because the bias-testing datasets are small (35–38 scenario pairs), and assuming normality on that few data points would be a shaky assumption. A more familiar test like the paired t-test does make that normality assumption; Wilcoxon only assumes the differences are roughly symmetric, which is a much safer thing to assume with this little data.

### 6.4 Why matching is always done by scenario ID, never by row order

Every metric calculation in this project — alignment, cross-model agreement, and bias testing — joins two sets of scores together using the scenario's unique ID, never by assuming row 1 in one file lines up with row 1 in another. This sounds like a minor implementation detail, but it is a genuine correctness safeguard: several of this project's data files are produced incrementally and can resume after being interrupted, so two files covering "the same" scenarios can easily end up with the same rows in a different order, or with one file simply missing a few rows the other has. Joining by ID means a partially-finished or resumed file can never silently pair the wrong human score with the wrong model score — a mismatch here would be invisible in the metric number but would quietly poison the result. Any scenario missing from either side is dropped from that comparison rather than guessed at or filled in with zero.

---

## 7. Validation and Reproducibility Methodology

Because every result in this dissertation ultimately rests on the metric functions being correct, they were checked in more than one way rather than trusted on the strength of the code alone:

- **Independent re-derivation.** CCC, MAE, RMSE, Pearson, and Spearman were each recomputed completely from scratch, using only `numpy`/`scipy` directly with no shared code with the project's own implementation, for every model. The two sets of numbers matched to six decimal places in every case.
- **A dedicated automated test suite** was written for the Cross-Model Agreement calculation specifically, covering seven cases chosen to catch the most likely ways this kind of calculation can quietly go wrong: perfect agreement (should give a score of exactly 1.0), perfect disagreement (should give exactly -1.0), a small hand-worked example with a manually checked expected answer, robustness to the rows being shuffled into a different order, correct handling when a scenario is missing from one model's file, a clear error (not a silent duplicate) when a scenario ID appears twice, and — importantly, given how central this is to the project's separation of concerns — a direct proof that changing the human annotation values has **no effect whatsoever** on the Cross-Model Agreement result, by making the loading of human data raise an error and confirming the calculation still runs successfully. All seven passed.
- **Resumability was tested against real interruptions**, not just in theory: several of the model export runs described in this dissertation were, in practice, interrupted partway through (API rate limits, a token quota being exhausted, a transient connection error) and resumed later. The final results reported in the next chapter reflect these resumed, completed runs, and the resumption logic itself was verified to correctly distinguish scenarios that had genuinely succeeded from ones that had failed and needed to be retried — a real bug of exactly this kind was found and fixed during the project (a scenario that failed was originally being marked as "done" instead of "needs retry").
