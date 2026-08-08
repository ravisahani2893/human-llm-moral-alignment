import re
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import mcp_client
from app.agent_runs import get_agent_run, list_agent_runs, start_agent_run
from app.dataset import load_dataset
from app.evaluator import MODEL_CLIENTS, MODEL_LABELS, evaluate_models_for_scenario
from app.export_jobs import get_export_job, list_export_jobs
from app.jobs import get_job, list_jobs, start_job

app = FastAPI(title="Human LLM Moral Alignment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _start_mcp_client():
    mcp_client.start()


class EvaluateRequest(BaseModel):
    scenario: str
    models: list[str]


class BatchJobRequest(BaseModel):
    models: list[str]
    sample_size: int | None = None


class ExportJobRequest(BaseModel):
    model: str


class AgentRunRequest(BaseModel):
    mode: str  # "scenario" | "random" | "all"
    models: list[str]
    scenario: str | None = None
    sample_size: int | None = None


def _validate_models(models: list[str]):
    unknown = [m for m in models if m not in MODEL_CLIENTS]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model(s): {unknown}. Available: {list(MODEL_CLIENTS.keys())}",
        )
    if not models:
        raise HTTPException(status_code=400, detail="At least one model must be selected.")


@app.get("/api/models")
def list_models():
    return [{"id": key, "label": MODEL_LABELS.get(key, key)} for key in MODEL_CLIENTS]


@app.get("/api/dataset/count")
def dataset_count():
    return {"count": len(load_dataset())}


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest):
    scenario = req.scenario.strip()
    if not scenario:
        raise HTTPException(status_code=400, detail="Scenario text is required.")
    _validate_models(req.models)

    results = evaluate_models_for_scenario(scenario, req.models)
    return {"scenario": scenario, "results": results}


@app.get("/api/jobs")
def jobs_list():
    return {"jobs": list_jobs()}


@app.post("/api/jobs")
def create_job(req: BatchJobRequest):
    _validate_models(req.models)
    if req.sample_size is not None and req.sample_size <= 0:
        raise HTTPException(status_code=400, detail="sample_size must be positive.")

    job = start_job(models=req.models, sample_size=req.sample_size)
    return job.snapshot()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job.snapshot()


@app.get("/api/jobs/{job_id}/results")
def job_results(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"rows": job.rows_snapshot()}


@app.get("/api/exports")
def exports_list():
    return {"exports": list_export_jobs()}


@app.post("/api/exports")
def create_export(req: ExportJobRequest):
    """
    Trigger a dataset export the same way the CLI's trigger_export_via_mcp.py
    does: as a real MCP client call to mcp_server.server's start_dataset_export
    tool, over a persistent connection held for the API server's lifetime
    (app.mcp_client). The background export thread runs inside that MCP
    server subprocess; status/CSV endpoints below read the same shared
    on-disk job state directly, so they don't need to go through MCP too.
    """
    _validate_models([req.model])
    return mcp_client.call_tool("start_dataset_export", {"model": req.model})


@app.get("/api/exports/{job_id}")
def export_status(job_id: str):
    job = get_export_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Export job not found.")
    return job.snapshot()


@app.get("/api/exports/{job_id}/csv")
def export_csv(job_id: str):
    job = get_export_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Export job not found.")
    snap = job.snapshot()
    csv_path = Path(snap["csv_path"])
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV not yet available.")
    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=csv_path.name,
    )


@app.get("/api/jobs/{job_id}/csv")
def job_csv(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not job.csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV not yet available.")
    return FileResponse(
        job.csv_path,
        media_type="text/csv",
        filename=f"moral_alignment_{job_id}.csv",
    )


@app.post("/api/agent/runs")
def agent_run_create(req: AgentRunRequest):
    if req.mode not in ("scenario", "random", "all"):
        raise HTTPException(status_code=400, detail="mode must be one of: scenario, random, all")
    _validate_models(req.models)
    if req.mode == "scenario" and not (req.scenario and req.scenario.strip()):
        raise HTTPException(status_code=400, detail="scenario text is required for mode='scenario'.")
    if req.mode == "random" and not (req.sample_size and req.sample_size > 0):
        raise HTTPException(status_code=400, detail="a positive sample_size is required for mode='random'.")

    run = start_agent_run(
        mode=req.mode,
        models=req.models,
        scenario=req.scenario,
        sample_size=req.sample_size,
    )
    return run.snapshot()


@app.get("/api/agent/runs")
def agent_runs_list():
    return {"runs": list_agent_runs()}


@app.get("/api/agent/runs/{run_id}")
def agent_run_status(run_id: str):
    run = get_agent_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found.")
    return run


def _underlying_job_id(run: dict) -> str | None:
    if not run.get("csv_path"):
        return None
    match = re.search(r"job_([0-9a-f]+)\.csv$", run["csv_path"])
    return match.group(1) if match else None


@app.get("/api/agent/runs/{run_id}/results")
def agent_run_results(run_id: str):
    run = get_agent_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found.")
    job_id = _underlying_job_id(run)
    if job_id is None:
        return {"rows": []}
    job = get_job(job_id)
    if job is None:
        return {"rows": []}
    return {"rows": job.rows_snapshot()}


@app.get("/api/agent/runs/{run_id}/csv")
def agent_run_csv(run_id: str):
    run = get_agent_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found.")
    job_id = _underlying_job_id(run)
    job = get_job(job_id) if job_id else None
    if job is None or not job.csv_path.exists():
        raise HTTPException(status_code=404, detail="No comparison CSV available for this run.")
    return FileResponse(
        job.csv_path,
        media_type="text/csv",
        filename=f"agent_run_{run_id}.csv",
    )


web_dir = Path(__file__).parent / "web"
app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    print("Starting API server on http://127.0.0.1:8000", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=8000)
