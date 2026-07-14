import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.dataset import load_dataset
from app.evaluator import MODEL_CLIENTS, MODEL_LABELS, evaluate_models_for_scenario
from app.jobs import get_job, start_job

app = FastAPI(title="Human LLM Moral Alignment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class EvaluateRequest(BaseModel):
    scenario: str
    models: list[str]


class BatchJobRequest(BaseModel):
    models: list[str]
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


web_dir = Path(__file__).parent / "web"
app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    print("Starting API server on http://127.0.0.1:8000", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=8000)
