import json
import threading
import time
import uuid
from pathlib import Path

from tools.export_model_dataset import export_model_dataset, export_output_path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Same cross-process visibility pattern as app/jobs.py: in-memory dict for
# the process that started a job, disk-persisted status JSON for every
# other process (web app, MCP server) to read.
_export_jobs: dict[str, "ExportJob"] = {}


def _status_path(job_id: str) -> Path:
    return OUTPUT_DIR / f"export_{job_id}.status.json"


class ExportJob:
    def __init__(self, model: str, prompt_version: str = "current"):
        self.id = uuid.uuid4().hex[:12]
        self.model = model
        self.prompt_version = prompt_version
        self.status = "running"  # running | completed | error
        self.error = None
        self.completed = 0
        self.total = 0
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.csv_path = export_output_path(model, prompt_version)

    def snapshot(self):
        return {
            "id": self.id,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "status": self.status,
            "error": self.error,
            "completed": self.completed,
            "total": self.total,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "csv_path": str(self.csv_path),
        }

    def write_status(self):
        self.updated_at = time.time()
        OUTPUT_DIR.mkdir(exist_ok=True)
        with open(_status_path(self.id), "w") as f:
            json.dump(self.snapshot(), f)


class ExportJobView:
    """Read-only view of an export job reconstructed from its status file."""

    def __init__(self, data: dict):
        self._data = data
        self.id = data["id"]
        self.status = data["status"]

    def snapshot(self):
        return dict(self._data)


def _run_export_job(job: ExportJob):
    def on_progress(completed: int, total: int):
        job.completed = completed
        job.total = total
        job.write_status()

    try:
        export_model_dataset(job.model, resume=True, on_progress=on_progress, prompt_version=job.prompt_version)
        job.status = "completed"
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
    finally:
        job.write_status()


def start_export_job(model: str, prompt_version: str = "current") -> ExportJob:
    job = ExportJob(model=model, prompt_version=prompt_version)
    _export_jobs[job.id] = job
    job.write_status()
    thread = threading.Thread(target=_run_export_job, args=(job,), daemon=True)
    thread.start()
    return job


def get_export_job(job_id: str) -> ExportJob | ExportJobView | None:
    if job_id in _export_jobs:
        return _export_jobs[job_id]

    status_path = _status_path(job_id)
    if status_path.exists():
        try:
            with open(status_path) as f:
                return ExportJobView(json.load(f))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def list_export_jobs() -> list[dict]:
    snapshots: dict[str, dict] = {}

    for status_path in OUTPUT_DIR.glob("export_*.status.json"):
        try:
            with open(status_path) as f:
                data = json.load(f)
            snapshots[data["id"]] = data
        except (json.JSONDecodeError, OSError):
            continue

    for job_id, job in _export_jobs.items():
        snapshots[job_id] = job.snapshot()

    return sorted(snapshots.values(), key=lambda d: d.get("created_at", 0), reverse=True)
