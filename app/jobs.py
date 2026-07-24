import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.dataset import load_dataset, sample_random
from app.evaluator import evaluate_single, MODEL_LABELS
from app.interpret import valence_label
from app.prompt_versions import compute_prompt_hash

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# In-process cache of jobs this process started. Cross-process visibility
# (e.g. the web app and the MCP server, which run as separate OS processes)
# comes from the status JSON file each job writes to disk on every update,
# not from this dict — see get_job()/list_jobs().
_jobs: dict[str, "Job"] = {}


def _status_path(job_id: str) -> Path:
    return OUTPUT_DIR / f"job_{job_id}.status.json"


class Job:
    def __init__(self, models: list[str], sample_size: int | None, prompt_version: str = "current"):
        self.id = uuid.uuid4().hex[:12]
        self.models = models
        self.sample_size = sample_size
        self.prompt_version = prompt_version
        self.prompt_template_hash = compute_prompt_hash(prompt_version)
        self.status = "running"  # running | completed | error
        self.error = None
        self.created_at = time.time()
        self.lock = threading.Lock()
        self.rows: dict[int, dict] = {}  # ID -> combined row
        self.total = 0
        self.completed_per_model = {m: 0 for m in models}
        self.errors_per_model = {m: 0 for m in models}
        self.csv_path = OUTPUT_DIR / f"job_{self.id}.csv"

    def snapshot(self):
        with self.lock:
            return {
                "id": self.id,
                "status": self.status,
                "error": self.error,
                "models": self.models,
                "prompt_version": self.prompt_version,
                "prompt_template_hash": self.prompt_template_hash,
                "total": self.total,
                "completed_per_model": dict(self.completed_per_model),
                "errors_per_model": dict(self.errors_per_model),
                "created_at": self.created_at,
                "csv_path": str(self.csv_path),
            }

    def write_status(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
        with open(_status_path(self.id), "w") as f:
            json.dump(self.snapshot(), f)

    def rows_snapshot(self):
        with self.lock:
            return [self.rows[k] for k in sorted(self.rows.keys())]

    def _column_order(self):
        cols = ["ID", "Scenario", "Prompt_Version", "Prompt_Template_Hash", "Human_Action", "Human_Consequence"]
        for m in self.models:
            label = MODEL_LABELS.get(m, m)
            cols += [
                f"{label}_Action",
                f"{label}_Action_Band",
                f"{label}_Action_Reasoning",
                f"{label}_Action_Factors",
                f"{label}_Consequence",
                f"{label}_Consequence_Band",
                f"{label}_Consequence_Reasoning",
                f"{label}_Consequence_Factors",
            ]
        return cols

    def write_csv(self):
        import pandas as pd

        cols = self._column_order()
        with self.lock:
            data = [self.rows[k] for k in sorted(self.rows.keys())]
        df = pd.DataFrame(data)
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols]
        OUTPUT_DIR.mkdir(exist_ok=True)
        df.to_csv(self.csv_path, index=False)


class JobView:
    """
    Read-only view of a job reconstructed from its on-disk status file.

    Job objects live only in the process that started them (in-memory,
    thread-locked). Any other process — e.g. the web app and the MCP server
    both talk to the same evaluation engine but run separately — needs this
    to see a job's status/results by id.
    """

    def __init__(self, data: dict):
        self._data = data
        self.id = data["id"]
        self.status = data["status"]
        self.csv_path = Path(data.get("csv_path", OUTPUT_DIR / f"job_{self.id}.csv"))

    def snapshot(self):
        return dict(self._data)

    def rows_snapshot(self):
        import pandas as pd

        if not self.csv_path.exists():
            return []
        return pd.read_csv(self.csv_path).to_dict(orient="records")


def _run_job(job: Job):
    try:
        df = load_dataset() if job.sample_size is None else sample_random(job.sample_size)
        job.total = len(df)

        with job.lock:
            for _, row in df.iterrows():
                job.rows[row["ID"]] = {
                    "ID": row["ID"],
                    "Scenario": row["input_sequence"],
                    "Prompt_Version": job.prompt_version,
                    "Prompt_Template_Hash": job.prompt_template_hash,
                    "Human_Action": row["Action_Valence"],
                    "Human_Consequence": row["Consequence_Valence"],
                }
        job.write_status()

        def run_model(model: str):
            label = MODEL_LABELS.get(model, model)
            for _, row in df.iterrows():
                try:
                    prediction = evaluate_single(row["input_sequence"], model=model, prompt_version=job.prompt_version)
                    action = prediction["action_valence"]
                    action_reasoning = prediction.get("action_reasoning", "")
                    action_factors = "; ".join(prediction.get("action_factors", []))
                    consequence = prediction["consequence_valence"]
                    consequence_reasoning = prediction.get("consequence_reasoning", "")
                    consequence_factors = "; ".join(prediction.get("consequence_factors", []))
                except Exception:
                    action = consequence = None
                    action_reasoning = consequence_reasoning = ""
                    action_factors = consequence_factors = ""
                    with job.lock:
                        job.errors_per_model[model] += 1

                with job.lock:
                    job.rows[row["ID"]][f"{label}_Action"] = action
                    job.rows[row["ID"]][f"{label}_Action_Band"] = valence_label(action)
                    job.rows[row["ID"]][f"{label}_Action_Reasoning"] = action_reasoning
                    job.rows[row["ID"]][f"{label}_Action_Factors"] = action_factors
                    job.rows[row["ID"]][f"{label}_Consequence"] = consequence
                    job.rows[row["ID"]][f"{label}_Consequence_Band"] = valence_label(consequence)
                    job.rows[row["ID"]][f"{label}_Consequence_Reasoning"] = consequence_reasoning
                    job.rows[row["ID"]][f"{label}_Consequence_Factors"] = consequence_factors
                    job.completed_per_model[model] += 1

                job.write_csv()
                job.write_status()

        with ThreadPoolExecutor(max_workers=len(job.models)) as pool:
            futures = [pool.submit(run_model, m) for m in job.models]
            for f in futures:
                f.result()

        job.write_csv()
        job.status = "completed"
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
    finally:
        job.write_status()


def start_job(models: list[str], sample_size: int | None = None, prompt_version: str = "current") -> Job:
    job = Job(models=models, sample_size=sample_size, prompt_version=prompt_version)
    _jobs[job.id] = job
    job.write_status()
    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()
    return job


def get_job(job_id: str) -> Job | JobView | None:
    if job_id in _jobs:
        return _jobs[job_id]

    status_path = _status_path(job_id)
    if status_path.exists():
        try:
            with open(status_path) as f:
                return JobView(json.load(f))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def list_jobs() -> list[dict]:
    """
    All known jobs, most recent first, regardless of which process started
    them — reads every status file on disk and overlays this process's live
    in-memory jobs (fresher than what's been flushed to disk).
    """
    snapshots: dict[str, dict] = {}

    for status_path in OUTPUT_DIR.glob("job_*.status.json"):
        try:
            with open(status_path) as f:
                data = json.load(f)
            snapshots[data["id"]] = data
        except (json.JSONDecodeError, OSError):
            continue

    for job_id, job in _jobs.items():
        snapshots[job_id] = job.snapshot()

    return sorted(snapshots.values(), key=lambda d: d.get("created_at", 0), reverse=True)
