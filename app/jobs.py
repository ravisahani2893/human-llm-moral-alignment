import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.dataset import load_dataset, sample_random
from app.evaluator import evaluate_single, MODEL_LABELS
from app.interpret import valence_label

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

_jobs: dict[str, "Job"] = {}


class Job:
    def __init__(self, models: list[str], sample_size: int | None):
        self.id = uuid.uuid4().hex[:12]
        self.models = models
        self.sample_size = sample_size
        self.status = "running"  # running | completed | error
        self.error = None
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
                "total": self.total,
                "completed_per_model": dict(self.completed_per_model),
                "errors_per_model": dict(self.errors_per_model),
            }

    def rows_snapshot(self):
        with self.lock:
            return [self.rows[k] for k in sorted(self.rows.keys())]

    def _column_order(self):
        cols = ["ID", "Scenario", "Human_Action", "Human_Consequence"]
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
        df.to_csv(self.csv_path, index=False)


def _run_job(job: Job):
    try:
        df = load_dataset() if job.sample_size is None else sample_random(job.sample_size)
        job.total = len(df)

        with job.lock:
            for _, row in df.iterrows():
                job.rows[row["ID"]] = {
                    "ID": row["ID"],
                    "Scenario": row["input_sequence"],
                    "Human_Action": row["Action_Valence"],
                    "Human_Consequence": row["Consequence_Valence"],
                }

        def run_model(model: str):
            label = MODEL_LABELS.get(model, model)
            for _, row in df.iterrows():
                try:
                    prediction = evaluate_single(row["input_sequence"], model=model)
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

        with ThreadPoolExecutor(max_workers=len(job.models)) as pool:
            futures = [pool.submit(run_model, m) for m in job.models]
            for f in futures:
                f.result()

        job.write_csv()
        job.status = "completed"
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)


def start_job(models: list[str], sample_size: int | None = None) -> Job:
    job = Job(models=models, sample_size=sample_size)
    _jobs[job.id] = job
    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)
