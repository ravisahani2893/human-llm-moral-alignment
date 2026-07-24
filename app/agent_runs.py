import asyncio
import json
import threading
import time
import uuid
from pathlib import Path

from agents.divergence_analyst import build_task, extract_csv_path, extract_scenario_evaluations, run_agent
from app.dataset import load_dataset

OUTPUT_DIR = Path("outputs") / "agent_runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_runs: dict[str, "AgentRun"] = {}


def _status_path(run_id: str) -> Path:
    return OUTPUT_DIR / f"run_{run_id}.status.json"


def lookup_human_labels(scenario: str) -> dict | None:
    """
    If the given scenario text exactly matches a row in the labeled
    dataset, return its human gold-standard labels — lets single-scenario
    mode show a real human-vs-model comparison instead of only comparing
    models against each other, whenever the demo scenario happens to be
    (or is deliberately chosen to be) one from the dataset.
    """
    df = load_dataset()
    match = df[df["input_sequence"].str.strip() == scenario.strip()]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "ID": int(row["ID"]),
        "Human_Action": float(row["Action_Valence"]),
        "Human_Consequence": float(row["Consequence_Valence"]),
    }


class AgentRun:
    def __init__(self, mode: str, models: list[str], scenario: str | None, sample_size: int | None):
        self.id = uuid.uuid4().hex[:12]
        self.mode = mode
        self.models = models
        self.scenario = scenario
        self.sample_size = sample_size
        self.status = "running"  # running | completed | error
        self.error: str | None = None
        self.report: str | None = None
        self.csv_path: str | None = None
        self.scenario_evaluations: list[dict] = []
        self.human_reference: dict | None = None
        self.created_at = time.time()
        self.log: list[dict] = []
        self.lock = threading.Lock()

    def append_log(self, entry: dict):
        with self.lock:
            self.log.append(entry)
        self.write_status()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "id": self.id,
                "mode": self.mode,
                "models": self.models,
                "scenario": self.scenario,
                "sample_size": self.sample_size,
                "status": self.status,
                "error": self.error,
                "report": self.report,
                "csv_path": self.csv_path,
                "scenario_evaluations": list(self.scenario_evaluations),
                "human_reference": self.human_reference,
                "created_at": self.created_at,
                "log": list(self.log),
            }

    def write_status(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(_status_path(self.id), "w") as f:
            json.dump(self.snapshot(), f)


def _run(agent_run: AgentRun, task: str):
    try:
        report, transcript = asyncio.run(run_agent(task, on_step=agent_run.append_log))
        csv_path = extract_csv_path(transcript)

        transcript_path = OUTPUT_DIR / f"run_{agent_run.id}.transcript.json"
        with open(transcript_path, "w") as f:
            json.dump(transcript, f, indent=2, default=str)

        with agent_run.lock:
            agent_run.report = report
            agent_run.csv_path = csv_path
            if agent_run.mode == "scenario":
                agent_run.scenario_evaluations = extract_scenario_evaluations(transcript)
            agent_run.status = "error" if report.startswith("(Agent stopped") else "completed"
    except Exception as exc:
        with agent_run.lock:
            agent_run.status = "error"
            agent_run.error = str(exc)
    finally:
        agent_run.write_status()


def start_agent_run(
    mode: str,
    models: list[str],
    scenario: str | None = None,
    sample_size: int | None = None,
) -> AgentRun:
    human_reference = lookup_human_labels(scenario) if mode == "scenario" and scenario else None
    task = build_task(mode, models, scenario=scenario, sample_size=sample_size, human_reference=human_reference)

    agent_run = AgentRun(mode=mode, models=models, scenario=scenario, sample_size=sample_size)
    agent_run.human_reference = human_reference
    _runs[agent_run.id] = agent_run
    agent_run.write_status()

    thread = threading.Thread(target=_run, args=(agent_run, task), daemon=True)
    thread.start()
    return agent_run


def get_agent_run(run_id: str) -> dict | None:
    if run_id in _runs:
        return _runs[run_id].snapshot()

    status_path = _status_path(run_id)
    if status_path.exists():
        try:
            with open(status_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def list_agent_runs() -> list[dict]:
    snapshots: dict[str, dict] = {}

    for status_path in OUTPUT_DIR.glob("run_*.status.json"):
        try:
            with open(status_path) as f:
                data = json.load(f)
            snapshots[data["id"]] = data
        except (json.JSONDecodeError, OSError):
            continue

    for run_id, run in _runs.items():
        snapshots[run_id] = run.snapshot()

    return sorted(snapshots.values(), key=lambda d: d.get("created_at", 0), reverse=True)
