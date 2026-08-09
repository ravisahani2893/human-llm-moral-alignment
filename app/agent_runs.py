import asyncio
import json
import threading
import time
import uuid
from pathlib import Path

from agents.general_agent import extract_csv_path, run_agent

OUTPUT_DIR = Path("outputs") / "agent_runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_runs: dict[str, "AgentRun"] = {}


def _status_path(run_id: str) -> Path:
    return OUTPUT_DIR / f"run_{run_id}.status.json"


class AgentRun:
    def __init__(self, instruction: str):
        self.id = uuid.uuid4().hex[:12]
        self.instruction = instruction
        self.status = "running"  # running | completed | error
        self.error: str | None = None
        self.report: str | None = None
        self.csv_path: str | None = None
        self.created_at = time.time()
        self.updated_at = self.created_at
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
                "instruction": self.instruction,
                "status": self.status,
                "error": self.error,
                "report": self.report,
                "csv_path": self.csv_path,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "log": list(self.log),
            }

    def write_status(self):
        self.updated_at = time.time()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(_status_path(self.id), "w") as f:
            json.dump(self.snapshot(), f)


def _run(agent_run: AgentRun, instruction: str):
    try:
        report, transcript = asyncio.run(run_agent(instruction, on_step=agent_run.append_log))
        csv_path = extract_csv_path(transcript)

        transcript_path = OUTPUT_DIR / f"run_{agent_run.id}.transcript.json"
        with open(transcript_path, "w") as f:
            json.dump(transcript, f, indent=2, default=str)

        with agent_run.lock:
            agent_run.report = report
            agent_run.csv_path = csv_path
            agent_run.status = "error" if report.startswith("(Agent stopped") else "completed"
    except Exception as exc:
        with agent_run.lock:
            agent_run.status = "error"
            agent_run.error = str(exc)
    finally:
        agent_run.write_status()


def start_agent_run(instruction: str) -> AgentRun:
    agent_run = AgentRun(instruction=instruction)
    _runs[agent_run.id] = agent_run
    agent_run.write_status()

    thread = threading.Thread(target=_run, args=(agent_run, instruction), daemon=True)
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
