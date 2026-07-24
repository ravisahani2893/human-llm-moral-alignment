"""
CLI tool: run a multi-model evaluation (or reuse an existing job's CSV) and
export the full CCC agreement matrix — every model AND Human as raters,
per axis — as CSV. Meant for a command-line/CSV-based demo, no web UI needed.

Usage:
    python -m tools.agreement_matrix --models gemini,deep-seek --sample-size 15
    python -m tools.agreement_matrix --csv outputs/job_<id>.csv --models gemini,deep-seek
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from app.evaluator import MODEL_LABELS
from app.jobs import start_job
from app.metrics import full_agreement_matrix

OUTPUT_DIR = Path("outputs")


def run_and_export(models: list[str], sample_size: int | None, csv_path: str | None, out_prefix: str):
    if csv_path:
        df = pd.read_csv(csv_path)
        model_labels = [MODEL_LABELS.get(m, m) for m in models]
        print(f"[matrix] using existing CSV: {csv_path}", file=sys.stderr)
    else:
        model_labels = [MODEL_LABELS.get(m, m) for m in models]
        print(f"[matrix] starting evaluation: models={models}, sample_size={sample_size}", file=sys.stderr)
        job = start_job(models=models, sample_size=sample_size)

        while job.status == "running":
            snap = job.snapshot()
            print(f"[matrix] {snap['status']} {snap['completed_per_model']}", file=sys.stderr)
            time.sleep(3)

        if job.status != "completed":
            print(f"[matrix] job failed: {job.error}", file=sys.stderr)
            sys.exit(1)

        df = pd.read_csv(job.csv_path)
        print(f"[matrix] evaluation complete: {job.csv_path}", file=sys.stderr)

    matrices = full_agreement_matrix(df, model_labels)

    OUTPUT_DIR.mkdir(exist_ok=True)
    for axis, mat in matrices.items():
        path = OUTPUT_DIR / f"{out_prefix}_{axis}.csv"
        mat.round(4).to_csv(path)
        print(f"\n=== {axis.capitalize()} CCC agreement matrix (n={len(df)}) ===")
        print(mat.round(3).to_string())
        print(f"[matrix] saved to {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Export a full CCC agreement matrix (models + Human) as CSV.")
    parser.add_argument("--models", default="gemini,deep-seek", help="Comma-separated model keys")
    parser.add_argument("--sample-size", type=int, default=15, help="Scenarios to sample (ignored if --csv given)")
    parser.add_argument("--csv", default=None, help="Reuse an existing job CSV instead of running a new evaluation")
    parser.add_argument("--out-prefix", default="agreement_matrix", help="Output filename prefix (under outputs/)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    run_and_export(models, args.sample_size, args.csv, args.out_prefix)


if __name__ == "__main__":
    main()
