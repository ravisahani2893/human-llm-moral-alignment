"""
CLI tool: run one model on the entire dataset, saving raw predictions to
outputs/output_<model>_entire_dataset.csv — no human comparison, no metrics,
just the model's own action/consequence valence + reasoning, one file per
model, meant as raw input for evaluation metrics computed separately later.

Saves progressively (one row at a time) and resumes automatically if
interrupted, so a long run never has to restart from scratch.

Usage:
    python -m tools.export_model_dataset --model gemini
    python -m tools.export_model_dataset --model gemini --limit 5   # quick test
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from app.dataset import load_dataset
from app.evaluator import MODEL_CLIENTS, evaluate_single

OUTPUT_DIR = Path("outputs")


def export_model_dataset(
    model: str,
    limit: int | None = None,
    resume: bool = True,
    on_progress=None,
) -> Path:
    if model not in MODEL_CLIENTS:
        raise ValueError(f"Unknown model {model!r}. Expected one of {list(MODEL_CLIENTS.keys())}")

    out_path = OUTPUT_DIR / f"output_{model.replace('/', '_')}_entire_dataset.csv"
    df = load_dataset()
    if limit:
        df = df.head(limit)
    total = len(df)

    rows: list[dict] = []
    done_ids: set = set()
    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        rows = existing.to_dict(orient="records")
        done_ids = set(existing["ID"].tolist())
        print(f"[export] resuming {model!r}: {len(done_ids)} scenarios already done", file=sys.stderr)

    remaining = df[~df["ID"].isin(done_ids)]
    print(f"[export] model={model!r}, total={total}, remaining={len(remaining)}", file=sys.stderr)

    action_col = f"{model}_action"
    consequences_col = f"{model}_consequences"

    for i, (_, row) in enumerate(remaining.iterrows(), start=1):
        try:
            prediction = evaluate_single(row["input_sequence"], model=model)
            new_row = {
                "ID": row["ID"],
                "Scenario": row["input_sequence"],
                action_col: prediction.get("action_valence"),
                consequences_col: prediction.get("consequence_valence"),
                "action_reasoning": prediction.get("action_reasoning", ""),
                "consequences_reasoning": prediction.get("consequence_reasoning", ""),
            }
        except Exception as exc:
            print(f"[export] scenario {row['ID']} FAILED: {exc}", file=sys.stderr)
            new_row = {
                "ID": row["ID"],
                "Scenario": row["input_sequence"],
                action_col: None,
                consequences_col: None,
                "action_reasoning": "",
                "consequences_reasoning": "",
            }

        rows.append(new_row)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[export] {len(done_ids) + i}/{total} done (ID {row['ID']})", file=sys.stderr)
        if on_progress:
            on_progress(len(done_ids) + i, total)

    print(f"[export] complete: {out_path}", file=sys.stderr)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run one model on the entire dataset, save to a per-model CSV.")
    parser.add_argument("--model", required=True, help=f"Model key: {list(MODEL_CLIENTS.keys())}")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N scenarios (for testing)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh instead of resuming a partial run")
    args = parser.parse_args()

    export_model_dataset(args.model, limit=args.limit, resume=not args.no_resume)


if __name__ == "__main__":
    main()
