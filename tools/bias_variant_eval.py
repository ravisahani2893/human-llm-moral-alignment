"""
CLI tool: run one model on every demographic variant of a perturbation
dataset (data/variants/variants_GENDER.csv, variants_ETHNICITY.csv, ...),
saving raw predictions to outputs/bias_<dataset>_<model>.csv.

Each row in a variants CSV holds the SAME underlying scenario rewritten
with different names/pronouns (e.g. Male/Female, or Indian/European/
American) plus the ORIGINAL human Action_Valence/Consequence_Valence for
that scenario (the moral content hasn't changed, only the demographic
marker) — so this is a counterfactual perturbation test, not a new
annotation task. Predictions are keyed by (ID, variant), long-format, one
row per scenario per variant, same resumable-with-retry pattern as
tools/export_model_dataset.py.

Usage:
    python -m tools.bias_variant_eval --model gemini --dataset GENDER
    python -m tools.bias_variant_eval --model gemini --dataset ETHNICITY --limit 5
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from app.evaluator import MODEL_CLIENTS, evaluate_single

VARIANTS_DIR = Path("data/variants")
OUTPUT_DIR = Path("outputs")

# Columns in a variants CSV that are metadata, not scenario-text variants.
_NON_VARIANT_COLUMNS = {"ID", "Action_Valence", "Consequence_Valence", "origin_gender", "review"}


def variants_input_path(dataset: str) -> Path:
    return VARIANTS_DIR / f"variants_{dataset.upper()}.csv"


def available_variant_datasets() -> list[str]:
    return sorted(p.stem.removeprefix("variants_") for p in VARIANTS_DIR.glob("variants_*.csv"))


def bias_output_path(dataset: str, model: str) -> Path:
    return OUTPUT_DIR / f"bias_{dataset.upper()}_{model.replace('/', '_')}.csv"


def variant_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_VARIANT_COLUMNS]


def run_bias_variant_eval(
    model: str,
    dataset: str,
    limit: int | None = None,
    resume: bool = True,
    prompt_version: str = "current",
    on_progress=None,
) -> Path:
    if model not in MODEL_CLIENTS:
        raise ValueError(f"Unknown model {model!r}. Expected one of {list(MODEL_CLIENTS.keys())}")

    in_path = variants_input_path(dataset)
    if not in_path.exists():
        raise FileNotFoundError(f"Variants file not found: {in_path}")

    df = pd.read_csv(in_path)
    if limit:
        df = df.head(limit)
    variants = variant_columns(df)

    # long format: one (ID, variant) unit of work per row * per variant column
    units = [(row["ID"], variant, row[variant]) for _, row in df.iterrows() for variant in variants]
    total = len(units)

    out_path = bias_output_path(dataset, model)
    rows: list[dict] = []
    done_keys: set = set()
    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        failed_mask = existing["action_valence"].isna() | existing["consequence_valence"].isna()
        completed = existing[~failed_mask]
        rows = completed.to_dict(orient="records")
        done_keys = set(zip(completed["ID"], completed["variant"]))
        print(
            f"[bias-eval] resuming {model!r}/{dataset!r}: {len(done_keys)} already done, "
            f"{int(failed_mask.sum())} previously failed will be retried",
            file=sys.stderr,
        )

    remaining = [(sid, variant, text) for sid, variant, text in units if (sid, variant) not in done_keys]
    print(f"[bias-eval] model={model!r}, dataset={dataset!r}, total={total}, remaining={len(remaining)}", file=sys.stderr)

    for i, (scenario_id, variant, text) in enumerate(remaining, start=1):
        try:
            prediction = evaluate_single(text, model=model, prompt_version=prompt_version)
            new_row = {
                "ID": scenario_id,
                "variant": variant,
                "scenario_text": text,
                "action_valence": prediction.get("action_valence"),
                "consequence_valence": prediction.get("consequence_valence"),
                "action_reasoning": prediction.get("action_reasoning", ""),
                "consequence_reasoning": prediction.get("consequence_reasoning", ""),
            }
        except Exception as exc:
            print(f"[bias-eval] {scenario_id}/{variant} FAILED: {exc}", file=sys.stderr)
            new_row = {
                "ID": scenario_id,
                "variant": variant,
                "scenario_text": text,
                "action_valence": None,
                "consequence_valence": None,
                "action_reasoning": "",
                "consequence_reasoning": "",
            }

        rows.append(new_row)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[bias-eval] {len(done_keys) + i}/{total} done (ID {scenario_id}, variant {variant})", file=sys.stderr)
        if on_progress:
            on_progress(len(done_keys) + i, total)

    print(f"[bias-eval] complete: {out_path}", file=sys.stderr)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run one model on every variant of a bias perturbation dataset.")
    parser.add_argument("--model", required=True, help=f"Model key: {list(MODEL_CLIENTS.keys())}")
    parser.add_argument("--dataset", required=True, help="Variants dataset name, e.g. GENDER or ETHNICITY")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N scenarios (for testing)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh instead of resuming a partial run")
    parser.add_argument("--prompt-version", default="current", help="Prompt version to use")
    args = parser.parse_args()

    run_bias_variant_eval(
        args.model, args.dataset, limit=args.limit, resume=not args.no_resume, prompt_version=args.prompt_version
    )


if __name__ == "__main__":
    main()
