"""
CLI tool: compare prompt versions on the same golden set (fixed-seed
reproducible sample), reporting real alignment metrics per version — for
proving a prompt change actually helped, not just asserting it.

Usage:
    python -m tools.compare_prompts --model gemini --versions v2,current --size 15
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from app.evals import compare_prompt_versions
from app.prompt_versions import compute_prompt_hash

OUTPUT_DIR = Path("outputs")


def main():
    parser = argparse.ArgumentParser(description="Compare prompt versions on the same golden set.")
    parser.add_argument("--model", default="gemini", help="Model key to evaluate")
    parser.add_argument("--versions", default="v2,current", help="Comma-separated prompt versions to compare")
    parser.add_argument("--size", type=int, default=15, help="Golden set size (fixed-seed, reproducible)")
    parser.add_argument("--out", default="prompt_comparison.csv", help="Output filename (under outputs/)")
    args = parser.parse_args()

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]

    print(f"[compare] model={args.model!r}, versions={versions}, golden set size={args.size}", file=sys.stderr)
    for v in versions:
        print(f"[compare]   {v} -> template hash {compute_prompt_hash(v)}", file=sys.stderr)

    results = compare_prompt_versions(model=args.model, size=args.size, versions=versions)

    rows = []
    for r in results:
        rows.append({
            "version": r["version"],
            "prompt_template_hash": compute_prompt_hash(r["version"]),
            "n_evaluated": r["n_evaluated"],
            "n_errors": r["n_errors"],
            "action_ccc": r["action"]["ccc"],
            "action_pearson_r": r["action"]["pearson_r"],
            "action_mae": r["action"]["mae"],
            "action_sign_agreement": r["action"]["sign_agreement"],
            "consequence_ccc": r["consequence"]["ccc"],
            "consequence_pearson_r": r["consequence"]["pearson_r"],
            "consequence_mae": r["consequence"]["mae"],
            "consequence_sign_agreement": r["consequence"]["sign_agreement"],
        })

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / args.out
    df.to_csv(out_path, index=False)

    print(f"\n=== Prompt version comparison: {args.model} (golden set n={args.size}) ===")
    print(df.to_string(index=False))
    print(f"\n[compare] saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
