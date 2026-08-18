"""
Finds scenarios where the 5 models broadly AGREE on Action Valence (same
sign, low spread) but DISAGREE on Consequence Valence (mixed sign, high
spread) — the "models condemn the act but split on the outcome" pattern.

Zero-shot ("current" prompt version) only by default. Reads the same
per-model export CSVs used by evaluate_alignment/calculate_cross_model_agreement
(outputs/output_<model>_entire_dataset.csv), merges by scenario ID, and
writes one row per qualifying scenario to
outputs/action_consequence_divergence.csv, sorted by how large the gap is
between action agreement and consequence disagreement (largest gap first).

Usage:
    python -m tools.find_action_consequence_divergence
"""
import pandas as pd

from app.evaluator import MODEL_CLIENTS

OUTPUT_CSV = "outputs/action_consequence_divergence.csv"


def _compute_divergence_table(prompt_version: str = "current") -> tuple[pd.DataFrame, list[str]]:
    """
    Shared merge + per-scenario statistics computation, reused by both
    find_divergence() (candidates only, DataFrame shape for the CSV export)
    and find_divergence_report() (candidates + totals, dict shape for the
    MCP tool) so the two never compute this differently.
    """
    models = list(MODEL_CLIENTS.keys())
    suffix = "" if prompt_version == "current" else f"_{prompt_version}"

    per_model = {}
    for model in models:
        path = f"outputs/output_{model.replace('/', '_')}{suffix}_entire_dataset.csv"
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Predictions file not found for model {model!r}: {path}. "
                f"Run start_dataset_export for this model/prompt_version first."
            )
        per_model[model] = df[["ID", "Scenario", f"{model}_action", f"{model}_consequences"]].dropna()

    merged = per_model[models[0]][["ID", "Scenario"]]
    for model in models:
        merged = merged.merge(
            per_model[model][["ID", f"{model}_action", f"{model}_consequences"]], on="ID", how="inner"
        )

    action_cols = [f"{m}_action" for m in models]
    cons_cols = [f"{m}_consequences" for m in models]

    merged["action_std"] = merged[action_cols].std(axis=1)
    merged["consequence_std"] = merged[cons_cols].std(axis=1)
    merged["action_all_same_sign"] = merged[action_cols].apply(lambda r: (r > 0).all() or (r < 0).all(), axis=1)
    merged["consequence_sign_mixed"] = merged[cons_cols].apply(
        lambda r: not ((r > 0).all() or (r < 0).all()), axis=1
    )
    merged["divergence_gap"] = merged["consequence_std"] - merged["action_std"]

    return merged, models


def find_divergence(prompt_version: str = "current") -> pd.DataFrame:
    merged, models = _compute_divergence_table(prompt_version)

    candidates = merged[merged["action_all_same_sign"] & merged["consequence_sign_mixed"]].copy()
    candidates = candidates.sort_values("divergence_gap", ascending=False).reset_index(drop=True)

    ordered_cols = ["ID", "Scenario", "action_std", "consequence_std", "divergence_gap"]
    for m in models:
        ordered_cols += [f"{m}_action", f"{m}_consequences"]
    return candidates[ordered_cols]


def find_divergence_report(prompt_version: str = "current", top_n: int = 20) -> dict:
    """
    Same underlying computation as find_divergence(), returned as a plain
    JSON-serialisable dict (Pydantic-model-ready) instead of a DataFrame,
    for the MCP tool / agent path. top_n caps how many of the qualifying
    scenarios are returned in full detail (they're already sorted by
    divergence_gap, largest first, so top_n keeps the most interesting
    ones) — the full set's size is still reported via n_divergent even
    when top_n truncates the "results" list.
    """
    merged, models = _compute_divergence_table(prompt_version)

    n_total = len(merged)
    candidates = merged[merged["action_all_same_sign"] & merged["consequence_sign_mixed"]].copy()
    candidates = candidates.sort_values("divergence_gap", ascending=False).reset_index(drop=True)
    n_divergent = len(candidates)

    results = []
    for _, row in candidates.head(top_n).iterrows():
        results.append({
            "id": int(row["ID"]),
            "scenario": row["Scenario"],
            "action_std": float(row["action_std"]),
            "consequence_std": float(row["consequence_std"]),
            "divergence_gap": float(row["divergence_gap"]),
            "per_model": {
                m: {"action": float(row[f"{m}_action"]), "consequence": float(row[f"{m}_consequences"])}
                for m in models
            },
        })

    return {
        "analysis": "action_consequence_divergence",
        "prompt_version": prompt_version,
        "models": models,
        "n_scenarios_total": n_total,
        "n_divergent": n_divergent,
        "divergent_fraction": n_divergent / n_total if n_total else 0.0,
        "results": results,
    }


def main():
    result = find_divergence()
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"[divergence] {len(result)} scenarios written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
