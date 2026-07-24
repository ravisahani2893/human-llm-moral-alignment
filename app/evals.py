import statistics

import pandas as pd

from app.dataset import load_dataset
from app.evaluator import evaluate_single
from app.metrics import axis_metrics
from app.prompt_versions import available_versions

GOLDEN_SET_SIZE = 30
GOLDEN_SET_SEED = 42


def golden_set_ids(size: int = GOLDEN_SET_SIZE, seed: int = GOLDEN_SET_SEED) -> list[int]:
    """
    Fixed-seed sample of scenario IDs used as a repeatable regression set.
    Not hand-curated for difficulty (a reasonable future improvement) —
    what matters here is reproducibility: the same seed always returns the
    same IDs, so results are comparable run over run and version over
    version, which is the whole point of a golden set.
    """
    df = load_dataset()
    sample = df.sample(n=min(size, len(df)), random_state=seed)
    return sample["ID"].tolist()


def evaluate_golden_set(model: str, version: str = "current", size: int = GOLDEN_SET_SIZE) -> dict:
    """
    Run the golden set through one model/prompt-version combination and
    report alignment metrics against the human gold standard.
    """
    ids = golden_set_ids(size)
    df = load_dataset()
    subset = df[df["ID"].isin(ids)]

    rows = []
    errors = 0
    for _, row in subset.iterrows():
        try:
            prediction = evaluate_single(row["input_sequence"], model=model, prompt_version=version)
            rows.append({
                "ID": row["ID"],
                "Human_Action": row["Action_Valence"],
                "Human_Consequence": row["Consequence_Valence"],
                "Predicted_Action": prediction["action_valence"],
                "Predicted_Consequence": prediction["consequence_valence"],
            })
        except Exception:
            errors += 1

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        action = consequence = axis_metrics(pd.Series(dtype=float), pd.Series(dtype=float))
    else:
        action = axis_metrics(result_df["Human_Action"], result_df["Predicted_Action"])
        consequence = axis_metrics(result_df["Human_Consequence"], result_df["Predicted_Consequence"])

    return {
        "model": model,
        "version": version,
        "n_requested": len(ids),
        "n_evaluated": len(rows),
        "n_errors": errors,
        "action": action,
        "consequence": consequence,
    }


def compare_prompt_versions(model: str, size: int = GOLDEN_SET_SIZE, versions: list[str] | None = None) -> list[dict]:
    """
    Run the same golden set through every prompt version (v1, v2, current
    by default) for one model, so a version change can be judged by
    measured alignment against the human gold standard instead of by
    unrecorded impression.
    """
    versions = versions or available_versions()
    return [evaluate_golden_set(model, version=v, size=size) for v in versions]


def stability_check(model: str, sample_size: int = 5, repeats: int = 3) -> list[dict]:
    """
    Re-run the same scenarios multiple times to quantify how much a
    model's valence output varies run-to-run (sampling noise) versus a
    genuine, stable disagreement with humans. High stdev here means some
    of the observed "divergence from humans" elsewhere may be noise, not
    signal.
    """
    full_df = load_dataset()
    df = full_df.sample(n=min(sample_size, len(full_df)), random_state=GOLDEN_SET_SEED)

    results = []
    for _, row in df.iterrows():
        action_vals, consequence_vals = [], []
        for _ in range(repeats):
            try:
                prediction = evaluate_single(row["input_sequence"], model=model)
                action_vals.append(prediction["action_valence"])
                consequence_vals.append(prediction["consequence_valence"])
            except Exception:
                continue

        results.append({
            "ID": int(row["ID"]),
            "n_successful_runs": len(action_vals),
            "action_mean": round(statistics.mean(action_vals), 4) if action_vals else None,
            "action_stdev": round(statistics.stdev(action_vals), 4) if len(action_vals) > 1 else None,
            "consequence_mean": round(statistics.mean(consequence_vals), 4) if consequence_vals else None,
            "consequence_stdev": round(statistics.stdev(consequence_vals), 4) if len(consequence_vals) > 1 else None,
        })
    return results
