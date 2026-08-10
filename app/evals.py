import statistics

import pandas as pd

from app.dataset import load_dataset
from app.evaluator import evaluate_single
from app.metric import calculate_ccc, calculate_mae, calculate_pearson, calculate_rmse, calculate_spearman
from app.prompt_versions import available_versions

FIXED_SAMPLE_SIZE = 30
FIXED_SAMPLE_SEED = 42

# Scenario IDs used as worked examples in app/prompts/few_shot.txt. Excluded
# from the fixed evaluation sample here (not just by seed luck) so the
# few_shot prompt version is never evaluated on scenarios it was shown the
# answer to — regardless of future changes to sample size/seed.
FEW_SHOT_EXAMPLE_IDS = {918029, 449683, 37563, 833314, 402529, 507964, 934149, 915520, 356665, 209012}


def _axis_metrics(human: pd.Series, predicted: pd.Series) -> dict:
    """
    Alignment metrics between one axis of human labels and one axis of
    model predictions, built on app.metric's calculate_* functions
    (replaces the retired app.metrics module).
    """
    human = pd.Series(human).astype(float)
    predicted = pd.Series(predicted).astype(float)
    n = len(human)

    if n == 0:
        return {
            "n": 0, "pearson_r": None, "spearman_r": None, "ccc": None,
            "mae": None, "rmse": None, "sign_agreement": None, "mean_bias": None,
        }

    def sign(v):
        if v > 0.05:
            return 1
        if v < -0.05:
            return -1
        return 0

    signs_match = (human.apply(sign) == predicted.apply(sign)).mean()
    mean_bias = (predicted - human).mean()

    pearson_r = spearman_r = ccc = None
    if n >= 2:
        pearson_r = round(float(calculate_pearson(human, predicted)), 4)
        spearman_r = round(float(calculate_spearman(human, predicted)), 4)
        ccc = round(float(calculate_ccc(human, predicted)), 4)

    return {
        "n": n,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "ccc": ccc,
        "mae": round(float(calculate_mae(human, predicted)), 4),
        "rmse": round(float(calculate_rmse(human, predicted)), 4),
        "sign_agreement": round(float(signs_match), 4),
        "mean_bias": round(float(mean_bias), 4),
    }


def fixed_sample_ids(size: int = FIXED_SAMPLE_SIZE, seed: int = FIXED_SAMPLE_SEED) -> list[int]:
    """
    Fixed-seed sample of scenario IDs used as a repeatable regression set.
    Not hand-curated for difficulty (a reasonable future improvement) —
    what matters here is reproducibility: the same seed always returns the
    same IDs, so results are comparable run over run and version over
    version.
    """
    df = load_dataset()
    df = df[~df["ID"].isin(FEW_SHOT_EXAMPLE_IDS)]
    sample = df.sample(n=min(size, len(df)), random_state=seed)
    return sample["ID"].tolist()


def evaluate_fixed_sample(model: str, version: str = "current", size: int = FIXED_SAMPLE_SIZE) -> dict:
    """
    Run the fixed evaluation sample through one model/prompt-version
    combination and report alignment metrics against the human labels.
    """
    ids = fixed_sample_ids(size)
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
        action = consequence = _axis_metrics(pd.Series(dtype=float), pd.Series(dtype=float))
    else:
        action = _axis_metrics(result_df["Human_Action"], result_df["Predicted_Action"])
        consequence = _axis_metrics(result_df["Human_Consequence"], result_df["Predicted_Consequence"])

    return {
        "model": model,
        "version": version,
        "n_requested": len(ids),
        "n_evaluated": len(rows),
        "n_errors": errors,
        "action": action,
        "consequence": consequence,
    }


def compare_prompt_versions(model: str, size: int = FIXED_SAMPLE_SIZE, versions: list[str] | None = None) -> list[dict]:
    """
    Run the same fixed evaluation sample through every prompt version (v1,
    v2, current by default) for one model, so a version change can be
    judged by measured alignment against the human labels instead of by
    unrecorded impression.
    """
    versions = versions or available_versions()
    return [evaluate_fixed_sample(model, version=v, size=size) for v in versions]


def stability_check(model: str, sample_size: int = 5, repeats: int = 3) -> list[dict]:
    """
    Re-run the same scenarios multiple times to quantify how much a
    model's valence output varies run-to-run (sampling noise) versus a
    genuine, stable disagreement with humans. High stdev here means some
    of the observed "divergence from humans" elsewhere may be noise, not
    signal.
    """
    full_df = load_dataset()
    df = full_df.sample(n=min(sample_size, len(full_df)), random_state=FIXED_SAMPLE_SEED)

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
