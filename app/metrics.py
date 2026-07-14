import numpy as np
import pandas as pd

from app.dataset import load_dataset

STRATA_COLUMNS = ["pattern", "source", "input_type"]


def _clean_pair(human: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """Align two series and drop rows where either side is missing."""
    df = pd.DataFrame({"human": human, "predicted": predicted}).dropna()
    return df


def axis_metrics(human: pd.Series, predicted: pd.Series) -> dict:
    """
    Core alignment metrics between one axis of human labels and one axis
    of model predictions (works for either action or consequence valence).
    """
    df = _clean_pair(human, predicted)
    n = len(df)

    if n == 0:
        return {
            "n": 0,
            "pearson_r": None,
            "spearman_r": None,
            "mae": None,
            "rmse": None,
            "sign_agreement": None,
            "mean_bias": None,
        }

    error = df["predicted"] - df["human"]
    mae = error.abs().mean()
    rmse = float(np.sqrt((error**2).mean()))
    mean_bias = error.mean()  # positive = model scores harsher/more positive than humans on average

    # Sign agreement: treat values within +/-0.05 of 0 as "neutral" so near-zero
    # noise on both sides doesn't get penalized as a sign mismatch.
    def sign(v):
        if v > 0.05:
            return 1
        if v < -0.05:
            return -1
        return 0

    signs_match = (df["human"].apply(sign) == df["predicted"].apply(sign)).mean()

    pearson_r = spearman_r = None
    if n >= 2:
        pearson_raw = df["human"].corr(df["predicted"], method="pearson")
        spearman_raw = df["human"].corr(df["predicted"], method="spearman")
        pearson_r = None if pd.isna(pearson_raw) else round(float(pearson_raw), 4)
        spearman_r = None if pd.isna(spearman_raw) else round(float(spearman_raw), 4)

    return {
        "n": n,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "mae": round(float(mae), 4),
        "rmse": round(rmse, 4),
        "sign_agreement": round(float(signs_match), 4),
        "mean_bias": round(float(mean_bias), 4),
    }


def model_metrics(df: pd.DataFrame, action_col: str, consequence_col: str,
                   human_action_col: str = "Human_Action",
                   human_consequence_col: str = "Human_Consequence") -> dict:
    """
    Full metrics for one model: action axis, consequence axis, and both combined.
    """
    action = axis_metrics(df[human_action_col], df[action_col])
    consequence = axis_metrics(df[human_consequence_col], df[consequence_col])

    combined_human = pd.concat([df[human_action_col], df[human_consequence_col]], ignore_index=True)
    combined_pred = pd.concat([df[action_col], df[consequence_col]], ignore_index=True)
    combined = axis_metrics(combined_human, combined_pred)

    return {"action": action, "consequence": consequence, "combined": combined}


def cross_model_agreement(df: pd.DataFrame, model_columns: dict) -> dict:
    """
    Pairwise correlation between models' predictions on the same axis,
    to see whether models cluster together or diverge independently
    from each other (not just from humans).

    model_columns: {label: {"action": col, "consequence": col}}
    """
    result = {"action": {}, "consequence": {}}
    labels = list(model_columns.keys())

    for axis in ("action", "consequence"):
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                col_a = model_columns[a][axis]
                col_b = model_columns[b][axis]
                pair = _clean_pair(df[col_a], df[col_b])
                r = None
                if len(pair) >= 2:
                    r = pair["human"].corr(pair["predicted"], method="pearson")
                    r = None if pd.isna(r) else round(float(r), 4)
                result[axis][f"{a} vs {b}"] = r

    return result


def stratified_metrics(df: pd.DataFrame, action_col: str, consequence_col: str,
                        human_action_col: str = "Human_Action",
                        human_consequence_col: str = "Human_Consequence",
                        strata_col: str = "pattern") -> list[dict]:
    """
    Break down MAE and sign agreement by a dataset metadata column
    (pattern / source / input_type), to see where a model diverges most
    from humans rather than just an aggregate score.
    """
    if strata_col not in df.columns:
        return []

    rows = []
    for value, group in df.groupby(strata_col):
        action = axis_metrics(group[human_action_col], group[action_col])
        consequence = axis_metrics(group[human_consequence_col], group[consequence_col])
        rows.append({
            "value": value,
            "n": len(group),
            "action_mae": action["mae"],
            "action_sign_agreement": action["sign_agreement"],
            "consequence_mae": consequence["mae"],
            "consequence_sign_agreement": consequence["sign_agreement"],
        })

    rows.sort(key=lambda r: (r["action_mae"] is None, r["action_mae"]), reverse=True)
    return rows


def attach_strata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join dataset metadata (pattern/source/input_type) onto a results
    dataframe by ID, so stratified breakdowns can be computed even though
    the job/results CSV itself doesn't carry those columns.
    """
    dataset = load_dataset()[["ID"] + STRATA_COLUMNS]
    return df.merge(dataset, on="ID", how="left")


def compute_report(df: pd.DataFrame, model_labels: list[str]) -> dict:
    """
    Full metrics report for a completed multi-model results dataframe
    (the shape produced by app.jobs.Job, one row per scenario with
    "{label}_Action" / "{label}_Consequence" columns per model).
    """
    df = attach_strata(df) if "ID" in df.columns else df

    model_columns = {
        label: {"action": f"{label}_Action", "consequence": f"{label}_Consequence"}
        for label in model_labels
    }

    per_model = {}
    per_model_strata = {}
    for label, cols in model_columns.items():
        if cols["action"] not in df.columns or cols["consequence"] not in df.columns:
            continue
        per_model[label] = model_metrics(df, cols["action"], cols["consequence"])
        per_model_strata[label] = {
            strata: stratified_metrics(df, cols["action"], cols["consequence"], strata_col=strata)
            for strata in STRATA_COLUMNS
            if strata in df.columns
        }

    return {
        "n_scenarios": len(df),
        "models": per_model,
        "cross_model_agreement": cross_model_agreement(df, model_columns) if len(model_columns) > 1 else {},
        "stratified": per_model_strata,
    }
