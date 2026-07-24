import numpy as np
import pandas as pd

from app.dataset import load_dataset
from app.models import AlignmentReport, AxisMetrics, ModelMetrics, StratumMetrics

STRATA_COLUMNS = ["pattern", "source", "input_type"]

# Lin's CCC and pairwise CCC figures reported in the source paper ("Can
# Valence Reflect Morality in Natural Language? A Preliminary Annotation
# Study", Table II) — the human-annotator baseline this project's model
# CCC scores are meant to be read against, same metric, same scale.
PAPER_REFERENCE_CCC = {
    "action": {"pairwise_human": 0.260, "human_vs_ewe_gold_standard": 0.512},
    "consequence": {"pairwise_human": 0.356, "human_vs_ewe_gold_standard": 0.609},
}


def _clean_pair(human: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """Align two series and drop rows where either side is missing."""
    df = pd.DataFrame({"human": human, "predicted": predicted}).dropna()
    return df


def concordance_correlation_coefficient(human: pd.Series, predicted: pd.Series) -> float | None:
    """
    Lin's Concordance Correlation Coefficient (CCC) — the exact metric the
    source paper uses (its Eq. 1) to report human-annotator agreement.
    Unlike Pearson r, CCC penalizes both imprecision (low correlation) AND
    inaccuracy (a systematic mean/scale shift), so a model that's perfectly
    correlated with humans but consistently shifted still scores poorly —
    which is the property that makes it directly comparable to the paper's
    own reported numbers.

        CCC = 2*cov(a,b) / (var(a) + var(b) + (mean(a) - mean(b))^2)
    """
    df = _clean_pair(human, predicted)
    if len(df) < 2:
        return None

    mean_h, mean_p = df["human"].mean(), df["predicted"].mean()
    var_h, var_p = df["human"].var(), df["predicted"].var()
    covariance = df["human"].cov(df["predicted"])

    denominator = var_h + var_p + (mean_h - mean_p) ** 2
    if denominator == 0:
        return None

    ccc = (2 * covariance) / denominator
    return None if pd.isna(ccc) else round(float(ccc), 4)


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
            "ccc": None,
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

    pearson_r = spearman_r = ccc = None
    if n >= 2:
        pearson_raw = df["human"].corr(df["predicted"], method="pearson")
        spearman_raw = df["human"].corr(df["predicted"], method="spearman")
        pearson_r = None if pd.isna(pearson_raw) else round(float(pearson_raw), 4)
        spearman_r = None if pd.isna(spearman_raw) else round(float(spearman_raw), 4)
        ccc = concordance_correlation_coefficient(df["human"], df["predicted"])

    return {
        "n": n,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "ccc": ccc,
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
    Pairwise CCC between models' predictions on the same axis, to see
    whether models cluster together or diverge independently from each
    other (not just from humans) — the model-vs-model analog of the
    paper's own "pairwise annotator CCC" (Table II, µCCC_pairwise = 0.260
    action / 0.356 consequence for their 6 human annotators).

    model_columns: {label: {"action": col, "consequence": col}}
    """
    result = {"action": {}, "consequence": {}}
    labels = list(model_columns.keys())

    for axis in ("action", "consequence"):
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                col_a = model_columns[a][axis]
                col_b = model_columns[b][axis]
                ccc = concordance_correlation_coefficient(df[col_a], df[col_b])
                result[axis][f"{a} vs {b}"] = ccc

    return result


def full_agreement_matrix(df: pd.DataFrame, model_labels: list[str]) -> dict[str, pd.DataFrame]:
    """
    Symmetric CCC agreement matrix with every model AND "Human" as raters,
    per axis — a generalization of cross_model_agreement() (model-only) and
    model_metrics()'s per-model human-CCC into one unified (N+1)x(N+1) view.

    Returns {"action": DataFrame, "consequence": DataFrame}, each indexed
    and columned by rater name (model labels + "Human"), diagonal = 1.0.
    """
    raters = {label: {"action": f"{label}_Action", "consequence": f"{label}_Consequence"} for label in model_labels}
    raters["Human"] = {"action": "Human_Action", "consequence": "Human_Consequence"}

    matrices: dict[str, pd.DataFrame] = {}
    names = list(raters.keys())

    for axis in ("action", "consequence"):
        mat = pd.DataFrame(index=names, columns=names, dtype=float)
        for i, a in enumerate(names):
            mat.loc[a, a] = 1.0
            for b in names[i + 1:]:
                ccc = concordance_correlation_coefficient(df[raters[a][axis]], df[raters[b][axis]])
                mat.loc[a, b] = ccc
                mat.loc[b, a] = ccc
        matrices[axis] = mat

    return matrices


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


def compute_report(df: pd.DataFrame, model_labels: list[str]) -> AlignmentReport:
    """
    Full metrics report for a completed multi-model results dataframe
    (the shape produced by app.jobs.Job, one row per scenario with
    "{label}_Action" / "{label}_Consequence" columns per model).

    Ground truth is Human_Action/Human_Consequence — your own annotations
    (via the paper's R Shiny tool), already baked into the dataset.

    Returns a typed AlignmentReport rather than a raw dict, so callers
    (including MCP tools) get a discoverable schema instead of having to
    infer structure from docstring prose.
    """
    df = attach_strata(df) if "ID" in df.columns else df

    model_columns = {
        label: {"action": f"{label}_Action", "consequence": f"{label}_Consequence"}
        for label in model_labels
    }

    per_model: dict[str, ModelMetrics] = {}
    per_model_strata: dict[str, dict[str, list[StratumMetrics]]] = {}
    for label, cols in model_columns.items():
        if cols["action"] not in df.columns or cols["consequence"] not in df.columns:
            continue
        m = model_metrics(df, cols["action"], cols["consequence"])
        per_model[label] = ModelMetrics(
            action=AxisMetrics(**m["action"]),
            consequence=AxisMetrics(**m["consequence"]),
            combined=AxisMetrics(**m["combined"]),
        )
        per_model_strata[label] = {
            strata: [StratumMetrics(**row) for row in stratified_metrics(df, cols["action"], cols["consequence"], strata_col=strata)]
            for strata in STRATA_COLUMNS
            if strata in df.columns
        }

    return AlignmentReport(
        n_scenarios=len(df),
        models=per_model,
        cross_model_agreement=cross_model_agreement(df, model_columns) if len(model_columns) > 1 else {},
        stratified=per_model_strata,
    )
