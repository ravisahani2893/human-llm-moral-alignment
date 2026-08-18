import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from app.dataset import load_dataset, sample_random, save_predictions
from app.interpret import valence_label
from app.metric import calculate_ccc, calculate_mae, calculate_pearson, calculate_rmse, calculate_spearman, calculate_wilcoxon
from app.prompts import build_prompt
from clients.claude_client import ask_claude
from clients.deepseek_client import ask_deepseek
from clients.gemini_client import ask_gemini
from clients.groq_client import ask_groq
from clients.openai_client import ask_openai

MODEL_CLIENTS = {
    "gemini": lambda prompt: ask_gemini(prompt),
    "lama": lambda prompt: ask_groq(prompt, model="llama-3.3-70b-versatile"),
    "deep-seek": lambda prompt: ask_deepseek(prompt, model="deepseek-v4-pro"),
    "gpt-github": lambda prompt: ask_openai(prompt, model="gpt-4.1"),
    "claude": lambda prompt: ask_claude(prompt),
}

MODEL_LABELS = {
    "gemini": "Gemini 2.5 Flash",
    "lama": "Llama 3.3 70B",
    "deep-seek": "DeepSeek V4 Pro",
    "gpt-github": "GPT-4.1",
    "claude": "Claude Sonnet 5",
}


def _resolve_client(model: str):
    if model in MODEL_CLIENTS:
        return MODEL_CLIENTS[model]

    raise ValueError(
        f"Unsupported model: {model!r}. "
        f"Expected one of {list(MODEL_CLIENTS.keys())}"
    )


def evaluate_single(scenario: str, model: str = "gemini", prompt_version: str = "current"):

    print(f"[evaluator] evaluate_single called with model={model!r}, prompt_version={prompt_version!r}", file=sys.stderr)

    if prompt_version == "current":
        prompt = build_prompt(scenario)
    else:
        from app.prompt_versions import get_prompt_builder
        prompt = get_prompt_builder(prompt_version)(scenario)

    client_fn = _resolve_client(model)
    response = client_fn(prompt)

    response = response.strip()
    response = response.replace("```json", "")
    response = response.replace("```", "")

    return json.loads(response)


def evaluate_models_for_scenario(scenario: str, models: list[str]):
    """
    Evaluate a single scenario against multiple models, in parallel.
    Returns one result dict per model; failures are captured per-model
    instead of aborting the whole batch.
    """
    results = []

    def _run(model: str):
        try:
            prediction = evaluate_single(scenario, model=model)
            action_valence = prediction["action_valence"]
            consequence_valence = prediction["consequence_valence"]
            return {
                "model": model,
                "label": MODEL_LABELS.get(model, model),
                "action_valence": action_valence,
                "action_band": valence_label(action_valence),
                "action_reasoning": prediction.get("action_reasoning", ""),
                "action_factors": prediction.get("action_factors", []),
                "consequence_valence": consequence_valence,
                "consequence_band": valence_label(consequence_valence),
                "consequence_reasoning": prediction.get("consequence_reasoning", ""),
                "consequence_factors": prediction.get("consequence_factors", []),
                "error": None,
            }
        except Exception as exc:
            return {
                "model": model,
                "label": MODEL_LABELS.get(model, model),
                "action_valence": None,
                "action_band": None,
                "action_reasoning": "",
                "action_factors": [],
                "consequence_valence": None,
                "consequence_band": None,
                "consequence_reasoning": "",
                "consequence_factors": [],
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=max(1, len(models))) as pool:
        futures = [pool.submit(_run, model) for model in models]
        for future in as_completed(futures):
            results.append(future.result())

    order = {model: i for i, model in enumerate(models)}
    results.sort(key=lambda r: order[r["model"]])
    return results


def evaluate_random(sample_size=10, model: str = "gemini"):

    df = sample_random(sample_size)

    results = []

    for _, row in df.iterrows():

        prediction = evaluate_single(row["input_sequence"], model=model)

        results.append({
            "ID": row["ID"],
            "Scenario": row["input_sequence"],
            "Model": model,
            "Human_Action": row["Action_Valence"],
            "Human_Consequence": row["Consequence_Valence"],
            "Predicted_Action": prediction["action_valence"],
            "Predicted_Action_Reasoning": prediction.get("action_reasoning", ""),
            "Predicted_Action_Factors": "; ".join(prediction.get("action_factors", [])),
            "Predicted_Consequence": prediction["consequence_valence"],
            "Predicted_Consequence_Reasoning": prediction.get("consequence_reasoning", ""),
            "Predicted_Consequence_Factors": "; ".join(prediction.get("consequence_factors", [])),
        })

    results_df = pd.DataFrame(results)

    filepath = save_predictions(
        results_df,
        f"{model.replace('/', '_')}_random_predictions.csv"
    )

    return {
        "status": "Completed",
        "evaluated": len(results_df),
        "output_file": filepath
    }


def evaluate_dataset(model: str = "gemini"):

    df = load_dataset()

    results = []

    output_file = f"outputs/{model.replace('/', '_')}_predictions.csv"

    for index, row in df.iterrows():

        print(f"Evaluating {index+1}/{len(df)}")

        prediction = evaluate_single(row["input_sequence"], model=model)

        results.append({
            "ID": row["ID"],
            "Scenario": row["input_sequence"],
            "Model": model,
            "Human_Action": row["Action_Valence"],
            "Human_Consequence": row["Consequence_Valence"],
            "Predicted_Action": prediction["action_valence"],
            "Predicted_Action_Reasoning": prediction.get("action_reasoning", ""),
            "Predicted_Action_Factors": "; ".join(prediction.get("action_factors", [])),
            "Predicted_Consequence": prediction["consequence_valence"],
            "Predicted_Consequence_Reasoning": prediction.get("consequence_reasoning", ""),
            "Predicted_Consequence_Factors": "; ".join(prediction.get("consequence_factors", [])),
        })

        # Save after every prediction
        pd.DataFrame(results).to_csv(output_file, index=False)

    return {
        "status": "Completed",
        "evaluated": len(results),
        "output_file": output_file
    }


def evaluate_alignment(model_name: str, prompt_version: str = "current"):
    """
    Evaluate the alignment of a model's predictions with human annotations
    using Lin's Concordance Correlation Coefficient (CCC), MAE, RMSE,
    Pearson, and Spearman — merged by scenario ID (not positional order),
    so a partial or resumed export file can never silently misalign human
    and model scores against each other.
    """
    if model_name not in MODEL_CLIENTS:
        raise ValueError(f"Unsupported model: {model_name!r}. Expected one of {list(MODEL_CLIENTS.keys())}")

    df = load_dataset()

    suffix = "" if prompt_version == "current" else f"_{prompt_version}"
    output_file = f"outputs/output_{model_name.replace('/', '_')}{suffix}_entire_dataset.csv"
    print(f"Loading predictions from {output_file}")

    try:
        df_predictions = pd.read_csv(output_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Predictions file not found: {output_file}. Please run the export for this model/prompt_version first.")

    action_col = f"{model_name}_action"
    consequence_col = f"{model_name}_consequences"

    merged = df[["ID", "Action_Valence", "Consequence_Valence"]].merge(
        df_predictions[["ID", action_col, consequence_col]], on="ID", how="inner"
    ).dropna(subset=[action_col, consequence_col])

    human_action_valence = merged["Action_Valence"].tolist()
    human_consequence_valence = merged["Consequence_Valence"].tolist()
    model_action_valence = merged[action_col].tolist()
    model_consequence_valence = merged[consequence_col].tolist()

    action_results = {
        "ccc": calculate_ccc(human_action_valence, model_action_valence),
        "mae": calculate_mae(human_action_valence, model_action_valence),
        "rmse": calculate_rmse(human_action_valence, model_action_valence),
        "pearson": calculate_pearson(human_action_valence, model_action_valence),
        "spearman": calculate_spearman(human_action_valence, model_action_valence),
    }

    consequence_results = {
        "ccc": calculate_ccc(human_consequence_valence, model_consequence_valence),
        "mae": calculate_mae(human_consequence_valence, model_consequence_valence),
        "rmse": calculate_rmse(human_consequence_valence, model_consequence_valence),
        "pearson": calculate_pearson(human_consequence_valence, model_consequence_valence),
        "spearman": calculate_spearman(human_consequence_valence, model_consequence_valence),
    }

    return {
        "model": model_name,
        "prompt_version": prompt_version,
        "n_scenarios": len(merged),
        "action_results": action_results,
        "consequence_results": consequence_results,
    }


def calculate_cross_model_agreement(models: list[str] | None = None, prompt_version: str = "current") -> dict:
    """
    Cross-Model Agreement: how similarly do different LLMs evaluate the
    same moral scenarios, compared ONLY against each other.

    This is a fundamentally different question from evaluate_alignment()
    (Human-LLM Alignment: human annotations vs. one model's predictions).
    Cross-Model Agreement never reads Action_Valence / Consequence_Valence
    (the human columns) — it is purely {model A predictions} vs
    {model B predictions}, for every unique pair among the given models.

    Loads each model's raw export CSV (outputs/output_<model>[_<prompt_version>]
    _entire_dataset.csv, produced by start_dataset_export — same files
    evaluate_alignment reads, just without the human columns), inner-joins
    them all on scenario ID (never positional order), and drops any
    scenario missing a prediction from any of the selected models — no
    value is ever silently filled with zero. The same resulting aligned
    set of scenarios is used for every pairwise comparison, and its size is
    reported as n_scenarios.

    Parameters
    ----------
    models : list of model keys (subset of MODEL_CLIENTS), at least 2.
        Defaults to all 5 models (list(MODEL_CLIENTS.keys())).
    prompt_version : which export to read for every model (must be the
        same prompt_version across all models being compared).

    Returns
    -------
    dict with keys: analysis, prompt_version, models, n_scenarios,
    action (ccc_matrix, spearman_matrix), consequence (same), and
    pairwise (flat list of per-pair records) — see module docs for the
    exact shape. ccc_matrix uses Lin's Concordance Correlation Coefficient
    rather than Pearson, for the same reason it's the primary metric in
    evaluate_alignment(): unlike Pearson, CCC penalises two models whose
    scores are correlated but systematically offset in scale/location,
    not just uncorrelated ones — the same standard used to judge
    human-model alignment elsewhere in this project, so the two analyses
    are now directly, fairly comparable on the same metric.

    Raises
    ------
    ValueError - fewer than 2 models given, an unknown model key, a
        prediction file with duplicate scenario IDs, an empty/too-small
        aligned set after merging (n < 2).
    FileNotFoundError - a model's export for this prompt_version doesn't
        exist yet.
    KeyError - a model's export file is missing its expected action/
        consequence prediction columns.
    """
    models = list(models) if models else list(MODEL_CLIENTS.keys())

    if len(models) < 2:
        raise ValueError(f"Cross-model agreement requires at least 2 models, got {models!r}.")

    unknown = [m for m in models if m not in MODEL_CLIENTS]
    if unknown:
        raise ValueError(f"Unsupported model(s): {unknown}. Expected a subset of {list(MODEL_CLIENTS.keys())}")

    suffix = "" if prompt_version == "current" else f"_{prompt_version}"

    action_col = {}
    consequence_col = {}
    merged = None

    for model in models:
        output_file = f"outputs/output_{model.replace('/', '_')}{suffix}_entire_dataset.csv"
        try:
            df_pred = pd.read_csv(output_file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Predictions file not found for model {model!r}: {output_file}. "
                f"Run start_dataset_export for this model/prompt_version first."
            )

        dup_ids = df_pred["ID"][df_pred["ID"].duplicated()].tolist()
        if dup_ids:
            raise ValueError(
                f"Predictions file for model {model!r} ({output_file}) has duplicate "
                f"scenario IDs: {dup_ids[:10]}{'...' if len(dup_ids) > 10 else ''}. "
                f"Refusing to guess which row is correct."
            )

        a_col, c_col = f"{model}_action", f"{model}_consequences"
        if a_col not in df_pred.columns or c_col not in df_pred.columns:
            raise KeyError(
                f"Predictions file for model {model!r} ({output_file}) is missing "
                f"expected column(s) {a_col!r}/{c_col!r}. Found: {list(df_pred.columns)}"
            )
        action_col[model] = a_col
        consequence_col[model] = c_col

        subset = df_pred[["ID", a_col, c_col]].dropna(subset=[a_col, c_col])
        merged = subset if merged is None else merged.merge(subset, on="ID", how="inner")

    if merged is None or merged.empty:
        raise ValueError(
            f"No scenarios have predictions from every one of {models!r} for "
            f"prompt_version={prompt_version!r} — nothing to compare."
        )

    n_scenarios = len(merged)
    if n_scenarios < 2:
        raise ValueError(
            f"Only {n_scenarios} scenario(s) have predictions from every one of "
            f"{models!r} — at least 2 are required to compute a correlation."
        )

    def _matrices(col_map: dict[str, str]) -> tuple[dict, dict]:
        ccc_matrix = {m: {m2: (1.0 if m2 == m else None) for m2 in models} for m in models}
        spearman_matrix = {m: {m2: (1.0 if m2 == m else None) for m2 in models} for m in models}
        for i, m_a in enumerate(models):
            for m_b in models[i + 1:]:
                a_vals = merged[col_map[m_a]].tolist()
                b_vals = merged[col_map[m_b]].tolist()
                c = calculate_ccc(a_vals, b_vals)
                s = calculate_spearman(a_vals, b_vals)
                ccc_matrix[m_a][m_b] = ccc_matrix[m_b][m_a] = c
                spearman_matrix[m_a][m_b] = spearman_matrix[m_b][m_a] = s
        return ccc_matrix, spearman_matrix

    action_ccc, action_spearman = _matrices(action_col)
    consequence_ccc, consequence_spearman = _matrices(consequence_col)

    pairwise = []
    for i, m_a in enumerate(models):
        for m_b in models[i + 1:]:
            pairwise.append({
                "model_a": m_a,
                "model_b": m_b,
                "action_ccc": action_ccc[m_a][m_b],
                "action_spearman": action_spearman[m_a][m_b],
                "consequence_ccc": consequence_ccc[m_a][m_b],
                "consequence_spearman": consequence_spearman[m_a][m_b],
            })

    return {
        "analysis": "cross_model_agreement",
        "prompt_version": prompt_version,
        "models": models,
        "n_scenarios": n_scenarios,
        "action": {"ccc_matrix": action_ccc, "spearman_matrix": action_spearman},
        "consequence": {"ccc_matrix": consequence_ccc, "spearman_matrix": consequence_spearman},
        "pairwise": pairwise,
    }




def calculate_variant_bias(model: str, dataset: str, variant_a: str, variant_b: str) -> dict:
    """
    Demographic bias / robustness check: does a model's valence score for
    the SAME moral scenario shift when only a surface demographic marker
    (name/pronoun signalling gender or ethnicity) changes? Reads
    outputs/bias_<dataset>_<model>.csv (produced by
    tools/bias_variant_eval.py's run_bias_variant_eval), which holds one
    row per (scenario ID, variant) pair, and compares two variants of the
    same underlying scenario pairwise by scenario ID — never positionally,
    same rule as the alignment/cross-model functions above.

    This is deliberately independent of human annotations: the question is
    whether the MODEL's own score moves when demographics change, not
    whether it moves relative to a human judgment.
    """
    from tools.bias_variant_eval import bias_output_path

    output_file = bias_output_path(dataset, model)
    try:
        df = pd.read_csv(output_file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Bias predictions file not found: {output_file}. "
            f"Run tools.bias_variant_eval for model={model!r}, dataset={dataset!r} first."
        )

    for variant in (variant_a, variant_b):
        if variant not in df["variant"].unique():
            raise ValueError(
                f"Variant {variant!r} not found in {output_file}. Available: {sorted(df['variant'].unique())}"
            )

    a = df[df["variant"] == variant_a][["ID", "action_valence", "consequence_valence"]].dropna()
    b = df[df["variant"] == variant_b][["ID", "action_valence", "consequence_valence"]].dropna()

    dup_a = a["ID"][a["ID"].duplicated()].tolist()
    dup_b = b["ID"][b["ID"].duplicated()].tolist()
    if dup_a or dup_b:
        raise ValueError(f"Duplicate scenario IDs found for variant(s) in {output_file}: {dup_a or dup_b}")

    merged = a.merge(b, on="ID", how="inner", suffixes=(f"_{variant_a}", f"_{variant_b}"))
    n_scenarios = len(merged)
    if n_scenarios < 1:
        raise ValueError(
            f"No scenarios have predictions for both {variant_a!r} and {variant_b!r} — nothing to compare."
        )

    action_result = calculate_wilcoxon(
        merged[f"action_valence_{variant_a}"].tolist(), merged[f"action_valence_{variant_b}"].tolist()
    )
    consequence_result = calculate_wilcoxon(
        merged[f"consequence_valence_{variant_a}"].tolist(), merged[f"consequence_valence_{variant_b}"].tolist()
    )

    return {
        "analysis": "variant_bias",
        "model": model,
        "dataset": dataset,
        "variant_a": variant_a,
        "variant_b": variant_b,
        "n_scenarios": n_scenarios,
        "action": action_result,
        "consequence": consequence_result,
    }


def calculate_all_variant_bias(model: str, dataset: str) -> dict:
    """
    Runs calculate_variant_bias for every unique pair of DEMOGRAPHIC
    variants present in outputs/bias_<dataset>_<model>.csv (e.g. Male vs
    Female for GENDER, or Indian/European/American pairs for ETHNICITY)
    and assembles the pairwise results into one report — the bias-testing
    analogue of calculate_cross_model_agreement's pairwise list, same
    "discover pairs, reuse the single-pair function, never duplicate the
    math" shape.

    "Original" is intentionally excluded from pairwise comparison: it is
    the scenario's unmodified baseline wording, not itself a demographic
    variant, so a "vs Original" comparison doesn't answer a bias question
    the way a demographic-vs-demographic comparison does (e.g. Male vs
    Female). Only comparisons between two actual demographic variants are
    reported.
    """
    from tools.bias_variant_eval import bias_output_path

    output_file = bias_output_path(dataset, model)
    try:
        df = pd.read_csv(output_file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Bias predictions file not found: {output_file}. "
            f"Run start_bias_variant_eval for model={model!r}, dataset={dataset!r} first."
        )

    variants = sorted(v for v in df["variant"].unique().tolist() if v != "Original")
    if len(variants) < 2:
        raise ValueError(
            f"Need at least 2 demographic variants (excluding 'Original') to compare, "
            f"found {variants!r} in {output_file}"
        )

    pairwise = []
    for i, variant_a in enumerate(variants):
        for variant_b in variants[i + 1:]:
            pairwise.append(calculate_variant_bias(model, dataset, variant_a, variant_b))

    return {
        "analysis": "variant_bias_matrix",
        "model": model,
        "dataset": dataset,
        "variants": variants,
        "pairwise": pairwise,
    }
