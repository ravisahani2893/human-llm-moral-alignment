import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from app.dataset import load_dataset, sample_random, save_predictions
from app.interpret import valence_label
from app.metric import calculate_ccc, calculate_mae, calculate_pearson, calculate_rmse, calculate_spearman
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


