import json
import pandas as pd
from app.dataset import load_dataset, sample_random, save_predictions
from app.prompts import build_prompt
from app.gemini_client import ask_gemini


def evaluate_single(scenario: str):

    prompt = build_prompt(scenario)

    response = ask_gemini(prompt)

    response = response.strip()
    response = response.replace("```json", "")
    response = response.replace("```", "")

    return json.loads(response)


def evaluate_random(sample_size=10):

    df = sample_random(sample_size)

    results = []

    for _, row in df.iterrows():

        prediction = evaluate_single(row["input_sequence"])

        results.append({
            "ID": row["ID"],
            "Scenario": row["input_sequence"],
            "Human_Action": row["Action_Valence"],
            "Human_Consequence": row["Consequence_Valence"],
            "Gemini_Action": prediction["action_valence"],
            "Gemini_Consequence": prediction["consequence_valence"]
        })

    results_df = pd.DataFrame(results)

    filepath = save_predictions(
        results_df,
        "gemini_random_predictions.csv"
    )

    return {
        "status": "Completed",
        "evaluated": len(results_df),
        "output_file": filepath
    }


def evaluate_dataset():

    df = load_dataset()

    results = []

    for index, row in df.iterrows():

        print(f"Evaluating {index+1}/{len(df)}")

        prediction = evaluate_single(row["input_sequence"])

        results.append({
            "ID": row["ID"],
            "Scenario": row["input_sequence"],
            "Human_Action": row["Action_Valence"],
            "Human_Consequence": row["Consequence_Valence"],
            "Gemini_Action": prediction["action_valence"],
            "Gemini_Consequence": prediction["consequence_valence"]
        })

        # Save after every prediction
        pd.DataFrame(results).to_csv(
            "outputs/gemini_predictions.csv",
            index=False
        )

    return {
        "status": "Completed",
        "evaluated": len(results),
        "output_file": "outputs/gemini_predictions.csv"
    }