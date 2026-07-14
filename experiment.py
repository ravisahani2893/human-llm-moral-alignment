import json
import pandas as pd

from app.prompts import build_prompt
from clients.gemini_client import ask_gemini
import json

# Load dataset
df = pd.read_csv("data/processed/moralalign_dataset.csv")

# Random sample
sample_df = df.sample(n=1, random_state=42).reset_index(drop=True)


def parse_json(response: str):

    response = response.strip()
    print("Raw response:", response)

    response = response.replace("```json", "")
    response = response.replace("```", "")

    return json.loads(response)

results = []

for _, row in sample_df.iterrows():
    print(f"Processing ID: {row['input_sequence']}")
    prompt = build_prompt(row["input_sequence"])
    response = ask_gemini(prompt)


prediction = parse_json(response)




results.append({
    "ID": row["ID"],
    "Scenario": row["input_sequence"],
    "Human_Action": row["Action_Valence"],
    "Gemini_Action": prediction["action_valence"],
    "Human_Consequence": row["Consequence_Valence"],
    "Gemini_Consequence": prediction["consequence_valence"]
})

results_df = pd.DataFrame(results)

results_df.head()
results

