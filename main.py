import pandas as pd

from app.prompts import build_prompt
from clients.gemini_client import ask_gemini

df = pd.read_csv("data/processed/moralalign_dataset.csv")

scenario = df.iloc[0]["input_sequence"]

print(f"Scenario: {scenario}")

prompt = build_prompt(scenario)

response = ask_gemini(prompt)

print(response)