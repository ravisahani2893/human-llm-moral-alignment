import pandas as pd
from pathlib import Path

DATASET_PATH = "data/processed/moralalign_dataset.csv"


def load_dataset():
    """
    Load the complete dataset.
    """
    return pd.read_csv(DATASET_PATH)


def sample_random(sample_size=10, random_state=42):
    """
    Return random scenarios.
    """
    df = load_dataset()
    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def save_predictions(df, filename):
    """
    Save predictions to outputs folder.
    """
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / filename
    df.to_csv(filepath, index=False)

    return str(filepath)