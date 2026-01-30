from datasets import load_dataset
import os
import pandas as pd
from pathlib import Path

def load_en_fr(amount=50):
    # Ensure output directory exists
    output_dir = Path.cwd() / "Data" / "EuroParl"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset from HuggingFace
    europar = load_dataset(
        "Helsinki-NLP/europarl",
        "en-fr",
        split=f"train[:{amount}]"
    )

    # Data dictionary to help with DataFrame creation
    data = {
        "id": [],
        "english": [],
        "french": []
    }

    # Convert dictionary rows into ones to put into a DataFrame
    i = 1
    for row in europar:
        data["id"].append(i)
        data["english"].append(row["translation"]["en"])
        data["french"].append(row["translation"]["fr"])

        i += 1

    # Generate DataFrame from selected data
    europar_df = pd.DataFrame(data)
    europar_df.to_csv(output_dir / "europarl_metadata.csv", index=False)

    return europar_df