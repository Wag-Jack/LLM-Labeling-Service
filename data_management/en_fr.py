from datasets import load_dataset
import os
import random
import pandas as pd
from pathlib import Path

def load_en_fr(amount=50, randomize=True, seed=None, pool_multiplier=20):
    # Ensure output directory exists
    output_dir = Path.cwd() / "Data" / "EuroParl"
    os.makedirs(output_dir, exist_ok=True)

    # Pull a generous superset to shuffle within; the EuroParl en-fr split has
    # millions of pairs, so a multiplier keeps load time reasonable while still
    # giving randomization meaningful coverage.
    pool_size = max(amount * pool_multiplier, amount)
    europar = load_dataset(
        "Helsinki-NLP/europarl",
        "en-fr",
        split=f"train[:{pool_size}]"
    )

    indices = list(range(len(europar)))
    if randomize:
        rng = random.Random(seed)
        rng.shuffle(indices)
    indices = indices[:amount]

    # Data dictionary to help with DataFrame creation
    data = {
        "id": [],
        "english": [],
        "french": []
    }

    for new_id, dataset_idx in enumerate(indices, start=1):
        row = europar[dataset_idx]
        data["id"].append(new_id)
        data["english"].append(row["translation"]["en"])
        data["french"].append(row["translation"]["fr"])

    # Generate DataFrame from selected data
    europar_df = pd.DataFrame(data)
    europar_df.to_csv(output_dir / "europarl_metadata.csv", index=False)

    return europar_df