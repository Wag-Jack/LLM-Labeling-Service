import os
from pathlib import Path

import deepl
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "language_translation" / "services"
RESULTS_FILE = "deepl_trans.csv"


def run_deepl_translation(europarl_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        raise ValueError("DEEPL_API_KEY must be set in environment.")

    translator = deepl.Translator(api_key)

    data = {
        "id": [],
        "english_input": [],
        "service_output": [],
    }

    for _, row in europarl_data.iterrows():
        sample_id = row["id"]
        english = row["english"]
        print(f"DeepL Translate: ({sample_id:04d}) {english}")

        result = translator.translate_text(english, source_lang="EN", target_lang="FR")
        french = result.text if result is not None else ""
        print(french)

        data["id"].append(f"deepl_trans_{sample_id:04d}")
        data["english_input"].append(english)
        data["service_output"].append(french)

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(europarl_data):
    return run_deepl_translation(europarl_data)
