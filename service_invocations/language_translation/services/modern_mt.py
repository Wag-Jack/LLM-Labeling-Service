import os
from pathlib import Path

from dotenv import load_dotenv
from modernmt import ModernMT
import pandas as pd

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "language_translation" / "services"
RESULTS_FILE = "modern_mt_trans.csv"


def run_modern_mt(europarl_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    api_key = os.getenv("MODERNMT_API_KEY")
    if not api_key:
        raise ValueError("MODERNMT_API_KEY must be set in environment.")

    mmt = ModernMT(api_key)

    data = {
        "id": [],
        "english_input": [],
        "service_output": [],
    }

    for _, row in europarl_data.iterrows():
        sample_id = row["id"]
        english = row["english"]
        print(f"ModernMT Translate: ({sample_id:04d}) {english}")

        translation = mmt.translate("en", "fr", english)
        french = translation.translation or ""
        print(french)

        data["id"].append(f"modern_mt_trans_{sample_id:04d}")
        data["english_input"].append(english)
        data["service_output"].append(french)

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(europarl_data, results_path=None):
    return run_modern_mt(europarl_data, results_path=results_path)
