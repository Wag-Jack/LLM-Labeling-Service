import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import requests

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "language_translation" / "services"
RESULTS_FILE = "modern_mt_trans.csv"

# ModernMT REST API:
# https://www.modernmt.com/api/#translate
# Authentication is via the MMT-ApiKey header; translation can be requested
# via GET (?q=&source=&target=) or POST with a JSON body. We use POST so
# long inputs and unicode characters don't need URL encoding.
_DEFAULT_ENDPOINT = "https://api.modernmt.com/translate"


def _extract_translation(payload: dict) -> str:
    data = payload.get("data")
    if isinstance(data, dict):
        return data.get("translation", "") or ""
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return first.get("translation", "") or ""
    return ""


def run_modern_mt(europarl_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    api_key = os.getenv("MODERNMT_API_KEY")
    if not api_key:
        raise ValueError("MODERNMT_API_KEY must be set in environment.")

    endpoint = os.getenv("MODERNMT_ENDPOINT", _DEFAULT_ENDPOINT)
    headers = {
        "MMT-ApiKey": api_key,
        "Content-Type": "application/json",
    }

    data = {
        "id": [],
        "english_input": [],
        "service_output": [],
    }

    for _, row in europarl_data.iterrows():
        sample_id = row["id"]
        english = row["english"]
        print(f"ModernMT Translate: ({sample_id:04d}) {english}")

        response = requests.post(
            endpoint,
            headers=headers,
            json={"q": english, "source": "en", "target": "fr"},
            timeout=30,
        )
        response.raise_for_status()
        french = _extract_translation(response.json())
        print(french)

        data["id"].append(f"modern_mt_trans_{sample_id:04d}")
        data["english_input"].append(english)
        data["service_output"].append(french)

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(europarl_data):
    return run_modern_mt(europarl_data)
