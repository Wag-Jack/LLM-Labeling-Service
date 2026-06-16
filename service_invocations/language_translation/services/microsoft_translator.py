import os
from pathlib import Path
import time
import uuid

from dotenv import load_dotenv
import pandas as pd
import requests

from service_invocations.core.service_cost import record_service_call

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "language_translation" / "services"
RESULTS_FILE = "ms_trans.csv"
_TASK_NAME = "language_translation"
_SERVICE_NAME = "microsoft_translator"


def run_micro_translation(europarl_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    key = os.getenv("MICROSOFT_ACCESS_KEY")
    if not key:
        raise ValueError("MICROSOFT_ACCESS_KEY must be set in environment.")

    endpoint = os.getenv(
        "MICROSOFT_TRANSLATOR_ENDPOINT",
        "https://api.cognitive.microsofttranslator.com",
    )
    region = os.getenv("MICROSOFT_TRANSLATOR_REGION", "eastus")
    url = f"{endpoint.rstrip('/')}/translate"
    params = {"api-version": "3.0", "from": "en", "to": "fr"}
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Ocp-Apim-Subscription-Region": region,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }

    data = {
        "id": [],
        "english_input": [],
        "service_output": [],
        "latency_ms": [],
        "cost_usd": [],
    }

    for _, row in europarl_data.iterrows():
        sample_id = row["id"]
        english = row["english"]
        print(f"Microsoft Translator: ({sample_id:04d}) {english}")

        start_time = time.perf_counter()
        response = requests.post(
            url,
            params=params,
            headers=headers,
            json=[{"text": english}],
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        french = payload[0]["translations"][0]["text"] if payload else ""
        print(french)

        cost = record_service_call(
            _TASK_NAME, _SERVICE_NAME, sample_id, characters=len(english or "")
        )
        data["id"].append(f"ms_trans_{sample_id:04d}")
        data["english_input"].append(english)
        data["service_output"].append(french)
        data["latency_ms"].append(round(latency_ms, 2))
        data["cost_usd"].append(cost)

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(europarl_data, results_path=None):
    return run_micro_translation(europarl_data, results_path=results_path)
