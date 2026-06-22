from html import unescape
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google.oauth2 import service_account
from google.cloud import translate
import pandas as pd

from service_invocations.core.service_cost import record_service_call

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "language_translation" / "services"
RESULTS_FILE = "gc_trans.csv"
_TASK_NAME = "language_translation"
_SERVICE_NAME = "google_cloud_translation"


def _resolve_credentials_path() -> Path:
    env_path = os.getenv("GOOGLE_TRANSLATE_CREDENTIALS") or os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
    if env_path:
        return Path(env_path)
    return Path.cwd() / "credentials" / "speech_recognition" / "llm-as-a-judge_gc.json"


def run_gc_translation(europarl_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    credentials = service_account.Credentials.from_service_account_file(
        _resolve_credentials_path()
    )
    project_id = os.getenv("GOOGLE_TRANSLATE_PROJECT", "llm-as-a-judge-485501")
    client = translate.TranslationServiceClient(credentials=credentials)

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
        print(f"Google Cloud Translate: ({sample_id:04d}) {english}")

        start_time = time.perf_counter()
        response = client.translate_text(
            request={
                "contents": [english],
                "parent": f"projects/{project_id}",
                "target_language_code": "fr",
                "source_language_code": "en",
            }
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        french = unescape(response.translations[0].translated_text)
        print(f"  -> {french}", flush=True)

        cost = record_service_call(
            _TASK_NAME, _SERVICE_NAME, sample_id, characters=len(english or "")
        )
        data["id"].append(f"gc_trans_{sample_id:04d}")
        data["english_input"].append(english)
        data["service_output"].append(french)
        data["latency_ms"].append(round(latency_ms, 2))
        data["cost_usd"].append(cost)

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(europarl_data, results_path=None):
    return run_gc_translation(europarl_data, results_path=results_path)
