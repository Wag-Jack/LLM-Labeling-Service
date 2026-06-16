import os
import time
from pathlib import Path

from dotenv import load_dotenv
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1
import pandas as pd

from service_invocations.core.service_cost import audio_minutes, record_service_call

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "speech_recognition" / "services"
RESULTS_FILE = "ibm_stt.csv"
_TASK_NAME = "speech_recognition"
_SERVICE_NAME = "ibm_watson_stt"


def _extract_transcript(result: dict) -> str:
    parts = []
    for item in result.get("results", []):
        alternatives = item.get("alternatives", [])
        if alternatives:
            parts.append(alternatives[0].get("transcript", ""))
    return " ".join(part.strip() for part in parts if part).strip()


def run_ibm_watson_stt(edacc_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    api_key = os.getenv("IBM_WATSON_API_KEY")
    service_url = os.getenv("IBM_WATSON_URL")
    if not api_key or not service_url:
        raise ValueError("IBM_WATSON_API_KEY and IBM_WATSON_URL must be set in environment.")

    authenticator = IAMAuthenticator(api_key)
    stt = SpeechToTextV1(authenticator=authenticator)
    stt.set_service_url(service_url)
    model = os.getenv("IBM_WATSON_MODEL", "en-US_BroadbandModel")

    data = {
        "id": [],
        "service_output": [],
        "latency_ms": [],
        "cost_usd": [],
        "wav_file": [],
    }

    for _, row in edacc_data.iterrows():
        audio_file = row["audio"]
        sample_id = row["id"]
        print(f"IBM Watson STT: {audio_file}")

        with open(audio_file, "rb") as f:
            audio_bytes = f.read()

        start_time = time.perf_counter()
        response = stt.recognize(
            audio=audio_bytes,
            content_type="audio/wav",
            model=model,
        ).get_result()
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        transcript = _extract_transcript(response)
        print(transcript)

        cost = record_service_call(
            _TASK_NAME, _SERVICE_NAME, sample_id, minutes=audio_minutes(row)
        )
        data["id"].append(f"ibm_stt_{sample_id:04d}")
        data["service_output"].append(transcript)
        data["latency_ms"].append(round(latency_ms, 2))
        data["cost_usd"].append(cost)
        data["wav_file"].append(audio_file)

    df = pd.DataFrame(data, columns=["id", "service_output", "latency_ms", "cost_usd", "wav_file"])
    df.to_csv(results_path, index=False)
    return df


def run(edacc_data, results_path=None):
    return run_ibm_watson_stt(edacc_data, results_path=results_path)
