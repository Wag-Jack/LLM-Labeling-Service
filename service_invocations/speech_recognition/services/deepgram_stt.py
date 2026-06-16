import os
import time
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import requests

from service_invocations.core.service_cost import audio_minutes, record_service_call

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "speech_recognition" / "services"
RESULTS_FILE = "deepgram_stt.csv"
_TASK_NAME = "speech_recognition"
_SERVICE_NAME = "deepgram_stt"


def _extract_transcript(payload: dict) -> str:
    return (
        payload.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("transcript", "")
    )


def run_deepgram_stt(edacc_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise ValueError("DEEPGRAM_API_KEY must be set in environment.")

    url = "https://api.deepgram.com/v1/listen"
    json_headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    bytes_headers = {"Authorization": f"Token {api_key}"}

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
        print(f"Deepgram STT: {audio_file}")

        start_time = time.perf_counter()
        if isinstance(audio_file, str) and audio_file.startswith(("http://", "https://")):
            response = requests.post(url, json={"url": audio_file}, headers=json_headers, timeout=60)
        else:
            with open(audio_file, "rb") as f:
                response = requests.post(url, headers=bytes_headers, data=f, timeout=120)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        response.raise_for_status()
        transcript = _extract_transcript(response.json())
        print(transcript)

        cost = record_service_call(
            _TASK_NAME, _SERVICE_NAME, sample_id, minutes=audio_minutes(row)
        )
        data["id"].append(f"deepgram_stt_{sample_id:04d}")
        data["service_output"].append(transcript)
        data["latency_ms"].append(round(latency_ms, 2))
        data["cost_usd"].append(cost)
        data["wav_file"].append(audio_file)

    df = pd.DataFrame(data, columns=["id", "service_output", "latency_ms", "cost_usd", "wav_file"])
    df.to_csv(results_path, index=False)
    return df


def run(edacc_data, results_path=None):
    return run_deepgram_stt(edacc_data, results_path=results_path)
