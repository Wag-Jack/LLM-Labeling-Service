import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google.oauth2 import service_account
from google.cloud import speech
import pandas as pd

from service_invocations.core.service_cost import audio_minutes, record_service_call

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "speech_recognition" / "services"
RESULTS_FILE = "gc_stt.csv"
_TASK_NAME = "speech_recognition"
_SERVICE_NAME = "google_cloud_stt"


def _resolve_credentials_path() -> Path:
    env_path = os.getenv("GOOGLE_STT_CREDENTIALS") or os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
    if env_path:
        return Path(env_path)
    return Path.cwd() / "credentials" / "speech_recognition" / "llm-as-a-judge_gc.json"


def _combine_response(response) -> str:
    parts = []
    for result in response.results:
        if result.alternatives:
            parts.append(result.alternatives[0].transcript)
    return " ".join(part.strip() for part in parts if part).strip()


def run_gc_stt(edacc_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    credentials = service_account.Credentials.from_service_account_file(
        _resolve_credentials_path()
    )
    client = speech.SpeechClient(credentials=credentials)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
    )

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
        print(f"Google Cloud STT: {audio_file}")

        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        audio = speech.RecognitionAudio(content=audio_bytes)

        start_time = time.perf_counter()
        response = client.recognize(config=config, audio=audio)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        transcript = _combine_response(response)
        print(f"  -> {transcript}", flush=True)

        cost = record_service_call(
            _TASK_NAME, _SERVICE_NAME, sample_id, minutes=audio_minutes(row)
        )
        data["id"].append(f"gc_stt_{sample_id:04d}")
        data["service_output"].append(transcript)
        data["latency_ms"].append(round(latency_ms, 2))
        data["cost_usd"].append(cost)
        data["wav_file"].append(audio_file)

    df = pd.DataFrame(data, columns=["id", "service_output", "latency_ms", "cost_usd", "wav_file"])
    df.to_csv(results_path, index=False)
    return df


def run(edacc_data, results_path=None):
    return run_gc_stt(edacc_data, results_path=results_path)
