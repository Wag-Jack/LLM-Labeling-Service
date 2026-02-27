from dotenv import load_dotenv
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1
import os
import pandas as pd
from pathlib import Path
import time

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "speech_recognition"  # Task-scoped outputs.
# Runs in the main project environment (no provider-specific venv required).


def _extract_transcript(result: dict) -> str:
    # IBM returns a list of results with alternatives; take the top transcript from each.
    combined = []
    for item in result.get("results", []):
        alternatives = item.get("alternatives", [])
        if alternatives:
            combined.append(alternatives[0].get("transcript", ""))
    return " ".join(part.strip() for part in combined if part).strip()


def run_ibm_watson_stt(edacc_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Allow an explicit output path for orchestrated runs.
    if results_path is None:
        results_path = _RESULTS_DIR / "ibm_stt.csv"

    api_key = os.getenv("IBM_WATSON_API_KEY")
    service_url = os.getenv("IBM_WATSON_URL")
    if not api_key or not service_url:
        raise ValueError("IBM_WATSON_API_KEY and IBM_WATSON_URL must be set in environment.")

    authenticator = IAMAuthenticator(api_key)
    stt = SpeechToTextV1(authenticator=authenticator)
    stt.set_service_url(service_url)

    data = {
        "id": [],
        "wav_file": [],
        "service_output": [],
        "latency_ms": [],
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
            model="en-US_BroadbandModel",
        ).get_result()
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        transcript = _extract_transcript(response)
        data["id"].append(f"ibm_stt_{sample_id:04d}")
        data["wav_file"].append(audio_file)
        data["service_output"].append(transcript)
        data["latency_ms"].append(round(latency_ms, 2))
        print(transcript)

    data["llm_judge_score"] = [0.0 for _ in data["id"]]

    ibm_df = pd.DataFrame(data)
    ibm_df.to_csv(results_path, index=False)
    return ibm_df
