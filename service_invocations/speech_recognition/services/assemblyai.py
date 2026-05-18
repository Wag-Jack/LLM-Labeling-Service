import os
import time
from pathlib import Path

import assemblyai as aai
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "speech_recognition" / "services"
RESULTS_FILE = "aa_stt.csv"


def run_assemblyai(edacc_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise ValueError("ASSEMBLYAI_API_KEY must be set in environment.")
    aai.settings.api_key = api_key
    transcriber = aai.Transcriber()

    data = {
        "id": [],
        "service_output": [],
        "latency_ms": [],
        "wav_file": [],
    }

    for _, row in edacc_data.iterrows():
        audio_file = row["audio"]
        sample_id = row["id"]
        print(f"AssemblyAI STT: {audio_file}")

        start_time = time.perf_counter()
        transcript = transcriber.transcribe(audio_file)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        text = transcript.text or ""
        print(text)

        data["id"].append(f"aa_stt_{sample_id:04d}")
        data["service_output"].append(text)
        data["latency_ms"].append(round(latency_ms, 2))
        data["wav_file"].append(audio_file)

    data["llm_judge_score"] = [0.0 for _ in data["id"]]

    df = pd.DataFrame(data, columns=["id", "service_output", "latency_ms", "llm_judge_score", "wav_file"])
    df.to_csv(results_path, index=False)
    return df


def run(edacc_data):
    return run_assemblyai(edacc_data)
