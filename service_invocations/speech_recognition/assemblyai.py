import assemblyai as aai
from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path
import time

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "speech_recognition"  # Task-scoped outputs.

def run_assemblyai(edacc_data):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Establish connection with AsssemblyAI API
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

    # Data dictionary to hold results
    data = {
        "id": [],
        "wav_file": [],
        "service_output": [],
        "latency_ms": [],
    }

    for _, row in edacc_data.iterrows():
        # Run the service on the selected file
        audio_file = row["audio"]
        print(f"AssemblyAI STT: {audio_file}")

        # Upload the audio file to AssemblyAI
        transcriber = aai.Transcriber()
        start_time = time.perf_counter()
        transcript = transcriber.transcribe(audio_file)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        data["id"].append(f"aa_stt_{row['id']:04d}")
        data["wav_file"].append(row["audio"])
        data["latency_ms"].append(round(latency_ms, 2))
        print(transcript.text)

        data["service_output"].append(transcript.text)

    # Add in blank column for LLM judge score
    data["llm_judge_score"] = [0.0 for r in data["id"]]

    # Convert into DataFrame and save to CSV
    aa_stt_df = pd.DataFrame(data)
    aa_stt_df.to_csv(_RESULTS_DIR / "aa_stt.csv", index=False)
    return aa_stt_df
