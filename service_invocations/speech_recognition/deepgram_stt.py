from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path
import requests

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "speech_recognition"  # Task-scoped outputs.
# Uses the Deepgram REST API directly (no SDK dependency).


def run_deepgram_stt(edacc_data):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise ValueError("DEEPGRAM_API_KEY must be set in environment.")

    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "id": [],
        "wav_file": [],
        "service_output": [],
    }

    for _, row in edacc_data.iterrows():
        audio_file = row["audio"]
        sample_id = row["id"]
        print(f"Deepgram STT: {audio_file}")

        # Provide a file path URL if available, otherwise fall back to file upload.
        if isinstance(audio_file, str) and audio_file.startswith(("http://", "https://")):
            payload = {"url": audio_file}
            response = requests.post(url, json=payload, headers=headers, timeout=60)
        else:
            with open(audio_file, "rb") as f:
                response = requests.post(
                    url,
                    headers={"Authorization": f"Token {api_key}"},
                    data=f,
                    timeout=120,
                )

        response.raise_for_status()
        result = response.json()
        transcript = (
            result.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
        )

        data["id"].append(f"deepgram_stt_{sample_id:04d}")
        data["wav_file"].append(audio_file)
        data["service_output"].append(transcript)
        print(transcript)

    data["llm_judge_score"] = [0.0 for _ in data["id"]]

    deepgram_df = pd.DataFrame(data)
    deepgram_df.to_csv(_RESULTS_DIR / "deepgram_stt.csv", index=False)
    return deepgram_df
