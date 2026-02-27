import asyncio
from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path

from speechmatics.batch import AsyncClient, JobConfig, JobType, TranscriptionConfig

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "speech_recognition"  # Task-scoped outputs.
# Runs in the main project environment (no provider-specific venv required).


def _extract_transcript(result) -> str:
    # speechmatics-batch returns a result object with transcript_text; fall back to dict access.
    if hasattr(result, "transcript_text"):
        return result.transcript_text or ""
    if isinstance(result, dict):
        return result.get("transcript_text") or ""
    return ""


async def _transcribe_all(edacc_data) -> list[str]:
    config = JobConfig(
        type=JobType.TRANSCRIPTION,
        transcription_config=TranscriptionConfig(language="en"),
    )

    transcripts: list[str] = []
    async with AsyncClient() as client:
        for _, row in edacc_data.iterrows():
            audio_file = row["audio"]
            print(f"Speechmatics STT: {audio_file}")
            result = await client.transcribe(audio_file, config=config)
            transcript = _extract_transcript(result)
            transcripts.append(transcript)
            print(transcript)
    return transcripts


def run_speechmatics_stt(edacc_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / "speechmatics_stt.csv"

    api_key = os.getenv("SPEECHMATICS_API_KEY")
    if not api_key:
        raise ValueError("SPEECHMATICS_API_KEY must be set in environment.")

    transcripts = asyncio.run(_transcribe_all(edacc_data))

    data = {
        "id": [],
        "wav_file": [],
        "service_output": [],
    }

    for (_, row), transcript in zip(edacc_data.iterrows(), transcripts):
        sample_id = row["id"]
        audio_file = row["audio"]
        data["id"].append(f"speechmatics_stt_{sample_id:04d}")
        data["wav_file"].append(audio_file)
        data["service_output"].append(transcript)

    data["llm_judge_score"] = [0.0 for _ in data["id"]]

    sm_df = pd.DataFrame(data)
    sm_df.to_csv(results_path, index=False)
    return sm_df
