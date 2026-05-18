import asyncio
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

from speechmatics.batch import AsyncClient, JobConfig, JobType, TranscriptionConfig

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "speech_recognition" / "services"
RESULTS_FILE = "speechmatics_stt.csv"

_SPEAKER_PREFIX_RE = re.compile(r"^SPEAKER\s+[^:]+:\s*", re.IGNORECASE | re.MULTILINE)


def _clean_transcript(text: str) -> str:
    if not text:
        return ""
    return _SPEAKER_PREFIX_RE.sub("", text).strip()


def _extract_transcript(result) -> str:
    if hasattr(result, "transcript_text"):
        return _clean_transcript(result.transcript_text or "")
    if isinstance(result, dict):
        return _clean_transcript(result.get("transcript_text") or "")
    return ""


async def _transcribe_all(edacc_data) -> list[tuple[str, float]]:
    config = JobConfig(
        type=JobType.TRANSCRIPTION,
        transcription_config=TranscriptionConfig(language="en"),
    )

    transcripts: list[tuple[str, float]] = []
    async with AsyncClient() as client:
        for _, row in edacc_data.iterrows():
            audio_file = row["audio"]
            print(f"Speechmatics STT: {audio_file}")
            start_time = time.perf_counter()
            result = await client.transcribe(audio_file, config=config)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            transcript = _extract_transcript(result)
            transcripts.append((transcript, latency_ms))
            print(transcript)
    return transcripts


def run_speechmatics_stt(edacc_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    if not os.getenv("SPEECHMATICS_API_KEY"):
        raise ValueError("SPEECHMATICS_API_KEY must be set in environment.")

    transcripts = asyncio.run(_transcribe_all(edacc_data))

    data = {
        "id": [],
        "service_output": [],
        "latency_ms": [],
        "wav_file": [],
    }

    for (_, row), (transcript, latency_ms) in zip(edacc_data.iterrows(), transcripts):
        sample_id = row["id"]
        audio_file = row["audio"]
        data["id"].append(f"speechmatics_stt_{sample_id:04d}")
        data["service_output"].append(transcript)
        data["latency_ms"].append(round(latency_ms, 2))
        data["wav_file"].append(audio_file)

    data["llm_judge_score"] = [0.0 for _ in data["id"]]

    df = pd.DataFrame(data, columns=["id", "service_output", "latency_ms", "llm_judge_score", "wav_file"])
    df.to_csv(results_path, index=False)
    return df


def run(edacc_data):
    return run_speechmatics_stt(edacc_data)
