import os
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv
import pandas as pd
import requests

from service_invocations.core.service_cost import audio_minutes, record_service_call

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "speech_recognition" / "services"
RESULTS_FILE = "aws_stt.csv"
_TASK_NAME = "speech_recognition"
_SERVICE_NAME = "aws_transcribe"

S3_URI = os.getenv("AWS_S3_AUDIO_URI", "s3://llm-as-a-judge-edacc-storage")
S3_AUDIO_PREFIX = os.getenv("AWS_S3_PREFIX", "edacc/audio")


def _start_transcription_job(sample_id: int, transcribe) -> str:
    job_name = f"{int(sample_id):04d}_job-{int(time.time())}"
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": f"{S3_URI}/{S3_AUDIO_PREFIX}/{int(sample_id):04d}.wav"},
        MediaFormat="wav",
        LanguageCode="en-US",
    )
    return job_name


def _wait_for_job(job_name: str, transcribe) -> str | None:
    while True:
        response = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        status = response["TranscriptionJob"]["TranscriptionJobStatus"]
        if status == "COMPLETED":
            uri = response["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            transcribe.delete_transcription_job(TranscriptionJobName=job_name)
            return uri
        if status == "FAILED":
            return None
        time.sleep(1)


def _retrieve_transcript(transcript_uri: str | None) -> str:
    if not transcript_uri:
        return ""
    response = requests.get(transcript_uri, timeout=60).json()
    return response.get("results", {}).get("transcripts", [{}])[0].get("transcript", "")


def run_aws_transcribe(edacc_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    transcribe = boto3.client(
        "transcribe",
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
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
        print(f"AWS Transcribe STT: {audio_file}")

        start_time = time.perf_counter()
        job_name = _start_transcription_job(sample_id, transcribe)
        transcript_uri = _wait_for_job(job_name, transcribe)
        transcript = _retrieve_transcript(transcript_uri)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        print(transcript)

        cost = record_service_call(
            _TASK_NAME, _SERVICE_NAME, sample_id, minutes=audio_minutes(row)
        )
        data["id"].append(f"aws_stt_{sample_id:04d}")
        data["service_output"].append(transcript)
        data["latency_ms"].append(round(latency_ms, 2))
        data["cost_usd"].append(cost)
        data["wav_file"].append(audio_file)

    df = pd.DataFrame(data, columns=["id", "service_output", "latency_ms", "cost_usd", "wav_file"])
    df.to_csv(results_path, index=False)
    return df


def run(edacc_data, results_path=None):
    return run_aws_transcribe(edacc_data, results_path=results_path)
