import os
import time as perf_time
from pathlib import Path

import boto3
from dotenv import load_dotenv
import pandas as pd

from service_invocations.emotion_detection.services._shared import (
    format_service_output,
    label_to_name,
    normalize_emotions,
    pick_top_emotion,
)

load_dotenv()

_RESULTS_DIR = (
    Path.cwd()
    / "service_invocations"
    / "results"
    / "emotion_detection"
    / "services"
)
RESULTS_FILE = "aws_rekognition.csv"

_EMOTION_MAPPING = {
    "angry": "anger",
    "calm": "neutral",
    "confused": "neutral",
    "disgusted": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprised": "surprise",
}


def _extract_emotions(response: dict) -> dict[str, float | None]:
    faces = response.get("FaceDetails", [])
    if not faces:
        return {}
    emotions = faces[0].get("Emotions", [])
    scores = {}
    for entry in emotions:
        raw_type = str(entry.get("Type", "")).lower()
        score = entry.get("Confidence")
        if score is None:
            scores[raw_type] = None
        else:
            scores[raw_type] = float(score) / 100.0
    return scores


def run_aws_rekognition(vea_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    client = boto3.client(
        "rekognition",
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    data = {
        "id": [],
        "image_file": [],
        "label": [],
        "label_name": [],
        "service_output": [],
        "top_emotion": [],
        "latency_ms": [],
    }

    for _, row in vea_data.iterrows():
        image_file = row["image"]
        sample_id = int(row["id"])
        label = row.get("label")
        print(f"AWS Rekognition: {image_file}")

        try:
            with open(image_file, "rb") as f:
                image_bytes = f.read()

            start_time = perf_time.perf_counter()
            response = client.detect_faces(
                Image={"Bytes": image_bytes},
                Attributes=["EMOTIONS"],
            )
            latency_ms = (perf_time.perf_counter() - start_time) * 1000.0
            raw_scores = _extract_emotions(response)
            normalized = normalize_emotions(raw_scores, mapping=_EMOTION_MAPPING)
            top_emotion = pick_top_emotion(normalized)
            output = format_service_output(response, normalized)
        except Exception as exc:  # noqa: BLE001
            latency_ms = None
            normalized = {}
            top_emotion = None
            output = format_service_output({"error": str(exc)}, normalized)

        data["id"].append(f"aws_rekognition_{sample_id:04d}")
        data["image_file"].append(image_file)
        data["label"].append(label)
        data["label_name"].append(label_to_name(label))
        data["service_output"].append(output)
        data["top_emotion"].append(top_emotion)
        data["latency_ms"].append(None if latency_ms is None else round(latency_ms, 2))

    data["llm_judge_score"] = [0.0 for _ in data["id"]]

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(vea_data):
    return run_aws_rekognition(vea_data)
