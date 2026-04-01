from __future__ import annotations

from dataclasses import dataclass
import base64
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import boto3


MODEL_ID_ENV = "AMAZON_NOVA_MULTIMODAL_EMBEDDINGS_MODEL_ID"
DEFAULT_MODEL_ID = os.getenv(MODEL_ID_ENV)
SUPPORTED_MODALITIES = ["text", "image", "audio"]


@dataclass(frozen=True)
class EmbeddingResponse:
    embeddings: List[List[float]]
    latency_ms: float
    input_tokens: int | None
    raw_response: Dict[str, Any]


def _read_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, Path):
        return value.read_bytes()
    if isinstance(value, str):
        return Path(value).read_bytes()
    raise TypeError("input must be bytes or a filesystem path.")


def _infer_format(path_value: Any, default: str) -> str:
    if isinstance(path_value, (str, Path)):
        suffix = Path(path_value).suffix.lower().lstrip(".")
        if suffix:
            return suffix
    return default


def _load_client(region: str | None = None):
    resolved_region = (
        region
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
    )
    if not resolved_region:
        raise ValueError("AWS_REGION or AWS_DEFAULT_REGION must be set for Bedrock.")
    return boto3.client("bedrock-runtime", region_name=resolved_region)


def _extract_embeddings(payload: Dict[str, Any]) -> List[List[float]]:
    if "embedding" in payload:
        embedding = payload["embedding"]
        if isinstance(embedding, list):
            if embedding and isinstance(embedding[0], (int, float)):
                return [embedding]
            if embedding and isinstance(embedding[0], list):
                return embedding
            if not embedding:
                return []
    if "embeddings" in payload:
        embeddings = payload["embeddings"]
        if isinstance(embeddings, list):
            if embeddings and isinstance(embeddings[0], (int, float)):
                return [embeddings]
            if embeddings and isinstance(embeddings[0], list):
                return embeddings
    if "vector" in payload:
        vector = payload["vector"]
        if isinstance(vector, list):
            return [vector]
    if "vectors" in payload:
        vectors = payload["vectors"]
        if isinstance(vectors, list):
            if vectors and isinstance(vectors[0], list):
                return vectors
            if vectors and isinstance(vectors[0], (int, float)):
                return [vectors]
    return []


def _extract_input_tokens(payload: Dict[str, Any]) -> int | None:
    usage = payload.get("usage") or payload.get("usageMetadata") or {}
    if isinstance(usage, dict):
        for key in ("inputTokens", "input_tokens", "tokenCount", "tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                return value
    return None


def _build_payload(inputs: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}

    text_input = inputs.get("text")
    if text_input is not None:
        payload["inputText"] = text_input

    image_input = inputs.get("image")
    if image_input is not None:
        image_bytes = _read_bytes(image_input)
        image_format = inputs.get("image_format") or _infer_format(image_input, "png")
        payload["inputImage"] = {
            "bytes": base64.b64encode(image_bytes).decode("utf-8"),
            "format": image_format,
        }

    audio_input = inputs.get("audio")
    if audio_input is not None:
        audio_bytes = _read_bytes(audio_input)
        audio_format = inputs.get("audio_format") or _infer_format(audio_input, "wav")
        payload["inputAudio"] = {
            "bytes": base64.b64encode(audio_bytes).decode("utf-8"),
            "format": audio_format,
        }

    return payload


def embed(
    inputs: Dict[str, Any] | None = None,
    *,
    model_id: str | None = None,
    region: str | None = None,
    payload_override: Dict[str, Any] | None = None,
) -> EmbeddingResponse:
    resolved_model_id = model_id or DEFAULT_MODEL_ID
    if not resolved_model_id:
        raise ValueError(f"{MODEL_ID_ENV} must be set or model_id provided.")

    payload = payload_override or _build_payload(inputs or {})
    if not payload:
        raise ValueError("At least one of text, image, or audio inputs is required.")

    client = _load_client(region=region)

    start_time = time.perf_counter()
    response = client.invoke_model(
        modelId=resolved_model_id,
        body=json.dumps(payload),
    )
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    body_bytes = response.get("body").read() if response.get("body") is not None else b"{}"
    payload_response = json.loads(body_bytes.decode("utf-8"))
    embeddings = _extract_embeddings(payload_response)
    input_tokens = _extract_input_tokens(payload_response)

    return EmbeddingResponse(
        embeddings=embeddings,
        latency_ms=round(latency_ms, 2),
        input_tokens=input_tokens,
        raw_response=payload_response,
    )


def generate(*args, **kwargs):
    raise RuntimeError(
        "Amazon Nova Multimodal Embeddings provides embeddings only. "
        "Use embed(...) instead of generate(...)."
    )

