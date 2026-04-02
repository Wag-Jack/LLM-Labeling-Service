from __future__ import annotations

import base64
import os
from pathlib import Path
import time
from typing import Any, Dict, List

from service_invocations.core.llm_adapters import LLMResponse
from service_invocations.models.common import infer_modalities


MODEL_ID = os.getenv("QWEN_2_5_OMNI_7B_MODEL_ID", "Qwen/Qwen2.5-Omni-7B")
SUPPORTED_MODALITIES = ["text", "audio", "image"]


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


def _to_data_url(kind: str, payload: bytes, fmt: str) -> str:
    encoded = base64.b64encode(payload).decode("utf-8")
    return f"data:{kind}/{fmt};base64,{encoded}"


def _build_messages(prompt: str, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

    text_input = inputs.get("text")
    if text_input:
        content.append({"type": "text", "text": text_input})

    image_input = inputs.get("image")
    if image_input is not None:
        image_bytes = _read_bytes(image_input)
        image_format = inputs.get("image_format") or _infer_format(image_input, "png")
        content.append({
            "type": "image_url",
            "image_url": {"url": _to_data_url("image", image_bytes, image_format)},
        })

    audio_input = inputs.get("audio")
    if audio_input is not None:
        audio_bytes = _read_bytes(audio_input)
        audio_format = inputs.get("audio_format") or _infer_format(audio_input, "wav")
        content.append({
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(audio_bytes).decode("utf-8"),
                "format": audio_format,
            },
        })

    return [{"role": "user", "content": content}]


def _extract_content(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    message = getattr(response, "choices", None)
    if message:
        choice = message[0]
        msg = getattr(choice, "message", None) or getattr(choice, "delta", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                return content
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            msg = choices[0].get("message") or choices[0].get("delta") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
    return str(response)


def _extract_tokens(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        return input_tokens, output_tokens
    return None, None


def generate(
    prompt: str,
    inputs: Dict[str, Any] | None = None,
    modalities: List[str] | None = None,
) -> LLMResponse:
    payload_inputs = inputs or {}
    _ = modalities or infer_modalities(payload_inputs)

    try:
        from huggingface_hub import InferenceClient
    except Exception as exc:  # pragma: no cover - import guard for runtime environments
        raise RuntimeError(
            "huggingface_hub is required to use Qwen2.5-Omni. "
            "Install or enable it before running."
        ) from exc

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    client = InferenceClient(model=MODEL_ID, token=token)

    messages = _build_messages(prompt, payload_inputs)
    start_time = time.perf_counter()

    if hasattr(client, "chat_completion"):
        response = client.chat_completion(messages=messages, model=MODEL_ID)
        content = _extract_content(response)
        input_tokens, output_tokens = _extract_tokens(response)
    else:
        if payload_inputs.get("audio") is not None or payload_inputs.get("image") is not None:
            raise RuntimeError(
                "This version of huggingface_hub does not support multimodal "
                "chat completions. Upgrade huggingface_hub to enable audio/image inputs."
            )
        text_input = payload_inputs.get("text", "")
        prompt_text = f"{prompt}\n{text_input}".strip()
        response = client.text_generation(prompt_text, model=MODEL_ID)
        content = response if isinstance(response, str) else str(response)
        input_tokens = None
        output_tokens = None

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return LLMResponse(
        content=content,
        latency_ms=round(latency_ms, 2),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

