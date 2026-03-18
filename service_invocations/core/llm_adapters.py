from __future__ import annotations

from dataclasses import dataclass
import base64
import os
from pathlib import Path
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class UnsupportedProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    content: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None


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


def _build_openai_messages(prompt: str, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

    text_input = inputs.get("text")
    if text_input:
        content.append({"type": "text", "text": text_input})

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

    image_input = inputs.get("image")
    if image_input is not None:
        image_bytes = _read_bytes(image_input)
        image_format = inputs.get("image_format") or _infer_format(image_input, "png")
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{image_format};base64,{image_b64}",
            },
        })

    return [{"role": "user", "content": content}]


def _is_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _to_data_url(value: Any, media_type: str, default_format: str) -> str:
    if isinstance(value, str):
        if value.startswith("data:") or _is_http_url(value):
            return value
    media_bytes = _read_bytes(value)
    media_format = _infer_format(value, default_format)
    media_b64 = base64.b64encode(media_bytes).decode("utf-8")
    return f"data:{media_type}/{media_format};base64,{media_b64}"


def _build_dashscope_messages(prompt: str, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    text_parts: List[str] = []
    if prompt:
        text_parts.append(prompt)

    text_input = inputs.get("text")
    if text_input:
        text_parts.append(text_input)

    image_input = inputs.get("image")
    audio_input = inputs.get("audio")

    # DashScope expects a string for text-only inputs and a list for multimodal content.
    if image_input is None and audio_input is None:
        return [{"role": "user", "content": "\n".join(text_parts).strip()}]

    content: List[Dict[str, Any]] = []
    if text_parts:
        content.append({"text": "\n".join(text_parts)})

    if image_input is not None:
        image_format = inputs.get("image_format") or _infer_format(image_input, "png")
        content.append({"image": _to_data_url(image_input, "image", image_format)})

    if audio_input is not None:
        audio_format = inputs.get("audio_format") or _infer_format(audio_input, "wav")
        content.append({"audio": _to_data_url(audio_input, "audio", audio_format)})

    return [{"role": "user", "content": content}]


def _normalize_dashscope_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
            elif isinstance(item, str):
                text_parts.append(item)
        if text_parts:
            return "".join(text_parts)
    return str(content)


def _extract_dashscope_content(response: Any) -> str:
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")

    choices = None
    if output is not None:
        choices = getattr(output, "choices", None)
        if choices is None and isinstance(output, dict):
            choices = output.get("choices")

    if choices:
        first_choice = choices[0]
        message = None
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
        else:
            message = getattr(first_choice, "message", None)
        if message is not None:
            if isinstance(message, dict):
                return _normalize_dashscope_content(message.get("content"))
            return _normalize_dashscope_content(getattr(message, "content", None))

    if isinstance(output, dict):
        text = output.get("text") or output.get("output_text")
        if text is not None:
            return str(text)
    if isinstance(response, dict):
        text = response.get("text")
        if text is not None:
            return str(text)
    return str(response)


def _normalize_dashscope_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/compatible-mode/v1"):
        normalized = normalized.replace("/compatible-mode/v1", "/api/v1")
    return normalized


class OpenAIAdapter:
    def __init__(self) -> None:
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, model: str, prompt: str, inputs: Dict[str, Any],
                 modalities: List[str]) -> LLMResponse:
        messages = _build_openai_messages(prompt, inputs)
        start_time = time.perf_counter()
        response = self._client.chat.completions.create(
            model=model,
            modalities=modalities,
            messages=messages,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if input_tokens is None:
            input_tokens = prompt_tokens
        if output_tokens is None:
            output_tokens = completion_tokens
        content = response.choices[0].message.content
        return LLMResponse(
            content=content,
            latency_ms=round(latency_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class GeminiAdapter:
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise UnsupportedProviderError(
                "Gemini adapter requires GEMINI_API_KEY or GOOGLE_API_KEY."
            )

        try:
            from google import genai  # type: ignore
            from google.genai import types as genai_types  # type: ignore
        except Exception:
            genai = None
            genai_types = None

        if genai is not None and genai_types is not None:
            self._mode = "genai"
            self._client = genai.Client(api_key=api_key)
            self._types = genai_types
            return

        try:
            import google.generativeai as genai_legacy  # type: ignore
        except Exception as exc:
            raise UnsupportedProviderError(
                "Gemini adapter requires `google-genai` or `google-generativeai`."
            ) from exc

        genai_legacy.configure(api_key=api_key)
        self._mode = "generativeai"
        self._client = genai_legacy

    def generate(self, model: str, prompt: str, inputs: Dict[str, Any],
                 modalities: List[str]) -> LLMResponse:
        start_time = time.perf_counter()
        if self._mode == "genai":
            parts = [self._types.Part.from_text(prompt)]
            text_input = inputs.get("text")
            if text_input:
                parts.append(self._types.Part.from_text(text_input))

            audio_input = inputs.get("audio")
            if audio_input is not None:
                audio_bytes = _read_bytes(audio_input)
                audio_format = inputs.get("audio_format") or _infer_format(audio_input, "wav")
                parts.append(self._types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=f"audio/{audio_format}",
                ))

            image_input = inputs.get("image")
            if image_input is not None:
                image_bytes = _read_bytes(image_input)
                image_format = inputs.get("image_format") or _infer_format(image_input, "png")
                parts.append(self._types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=f"image/{image_format}",
                ))

            contents = [self._types.Content(role="user", parts=parts)]
            response = self._client.models.generate_content(
                model=model,
                contents=contents,
            )
            content = getattr(response, "text", None)
            if content is None and getattr(response, "candidates", None):
                candidate = response.candidates[0]
                content_parts = getattr(candidate, "content", None)
                if content_parts and getattr(content_parts, "parts", None):
                    content = getattr(content_parts.parts[0], "text", None)
            if content is None:
                content = str(response)
        else:
            model_client = self._client.GenerativeModel(model)
            parts: List[Any] = [prompt]
            text_input = inputs.get("text")
            if text_input:
                parts.append(text_input)
            audio_input = inputs.get("audio")
            if audio_input is not None:
                audio_bytes = _read_bytes(audio_input)
                audio_format = inputs.get("audio_format") or _infer_format(audio_input, "wav")
                parts.append(self._client.types.Blob(
                    mime_type=f"audio/{audio_format}",
                    data=audio_bytes,
                ))
            image_input = inputs.get("image")
            if image_input is not None:
                image_bytes = _read_bytes(image_input)
                image_format = inputs.get("image_format") or _infer_format(image_input, "png")
                parts.append(self._client.types.Blob(
                    mime_type=f"image/{image_format}",
                    data=image_bytes,
                ))
            response = model_client.generate_content(parts)
            content = getattr(response, "text", None) or str(response)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return LLMResponse(
            content=content,
            latency_ms=round(latency_ms, 2),
            input_tokens=None,
            output_tokens=None,
        )


class QwenAdapter:
    def __init__(self) -> None:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not api_key:
            raise UnsupportedProviderError(
                "Qwen adapter requires DASHSCOPE_API_KEY or QWEN_API_KEY."
            )
        try:
            import dashscope  # type: ignore
        except Exception as exc:
            raise UnsupportedProviderError(
                "Qwen adapter requires the DashScope SDK (`dashscope`)."
            ) from exc

        base_url = (
            os.getenv("DASHSCOPE_HTTP_API_URL")
            or os.getenv("DASHSCOPE_BASE_URL")
            or os.getenv("QWEN_BASE_URL")
            or "https://dashscope.aliyuncs.com/api/v1"
        )
        base_url = _normalize_dashscope_base_url(base_url)

        dashscope.base_http_api_url = base_url
        dashscope.api_key = api_key

        self._dashscope = dashscope
        self._api_key = api_key
        self._base_url = base_url

    def generate(self, model: str, prompt: str, inputs: Dict[str, Any],
                 modalities: List[str]) -> LLMResponse:
        messages = _build_dashscope_messages(prompt, inputs)
        audio_input = inputs.get("audio")
        start_time = time.perf_counter()
        use_multimodal = isinstance(messages[0].get("content"), list)
        if use_multimodal:
            kwargs: Dict[str, Any] = {
                "api_key": self._api_key,
                "model": model,
                "messages": messages,
                "result_format": "message",
            }
            if audio_input is not None and inputs.get("asr_options") is not None:
                kwargs["asr_options"] = inputs["asr_options"]
            response = self._dashscope.MultiModalConversation.call(**kwargs)
        else:
            response = self._dashscope.Generation.call(
                api_key=self._api_key,
                model=model,
                messages=messages,
                result_format="message",
            )
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")

        input_tokens = None
        output_tokens = None
        if usage is not None:
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
                output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
            else:
                input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)

        content = _extract_dashscope_content(response)
        return LLMResponse(
            content=content,
            latency_ms=round(latency_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


_ADAPTER_CACHE: Dict[str, Any] = {}


def get_llm_adapter(provider: str) -> Any:
    if provider is None:
        raise UnsupportedProviderError(
            "LLM provider must be specified; no default provider is assumed."
        )
    provider_key = str(provider).strip().lower()
    if not provider_key:
        raise UnsupportedProviderError(
            "LLM provider must be specified; no default provider is assumed."
        )
    provider_aliases = {
        "google": "gemini",
        "gemini": "gemini",
        "alibaba": "qwen",
        "qwen": "qwen",
        "quen": "qwen",
        "dashscope": "qwen",
    }
    provider_key = provider_aliases.get(provider_key, provider_key)
    adapter = _ADAPTER_CACHE.get(provider_key)
    if adapter is not None:
        return adapter
    if provider_key == "openai":
        adapter = OpenAIAdapter()
        _ADAPTER_CACHE[provider_key] = adapter
        return adapter
    if provider_key == "gemini":
        adapter = GeminiAdapter()
        _ADAPTER_CACHE[provider_key] = adapter
        return adapter
    if provider_key == "qwen":
        adapter = QwenAdapter()
        _ADAPTER_CACHE[provider_key] = adapter
        return adapter
    raise UnsupportedProviderError(f"provider '{provider}' is not supported yet")
