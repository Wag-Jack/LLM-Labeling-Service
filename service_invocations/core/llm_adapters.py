from __future__ import annotations

from dataclasses import dataclass
import json
import base64
import os
from pathlib import Path
import time
from typing import Any, Dict, List
from urllib import error as url_error
from urllib import request as url_request

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


class MicrosoftPhiAdapter:
    def __init__(self) -> None:
        api_key = os.getenv("MICROSOFT_PHI_KEY")
        if not api_key:
            raise UnsupportedProviderError(
                "Microsoft Phi adapter requires MICROSOFT_PHI_KEY."
            )
        target_uri = os.getenv("PHI_TARGET_URI")
        if not target_uri:
            raise UnsupportedProviderError(
                "Microsoft Phi adapter requires PHI_TARGET_URI."
            )
        self._api_key = api_key
        self._target_uri = target_uri

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "api-key": self._api_key,
        }
        req = url_request.Request(
            self._target_uri,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with url_request.urlopen(req) as resp:
                body = resp.read()
        except url_error.HTTPError as exc:
            body = exc.read()
            detail = body.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Microsoft Phi request failed: HTTP {exc.code} {exc.reason} - {detail}"
            ) from exc
        return json.loads(body.decode("utf-8"))

    def generate(self, model: str, prompt: str, inputs: Dict[str, Any],
                 modalities: List[str]) -> LLMResponse:
        messages = _build_openai_messages(prompt, inputs)
        start_time = time.perf_counter()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        response = self._post_json(payload)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        usage = response.get("usage") if isinstance(response, dict) else None
        input_tokens = None
        output_tokens = None
        prompt_tokens = None
        completion_tokens = None
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
        if input_tokens is None:
            input_tokens = prompt_tokens
        if output_tokens is None:
            output_tokens = completion_tokens
        content = ""
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
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
        "microsoft": "microsoft",
        "msft": "microsoft",
        "phi": "microsoft",
        "phi-4": "microsoft",
        "phi_4": "microsoft",
        "phi4": "microsoft",
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
    if provider_key == "microsoft":
        adapter = MicrosoftPhiAdapter()
        _ADAPTER_CACHE[provider_key] = adapter
        return adapter
    raise UnsupportedProviderError(f"provider '{provider}' is not supported yet")
