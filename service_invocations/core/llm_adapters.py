from __future__ import annotations

from dataclasses import dataclass
import json
import base64
import os
import random
import socket
import sys
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Tuple
from urllib import error as url_error
from urllib import request as url_request

from dotenv import load_dotenv

load_dotenv()


class UnsupportedProviderError(RuntimeError):
    pass


class ModelUnavailableError(RuntimeError):
    """Raised after retries are exhausted on a transient/retryable error.

    Callers (e.g., the failover runner) can catch this to defer remaining
    work for the model and proceed with the next one. Non-retryable errors
    are not wrapped — they propagate as-is.
    """

    def __init__(self, model_id: str, original: BaseException):
        super().__init__(f"Model '{model_id}' unavailable after retries: {original}")
        self.model_id = model_id
        self.original = original


_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
_RETRYABLE_KEYWORDS = (
    "503",
    "502",
    "504",
    "429",
    "unavailable",
    "overloaded",
    "temporarily",
    "timeout",
    "timed out",
    "deadline",
    "rate limit",
    "rate-limit",
    "resource exhausted",
    "connection reset",
    "connection aborted",
    "remote end closed",
    "internal server error",
    "bad gateway",
    "gateway timeout",
)

_DEFAULT_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "6"))
_DEFAULT_BASE_DELAY = float(os.getenv("LLM_RETRY_BASE_DELAY", "2.0"))
_DEFAULT_MAX_DELAY = float(os.getenv("LLM_RETRY_MAX_DELAY", "60.0"))
_DEFAULT_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "120.0"))


def _parse_retry_after(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _classify_exception(exc: BaseException) -> Tuple[bool, float | None]:
    """Return (is_retryable, suggested_delay_seconds)."""
    if isinstance(exc, url_error.HTTPError):
        if exc.code in _RETRYABLE_STATUS_CODES:
            retry_after = None
            if getattr(exc, "headers", None) is not None:
                retry_after = exc.headers.get("Retry-After")
            return True, _parse_retry_after(retry_after)
        return False, None

    if isinstance(exc, (socket.timeout, TimeoutError, ConnectionError)):
        return True, None

    if isinstance(exc, url_error.URLError):
        return True, None

    status_code = (
        getattr(exc, "code", None)
        or getattr(exc, "status_code", None)
        or getattr(exc, "status", None)
        or getattr(exc, "http_status", None)
    )
    if isinstance(status_code, int) and status_code in _RETRYABLE_STATUS_CODES:
        retry_after = None
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", None)
            if headers is not None:
                try:
                    retry_after = headers.get("Retry-After")
                except Exception:
                    retry_after = None
        return True, _parse_retry_after(retry_after)

    message = str(exc).lower()
    if any(keyword in message for keyword in _RETRYABLE_KEYWORDS):
        return True, None

    return False, None


def _retry_call(
    func: Callable[[], Any],
    *,
    description: str,
    model_id: str,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    base_delay: float = _DEFAULT_BASE_DELAY,
    max_delay: float = _DEFAULT_MAX_DELAY,
) -> Any:
    attempt = 0
    while True:
        try:
            return func()
        except Exception as exc:
            retryable, suggested = _classify_exception(exc)
            if not retryable:
                raise
            if attempt >= max_retries:
                print(
                    f"[llm-retry] {description} exhausted {max_retries + 1} attempts; "
                    f"marking model unavailable. Last error: "
                    f"{type(exc).__name__}: {exc}.",
                    file=sys.stderr,
                    flush=True,
                )
                raise ModelUnavailableError(model_id, exc) from exc
            if suggested is not None and suggested > 0:
                delay = min(suggested, max_delay)
            else:
                delay = min(base_delay * (2 ** attempt), max_delay)
            delay += random.uniform(0, max(0.25, delay * 0.25))
            print(
                f"[llm-retry] {description} failed "
                f"(attempt {attempt + 1}/{max_retries + 1}): "
                f"{type(exc).__name__}: {exc}. Retrying in {delay:.1f}s.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
            attempt += 1


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


def _is_realtime_model(model: str) -> bool:
    return "realtime" in (model or "").lower()


def _extract_realtime_text(response: Dict[str, Any] | None) -> str:
    if not isinstance(response, dict):
        return ""
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    chunks: List[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("output_text", "text"):
                    text = part.get("text")
                    if text:
                        chunks.append(text)
        text = item.get("text")
        if text:
            chunks.append(text)
    return "".join(chunks).strip()


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
                 modalities: List[str], temperature: float = 0.0) -> LLMResponse:
        start_time = time.perf_counter()
        if self._mode == "genai":
            parts = [self._types.Part.from_text(text=prompt)]
            text_input = inputs.get("text")
            if text_input:
                parts.append(self._types.Part.from_text(text=text_input))

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
            response = _retry_call(
                lambda: self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=self._types.GenerateContentConfig(temperature=temperature),
                ),
                description=f"Gemini generate_content ({model})",
                model_id=model,
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
            response = _retry_call(
                lambda: model_client.generate_content(
                    parts,
                    generation_config={"temperature": temperature},
                ),
                description=f"Gemini generate_content ({model})",
                model_id=model,
            )
            content = getattr(response, "text", None) or str(response)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        input_tokens, output_tokens = _extract_gemini_usage(response)
        return LLMResponse(
            content=content,
            latency_ms=round(latency_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def _extract_gemini_usage(response: Any) -> tuple[int | None, int | None]:
    """Pull prompt/output token counts off a Gemini response.

    Both the legacy google-generativeai SDK and the newer google-genai SDK
    expose token usage on a ``usage_metadata`` attribute (or dict). Field names
    differ slightly across SDK versions, so try several.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage_metadata") or response.get("usage")
    if usage is None:
        return None, None

    def _get(obj: Any, *keys: str) -> int | None:
        for key in keys:
            value = None
            if isinstance(obj, dict):
                value = obj.get(key)
            else:
                value = getattr(obj, key, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return None

    input_tokens = _get(usage, "prompt_token_count", "input_token_count", "input_tokens", "prompt_tokens")
    output_tokens = _get(usage, "candidates_token_count", "output_token_count", "output_tokens", "completion_tokens")
    return input_tokens, output_tokens


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

    def _post_json(self, payload: Dict[str, Any], model_id: str) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "api-key": self._api_key,
        }

        def _do_request() -> bytes:
            req = url_request.Request(
                self._target_uri,
                data=data,
                headers=headers,
                method="POST",
            )
            try:
                with url_request.urlopen(req, timeout=_DEFAULT_REQUEST_TIMEOUT) as resp:
                    return resp.read()
            except url_error.HTTPError as exc:
                if exc.code in _RETRYABLE_STATUS_CODES:
                    raise
                body = exc.read()
                detail = body.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Microsoft Phi request failed: HTTP {exc.code} {exc.reason} - {detail}"
                ) from exc

        body = _retry_call(
            _do_request,
            description=f"Microsoft Phi request ({model_id})",
            model_id=model_id,
        )
        return json.loads(body.decode("utf-8"))

    def generate(self, model: str, prompt: str, inputs: Dict[str, Any],
                 modalities: List[str], temperature: float = 0.0) -> LLMResponse:
        messages = _build_openai_messages(prompt, inputs)
        start_time = time.perf_counter()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        response = self._post_json(payload, model_id=model)
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
    if provider_key == "gemini":
        adapter = GeminiAdapter()
        _ADAPTER_CACHE[provider_key] = adapter
        return adapter
    if provider_key == "microsoft":
        adapter = MicrosoftPhiAdapter()
        _ADAPTER_CACHE[provider_key] = adapter
        return adapter
    raise UnsupportedProviderError(f"provider '{provider}' is not supported yet")
