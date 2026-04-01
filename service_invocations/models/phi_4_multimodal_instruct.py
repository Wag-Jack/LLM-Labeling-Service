from __future__ import annotations

from typing import Any, Dict, List
import os

from service_invocations.core.llm_adapters import LLMResponse, get_llm_adapter
from service_invocations.models.common import infer_modalities


MODEL_ID = os.getenv("PHI_4_MULTIMODAL_INSTRUCT_MODEL_ID", "phi-4-multimodal-instruct")
PROVIDER = "microsoft"
SUPPORTED_MODALITIES = ["text", "audio", "image"]


def generate(
    prompt: str,
    inputs: Dict[str, Any] | None = None,
    modalities: List[str] | None = None,
) -> LLMResponse:
    payload_inputs = inputs or {}
    requested_modalities = modalities or infer_modalities(payload_inputs)
    adapter = get_llm_adapter(PROVIDER)
    return adapter.generate(MODEL_ID, prompt, payload_inputs, requested_modalities)

