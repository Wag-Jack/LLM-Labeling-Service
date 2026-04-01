from __future__ import annotations

from typing import Any, Dict, List


def infer_modalities(inputs: Dict[str, Any] | None) -> List[str]:
    """Infer modalities from inputs; always include text for the prompt."""
    modalities = ["text"]
    if not inputs:
        return modalities
    if inputs.get("audio") is not None:
        modalities.append("audio")
    if inputs.get("image") is not None:
        modalities.append("image")
    return modalities

