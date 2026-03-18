import json
import pandas as pd
from pathlib import Path
import re

from service_invocations.core.llm_adapters import get_llm_adapter, UnsupportedProviderError


_ID_RE = re.compile(r"(\d+)$")  # Extract trailing digits for stable id normalization.


def _normalize_id(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)) and float(value).is_integer():
        return f"{int(value):04d}"
    value_str = str(value)
    match = _ID_RE.search(value_str)
    if match:
        digits = match.group(1)
        if len(digits) <= 4:
            return digits.zfill(4)
        return digits
    return value_str


def _slugify_model(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "model"


def _normalize_model_entries(model_entries):
    # Accept None (no models), dict (single), or list (multi) model inputs.
    if model_entries is None:
        return []
    if isinstance(model_entries, dict):
        if not model_entries.get("enabled", True):
            return []
        return [model_entries]
    if isinstance(model_entries, list):
        return [entry for entry in model_entries if entry.get("enabled", True)]
    raise ValueError("model_entries must be a list, dict, or None.")


def generate_oracle_translations(europarl_data, use_existing=False, results_path=None,
                                 results_dir=None, model_entries=None,
                                 return_by_model: bool = False):
    if results_dir is None:
        results_dir = Path.cwd() / "service_invocations" / "results" / "language_translation"
    results_dir.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = results_dir / "language_oracle.csv"

    normalized_models = _normalize_model_entries(model_entries)
    if len(normalized_models) > 1:
        return_by_model = True
    if not normalized_models:
        print("Skipping LLM Oracle Translation (no model entries provided).")
        return {} if return_by_model else pd.DataFrame({"id": [], "llm_oracle": []})

    english_inputs = europarl_data["english"].tolist()
    ids = europarl_data["id"].tolist()

    results_by_model = {}

    for model_entry in normalized_models:
        model_name = model_entry.get("name") or model_entry.get("model") or "model"
        provider = model_entry.get("provider")
        if not provider:
            print(f"Skipping model '{model_name}' (missing provider).")
            continue
        try:
            adapter = get_llm_adapter(provider)
        except UnsupportedProviderError:
            print(f"Skipping model '{model_name}' (provider {provider} not supported).")
            continue

        model_id = model_entry.get("model")
        if not model_id:
            print(f"Skipping model '{model_name}' (missing model id).")
            continue
        modalities = model_entry.get("modalities")
        if modalities is None:
            modalities = ["text"]
        elif isinstance(modalities, (list, tuple)):
            modalities = list(modalities)
        else:
            raise ValueError("modalities must be a list when provided.")
        if "text" not in modalities:
            print(f"Skipping model '{model_name}' (modalities exclude text).")
            continue

        model_slug = _slugify_model(model_name)
        model_results_path = results_path if not return_by_model else (
            results_dir / f"language_oracle__{model_slug}.csv"
        )

        if use_existing and model_results_path.exists():
            results_by_model[model_name] = pd.read_csv(model_results_path)
            continue

        data = {
            "id": [],
            "llm_oracle": [],
            "latency_ms": [],
            "input_tokens": [],
            "output_tokens": []
        }

        for sample_id, english in zip(ids, english_inputs):
            print(f"LLM Oracle Translation ({model_name}): {english}")

            prompt = """
                      Translate the following English text to French.
                      You MUST return ONLY valid JSON. Do not include markdown, code fences, or explanations.
                      JSON schema:
                      {
                        "llm_oracle": string|null
                      }
                      If you do not receive the English input, enter llm_oracle as 'n/a'.
                      Do NOT mention that you need the English input, only give the JSON schema output.
                      If you violate this, the output will be discarded.
                      """

            response = adapter.generate(
                model=model_id,
                prompt=prompt,
                inputs={"text": english},
                modalities=modalities,
            )

            print(f"{response.content}")
            try:
                llm_output = json.loads(response.content)
            except json.JSONDecodeError:
                llm_output = {"llm_oracle": "n/a"}

            data["id"].append(_normalize_id(sample_id))
            data["llm_oracle"].append(llm_output.get("llm_oracle", "n/a"))
            data["latency_ms"].append(response.latency_ms)
            data["input_tokens"].append(response.input_tokens)
            data["output_tokens"].append(response.output_tokens)

        oracle_results = pd.DataFrame(data)
        oracle_results.to_csv(model_results_path, index=False)
        results_by_model[model_name] = oracle_results

    if return_by_model:
        return results_by_model
    if not results_by_model:
        return pd.DataFrame({"id": [], "llm_oracle": []})
    return next(iter(results_by_model.values()))
