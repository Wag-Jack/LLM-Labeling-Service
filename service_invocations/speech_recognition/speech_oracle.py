import base64
from dotenv import load_dotenv
import json
from openai import OpenAI
import os
import pandas as pd
from pathlib import Path
import re

load_dotenv()


_ID_RE = re.compile(r"(\d+)$")  # Extract trailing digits for stable id normalization.


def _normalize_id(value) -> str:
    # Normalize ids like 12 -> "0012" and "gc_stt_0012" -> "0012".
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
    # Safe file suffix from model name (used in per-model output files).
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "model"


def _normalize_model_entries(model_entries):
    # Accept None (use default), dict (single), or list (multi) model inputs.
    if model_entries is None:
        return [{
            "name": "gpt_audio",
            "provider": "openai",
            "model": "gpt-audio",
        }]
    if isinstance(model_entries, dict):
        if not model_entries.get("enabled", True):
            return []
        return [model_entries]
    if isinstance(model_entries, list):
        return [entry for entry in model_entries if entry.get("enabled", True)]
    raise ValueError("model_entries must be a list, dict, or None.")


def generate_oracle_transcripts(edacc_data, use_existing=False, results_path=None,
                                results_dir=None, model_entries=None,
                                return_by_model: bool = False):
    if results_dir is None:
        results_dir = Path.cwd() / "service_invocations" / "results" / "speech_recognition"
    results_dir.mkdir(parents=True, exist_ok=True)
    # Default to a single output file unless explicitly asked to split by model.
    if results_path is None:
        results_path = results_dir / "speech_oracle.csv"

    normalized_models = _normalize_model_entries(model_entries)
    if len(normalized_models) > 1:
        return_by_model = True

    # Reuse a single OpenAI client for all models in this run.
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Open metadata of EdAcc and retrieve all the audio paths as a list
    wav_files = edacc_data['audio'].tolist()
    ids = edacc_data['id'].tolist()

    results_by_model = {}

    for model_entry in normalized_models:
        # Only OpenAI models are supported by this audio-oracle implementation.
        provider = model_entry.get("provider", "openai")
        if provider != "openai":
            print(f"Skipping model '{model_entry.get('name')}' (provider {provider} not supported).")
            continue

        model_name = model_entry.get("name") or model_entry.get("model") or "model"
        model_id = model_entry.get("model")
        if not model_id:
            print(f"Skipping model '{model_name}' (missing model id).")
            continue

        model_slug = _slugify_model(model_name)
        # Per-model output when multiple models are requested.
        model_results_path = results_path if not return_by_model else (
            results_dir / f"speech_oracle__{model_slug}.csv"
        )

        # If asked, reuse existing outputs from disk for this model.
        if use_existing and model_results_path.exists():
            results_by_model[model_name] = pd.read_csv(model_results_path)
            continue

        data = {
            "id": [],
            "llm_oracle": []
        }

        for sample_id, wav in zip(ids, wav_files):
            print(f"LLM Oracle Transcript ({model_name}): {wav}")

            prompt = """
                      Please give me a transcript for the following audio file.
                      You MUST return ONLY valid JSON. Do not include markdown, code fences, or explanations.
                      JSON schema:
                      {
                        "llm_oracle": string|null
                      }
                      If you do not receive the WAV file, enter llm_oracle as 'n/a'.
                      Do NOT mention that you need the WAV file, only give the JSON schema output.
                      If you violate this, the output will be discarded.
                      """

            # Open designated audio file
            with open(wav, 'rb') as f:
                audio_bytes = f.read()

            audio = base64.b64encode(audio_bytes).decode("utf-8")

            response = client.chat.completions.create(
                model=model_id,
                modalities=['text'],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
                        ]
                    }
                ]
            )

            # Compile JSON object from LLM output
            print(f"{response.choices[0].message.content}")
            try:
                llm_output = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                llm_output = {"llm_oracle": "n/a"}

            # Append data to resultant data dictionary
            data['id'].append(_normalize_id(sample_id))
            data['llm_oracle'].append(llm_output.get('llm_oracle', "n/a"))

        oracle_results = pd.DataFrame(data)
        oracle_results.to_csv(model_results_path, index=False)
        results_by_model[model_name] = oracle_results

    if return_by_model:
        return results_by_model
    if not results_by_model:
        return pd.DataFrame({"id": [], "llm_oracle": []})
    return next(iter(results_by_model.values()))
