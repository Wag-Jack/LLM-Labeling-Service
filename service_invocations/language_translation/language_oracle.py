from pathlib import Path
import re

import pandas as pd

from service_invocations.core.oracle_utils import extract_oracle as _extract_oracle
from service_invocations.models import get_enabled_models, get_model_generator

_PROMPT = """
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

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify_model(name: str) -> str:
    slug = _SLUG_RE.sub("_", name.lower()).strip("_")
    return slug or "model"


def generate_oracle_translations(
    europarl_data,
    use_existing: bool = False,
    results_dir: Path | None = None,
    models_path: Path | None = None,
):
    if results_dir is None:
        results_dir = Path.cwd() / "service_invocations" / "results" / "language_translation"
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"
    results_dir.mkdir(parents=True, exist_ok=True)

    enabled_models = get_enabled_models(models_path)
    if not enabled_models:
        print("Skipping LLM Oracle Translation (no enabled models).")
        return {}

    results_by_model: dict[str, pd.DataFrame] = {}
    multiple_models = len(enabled_models) > 1

    for model_name in enabled_models:
        model_slug = _slugify_model(model_name)
        results_path = results_dir / (
            f"language_oracle__{model_slug}.csv" if multiple_models else "language_oracle.csv"
        )

        if use_existing and results_path.exists():
            results_by_model[model_name] = pd.read_csv(results_path)
            continue

        generator = get_model_generator(model_name, models_path=models_path)

        data = {
            "id": [],
            "llm_oracle": [],
            "latency_ms": [],
            "input_tokens": [],
            "output_tokens": [],
        }

        for _, row in europarl_data.iterrows():
            sample_id = row["id"]
            english_text = row["english"]
            print(f"LLM Oracle Translation ({model_name}): {sample_id}")

            response = generator(
                _PROMPT,
                inputs={"text": english_text},
            )
            content = response.content
            print(content)
            llm_oracle = _extract_oracle(content)

            data["id"].append(sample_id)
            data["llm_oracle"].append(llm_oracle)
            data["latency_ms"].append(response.latency_ms)
            data["input_tokens"].append(response.input_tokens)
            data["output_tokens"].append(response.output_tokens)

        results_df = pd.DataFrame(data)
        results_df.to_csv(results_path, index=False)
        results_by_model[model_name] = results_df

    if multiple_models:
        return results_by_model
    return next(iter(results_by_model.values()))
