from pathlib import Path

import pandas as pd

from service_invocations.core.cost_tracker import compute_cost, session_tracker
from service_invocations.core.model_failover import run_with_failover
from service_invocations.core.oracle_utils import (
    extract_oracle as _extract_oracle,
    load_prompt as _load_prompt,
    normalize_id as _normalize_id,
    resolve_prompt_path as _resolve_prompt_path,
)
from service_invocations.core.results_io import write_oracle
from service_invocations.models import get_enabled_models, get_model_generator

_TASK_NAME = "language_translation"
_PARADIGM_NAME = "oracle"

_PROMPTS_ROOT = Path(__file__).parent / "prompts"
_PARADIGM = "oracle"

_DEFAULT_TASK_DIR = (
    Path.cwd() / "service_invocations" / "results" / "language_translation"
)


def generate_oracle_translations(
    europarl_data,
    prompt_name: str,
    use_existing: bool = False,
    results_dir: Path | None = None,
    models_path: Path | None = None,
):
    task_dir = results_dir if results_dir is not None else _DEFAULT_TASK_DIR
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"
    task_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = _resolve_prompt_path(_PROMPTS_ROOT, _PARADIGM, prompt_name)

    enabled_models = get_enabled_models(models_path)
    if not enabled_models:
        print("Skipping LLM Oracle Translation (no enabled models).")
        return {}

    results_by_model: dict[str, pd.DataFrame] = {}

    samples = [
        {"id": row["id"], "english": row["english"]}
        for _, row in europarl_data.iterrows()
    ]

    pending_models: list[str] = []
    existing_path = task_dir / "oracle.csv"
    existing_df = pd.read_csv(existing_path) if existing_path.exists() else None
    for model_name in enabled_models:
        if use_existing and existing_df is not None:
            slice_df = existing_df[
                (existing_df["prompt"] == prompt_name)
                & (existing_df["model"] == model_name)
            ]
            if not slice_df.empty:
                results_by_model[model_name] = slice_df.reset_index(drop=True)
                continue
        pending_models.append(model_name)

    def make_processor(model_name: str):
        generator = get_model_generator(model_name, models_path=models_path)
        prompt = _load_prompt(prompt_path)
        tracker = session_tracker()

        def process(sample: dict) -> dict:
            sample_id = sample["id"]
            english_text = sample["english"]
            print(f"LLM Oracle Translation ({model_name}): {sample_id}")

            response = generator(prompt, inputs={"text": english_text})
            content = response.content
            print(content)
            llm_oracle = _extract_oracle(content)
            cost = compute_cost(
                model_name, response.input_tokens, response.output_tokens, models_path
            )
            tracker.record(
                task=_TASK_NAME,
                paradigm=_PARADIGM_NAME,
                model=model_name,
                sample_id=sample_id,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=cost,
            )
            return {
                "id": _normalize_id(sample_id),
                "llm_oracle": llm_oracle,
                "latency_ms": response.latency_ms,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": cost,
            }

        return process

    def on_progress(model_name: str, rows: list[dict], is_final: bool) -> None:
        write_oracle(task_dir, _TASK_NAME, prompt_name, model_name, rows)
        if is_final:
            results_by_model[model_name] = pd.DataFrame(rows)

    if pending_models:
        run_with_failover(
            models=pending_models,
            samples=samples,
            make_processor=make_processor,
            on_progress=on_progress,
        )

    if len(enabled_models) > 1:
        return results_by_model
    if not results_by_model:
        return None
    return next(iter(results_by_model.values()))
