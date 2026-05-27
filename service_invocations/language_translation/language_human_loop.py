from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import yaml

from service_invocations.core.cost_tracker import compute_cost, session_tracker
from service_invocations.core.model_failover import run_with_failover
from service_invocations.core.oracle_utils import (
    load_prompt as _load_prompt,
    normalize_id as _normalize_id,
    resolve_prompt_path as _resolve_prompt_path,
)
from service_invocations.core.results_io import write_human_loop
from service_invocations.models import get_enabled_models, get_model_generator

_PARADIGM_NAME = "human_loop"
_TASK_NAME = "language_translation"
_PROMPTS_ROOT = Path(__file__).parent / "prompts"
_PARADIGM = "human-loop"
_CONFIDENCE_THRESHOLD = 0.7
_GROUND_TRUTH_COLUMN = "french"

_DEFAULT_TASK_DIR = (
    Path.cwd() / "service_invocations" / "results" / "language_translation"
)


def _load_enabled_entries(config_path: Path, task_name: str) -> list[str]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping.")

    task_cfg = config.get(task_name, {})
    if not isinstance(task_cfg, dict):
        raise ValueError(f"Config '{task_name}' must be a mapping.")

    enabled = []
    for name, entry in task_cfg.items():
        if isinstance(entry, dict) and entry.get("enabled", False):
            enabled.append(name)
    return enabled


def human_loop_translations(
    results_by_service: dict[str, pd.DataFrame],
    europarl_data: pd.DataFrame,
    prompt_name: str,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
    results_dir: Path | None = None,
    services_path: Path | None = None,
    models_path: Path | None = None,
    task_name: str = _TASK_NAME,
):
    task_dir = results_dir if results_dir is not None else _DEFAULT_TASK_DIR
    if services_path is None:
        services_path = Path.cwd() / "config" / "services.yaml"
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"
    task_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = _resolve_prompt_path(_PROMPTS_ROOT, _PARADIGM, prompt_name)

    enabled_services = _load_enabled_entries(services_path, task_name)
    enabled_models = get_enabled_models(models_path)

    if not enabled_models:
        print("--- Skipping LLM Human-Loop (no enabled models) ---")
        return None

    services = {
        name: df for name, df in results_by_service.items() if name in enabled_services
    }
    if not services:
        print("--- Skipping LLM Human-Loop (no enabled service results) ---")
        return None

    translations_by_service = {}
    for name, df in services.items():
        if "id" not in df.columns or "service_output" not in df.columns:
            raise ValueError(f"Missing required columns for {name}.")
        translations_by_service[name] = dict(
            zip(df["id"].map(_normalize_id), df["service_output"])
        )

    ground_truth = dict(
        zip(europarl_data["id"].map(_normalize_id), europarl_data[_GROUND_TRUTH_COLUMN])
    )
    english_input = europarl_data["english"].tolist()
    ids = europarl_data["id"].tolist()

    samples = [
        {"id": sample_id, "english": english}
        for sample_id, english in zip(ids, english_input)
    ]

    def make_processor(model_name: str):
        generator = get_model_generator(model_name, models_path=models_path)
        tracker = session_tracker()

        def process(sample: dict) -> dict:
            sample_id = sample["id"]
            english = sample["english"]
            id_key = _normalize_id(sample_id)
            print(f"LLM Human-Loop ({model_name}): {english}")

            service_blocks = "\n".join(
                f"{name}: {translations_by_service[name].get(id_key, '')}"
                for name in services.keys()
            )
            prompt = _load_prompt(
                prompt_path,
                source_text=english,
                service_blocks=service_blocks,
            )

            response = generator(prompt, inputs={"text": english})
            content = response.content
            print(content)
            cost = compute_cost(
                model_name, response.input_tokens, response.output_tokens, models_path
            )
            tracker.record(
                task=task_name,
                paradigm=_PARADIGM_NAME,
                model=model_name,
                sample_id=sample_id,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=cost,
            )

            default_scores = {name: -1 for name in services.keys()}
            try:
                llm_output = json.loads(content)
                if not isinstance(llm_output, dict):
                    llm_output = {}
            except json.JSONDecodeError:
                llm_output = {}

            scores = llm_output.get("scores", {})
            if not isinstance(scores, dict):
                scores = {}
            merged_scores = {**default_scores, **scores}

            confidence = llm_output.get("confidence")
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = 0.0

            llm_translation = llm_output.get("llm_translation", "n/a")
            fallback_used = confidence_value < confidence_threshold
            human_label_used = ""
            if fallback_used:
                human_label_used = ground_truth.get(id_key, "") or ""
                llm_translation = human_label_used or llm_translation

            winner = llm_output.get("winner") if isinstance(llm_output, dict) else None
            if winner is not None and winner not in services:
                winner = None

            return {
                "id": id_key,
                "llm_label": llm_translation,
                "winner": winner,
                "scores": merged_scores,
                "confidence": confidence_value,
                "fallback_used": fallback_used,
                "human_label_used": human_label_used,
                "latency_ms": response.latency_ms,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": cost,
            }

        return process

    def on_progress(model_name: str, rows: list[dict], is_final: bool) -> None:
        long_rows: list[dict] = []
        for r in rows:
            scores = r.get("scores") or {}
            winner = r.get("winner")
            for service_name in services.keys():
                long_rows.append({
                    "id": r["id"],
                    "service": service_name,
                    "score": scores.get(service_name, -1),
                    "is_winner": winner == service_name if winner is not None else False,
                    "llm_label": r.get("llm_label", "n/a"),
                    "winner": winner,
                    "confidence": r.get("confidence", 0.0),
                    "fallback_used": r.get("fallback_used", False),
                    "human_label_used": r.get("human_label_used", ""),
                    "latency_ms": r.get("latency_ms"),
                    "input_tokens": r.get("input_tokens"),
                    "output_tokens": r.get("output_tokens"),
                    "cost_usd": r.get("cost_usd"),
                })
        write_human_loop(task_dir, task_name, prompt_name, model_name, long_rows)

    run_with_failover(
        models=enabled_models,
        samples=samples,
        make_processor=make_processor,
        on_progress=on_progress,
    )

    return True
