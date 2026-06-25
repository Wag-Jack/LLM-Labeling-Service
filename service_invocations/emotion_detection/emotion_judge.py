from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import yaml

from service_invocations.core.cost_tracker import (
    make_attempt_recorder as _make_attempt_recorder,
    session_tracker,
)
from service_invocations.core.model_failover import run_with_failover
from service_invocations.core import run_context as rc
from service_invocations.core.oracle_utils import (
    is_fresh_run_requested as _is_fresh_run_requested,
    is_nullish_output as _is_nullish_output,
    judge_output_usable as _judge_output_usable,
    load_prompt as _load_prompt,
    normalize_id as _normalize_id,
    parse_json_payload as _parse_json_payload,
    resolve_prompt_path as _resolve_prompt_path,
    retry_until_valid as _retry_until_valid,
)
from service_invocations.core.results_io import (
    clear_completed_slice,
    load_completed_ids,
    write_judge,
)
from service_invocations.models import get_enabled_models, get_model_generator

_PARADIGM_NAME = "judge"
_TASK_NAME = "emotion_detection"
_PROMPTS_ROOT = Path(__file__).parent / "prompts"
_PARADIGM = "judge"

_DEFAULT_TASK_DIR = (
    Path.cwd() / "service_invocations" / "results" / "emotion_detection"
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


def _extract_top_emotion(service_output: str | None) -> str:
    if service_output is None:
        return ""
    try:
        payload = json.loads(service_output)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    top = payload.get("top_emotion")
    if isinstance(top, dict):
        name = top.get("name")
        if name:
            return str(name).strip().lower()
    return ""


def judge_emotions(
    results_by_service: dict[str, pd.DataFrame],
    affectnet_data: pd.DataFrame,
    prompt_name: str,
    results_dir: Path | None = None,
    services_path: Path | None = None,
    models_path: Path | None = None,
    task_name: str = _TASK_NAME,
    fresh_run: bool = False,
):
    task_dir = results_dir if results_dir is not None else _DEFAULT_TASK_DIR
    if services_path is None:
        services_path = rc.config_path("services.yaml")
    if models_path is None:
        models_path = rc.config_path("models.yaml")
    task_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = _resolve_prompt_path(_PROMPTS_ROOT, _PARADIGM, prompt_name)

    enabled_services = _load_enabled_entries(services_path, task_name)
    enabled_models = get_enabled_models(models_path)

    if not enabled_models:
        print("--- Skipping LLM Judging (no enabled models) ---")
        return None

    fresh_run = _is_fresh_run_requested(fresh_run)
    if fresh_run:
        removed = clear_completed_slice(task_dir, "judge", prompt_name, enabled_models)
        if removed:
            print(
                f"[fresh] emotion_judge: cleared {removed} prior row(s) "
                f"for prompt='{prompt_name}'."
            )

    services = {
        name: df for name, df in results_by_service.items() if name in enabled_services
    }
    if not services:
        print("--- Skipping LLM Judging (no enabled service results) ---")
        return None

    predictions_by_service: dict[str, dict[str, str]] = {}
    for name, df in services.items():
        if "id" not in df.columns or "service_output" not in df.columns:
            raise ValueError(f"Missing required columns for {name}.")
        predictions_by_service[name] = dict(
            zip(df["id"].map(_normalize_id), df["service_output"].map(_extract_top_emotion))
        )

    image_files = affectnet_data["image"].tolist()
    ids = affectnet_data["id"].tolist()

    samples = [
        {"id": sample_id, "image": image_file}
        for sample_id, image_file in zip(ids, image_files)
    ]

    def make_processor(model_name: str):
        generator = get_model_generator(model_name, models_path=models_path)
        tracker = session_tracker()
        done_ids: set[str] = (
            set()
            if fresh_run
            else set(load_completed_ids(task_dir, "judge", prompt_name, model_name))
        )
        if done_ids:
            print(
                f"[resume] emotion_judge {model_name}: "
                f"{len(done_ids)} sample(s) already in CSV — will skip."
            )

        def process(sample: dict) -> dict | None:
            sample_id = sample["id"]
            image_file = sample["image"]
            id_key = _normalize_id(sample_id)
            if id_key in done_ids:
                return None
            print(f"LLM Judging ({model_name}): {image_file}")

            service_blocks = "\n".join(
                f"{name}: {predictions_by_service[name].get(id_key, '')}"
                for name in services.keys()
            )
            prompt = _load_prompt(prompt_path, service_blocks=service_blocks)

            def _invoke_once():
                resp = generator(prompt, inputs={"image": image_file})
                print(resp.content)
                return resp, _parse_json_payload(resp.content)

            on_attempt, total_cost = _make_attempt_recorder(
                tracker,
                task=task_name,
                paradigm=_PARADIGM_NAME,
                model=model_name,
                sample_id=sample_id,
                usable=lambda pair: _judge_output_usable(pair[1], services),
                models_path=models_path,
            )
            response, llm_output = _retry_until_valid(
                _invoke_once,
                validate=lambda pair: not _is_nullish_output(pair[1].get("llm_emotion")),
                description=f"emotion_judge {model_name} sample={sample_id}",
                on_attempt=on_attempt,
            )
            cost = total_cost()

            default_scores = {name: -1 for name in services.keys()}
            scores = llm_output.get("scores", {})
            if not isinstance(scores, dict):
                scores = {}

            merged_scores = {**default_scores, **scores}
            winner = llm_output.get("winner") if isinstance(llm_output, dict) else None
            if winner is not None and winner not in services:
                winner = None
            done_ids.add(id_key)
            return {
                "id": id_key,
                "llm_label": llm_output.get("llm_emotion", "n/a"),
                "winner": winner,
                "scores": merged_scores,
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
                    "score": scores.get(service_name) if scores.get(service_name) is not None else -1,
                    "is_winner": winner == service_name if winner is not None else False,
                    "llm_label": r.get("llm_label", "n/a"),
                    "winner": winner,
                    "latency_ms": r.get("latency_ms"),
                    "input_tokens": r.get("input_tokens"),
                    "output_tokens": r.get("output_tokens"),
                    "cost_usd": r.get("cost_usd"),
                })
        write_judge(task_dir, task_name, prompt_name, model_name, long_rows)

    run_with_failover(
        models=enabled_models,
        samples=samples,
        make_processor=make_processor,
        on_progress=on_progress,
        progress_task=_TASK_NAME,
        progress_paradigm=_PARADIGM_NAME,
        progress_prompt=prompt_name,
        progress_total=len(samples),
        progress_task_dir=task_dir,
    )

    return True
