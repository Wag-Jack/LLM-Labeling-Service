from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import yaml

from service_invocations.core.cost_tracker import compute_cost, session_tracker
from service_invocations.core.model_failover import run_with_failover
from service_invocations.core.oracle_utils import (
    is_fresh_run_requested as _is_fresh_run_requested,
    is_nullish_output as _is_nullish_output,
    load_prompt as _load_prompt,
    normalize_id as _normalize_id,
    resolve_prompt_path as _resolve_prompt_path,
    retry_until_valid as _retry_until_valid,
)
from service_invocations.core.results_io import (
    clear_completed_slice,
    load_completed_ids,
    write_human_loop,
)
from service_invocations.models import get_enabled_models, get_model_generator

_PARADIGM_NAME = "human_loop"
_TASK_NAME = "speech_recognition"
_PROMPTS_ROOT = Path(__file__).parent / "prompts"
_PARADIGM = "human-loop"
_CONFIDENCE_THRESHOLD = 0.7
_GROUND_TRUTH_COLUMN = "text"

_DEFAULT_TASK_DIR = (
    Path.cwd() / "service_invocations" / "results" / "speech_recognition"
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


def human_loop_transcripts(
    results_by_service: dict[str, pd.DataFrame],
    edacc_data: pd.DataFrame,
    prompt_name: str,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
    results_dir: Path | None = None,
    services_path: Path | None = None,
    models_path: Path | None = None,
    task_name: str = _TASK_NAME,
    fresh_run: bool = False,
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

    fresh_run = _is_fresh_run_requested(fresh_run)
    if fresh_run:
        removed = clear_completed_slice(task_dir, "human_loop", prompt_name, enabled_models)
        if removed:
            print(
                f"[fresh] speech_human_loop: cleared {removed} prior row(s) "
                f"for prompt='{prompt_name}'."
            )

    services = {
        name: df for name, df in results_by_service.items() if name in enabled_services
    }
    if not services:
        print("--- Skipping LLM Human-Loop (no enabled service results) ---")
        return None

    transcripts_by_service = {}
    for name, df in services.items():
        if "id" not in df.columns or "service_output" not in df.columns:
            raise ValueError(f"Missing required columns for {name}.")
        transcripts_by_service[name] = dict(
            zip(df["id"].map(_normalize_id), df["service_output"])
        )

    ground_truth = dict(
        zip(edacc_data["id"].map(_normalize_id), edacc_data[_GROUND_TRUTH_COLUMN])
    )
    wav_files = edacc_data["audio"].tolist()
    ids = edacc_data["id"].tolist()

    samples = [
        {"id": sample_id, "audio": wav}
        for sample_id, wav in zip(ids, wav_files)
    ]

    def make_processor(model_name: str):
        generator = get_model_generator(model_name, models_path=models_path)
        tracker = session_tracker()
        done_ids: set[str] = (
            set()
            if fresh_run
            else set(load_completed_ids(task_dir, "human_loop", prompt_name, model_name))
        )
        if done_ids:
            print(
                f"[resume] speech_human_loop {model_name}: "
                f"{len(done_ids)} sample(s) already in CSV — will skip."
            )

        def process(sample: dict) -> dict | None:
            sample_id = sample["id"]
            wav = sample["audio"]
            id_key = _normalize_id(sample_id)
            if id_key in done_ids:
                print(
                    f"[resume] Skip speech_human_loop ({model_name}) "
                    f"sample={sample_id} (already done)."
                )
                return None
            print(f"LLM Human-Loop ({model_name}): {wav}")

            service_blocks = "\n".join(
                f"{name}: {transcripts_by_service[name].get(id_key, '')}"
                for name in services.keys()
            )
            prompt = _load_prompt(prompt_path, service_blocks=service_blocks)

            def _invoke_once():
                resp = generator(prompt, inputs={"audio": wav})
                print(resp.content)
                try:
                    parsed = json.loads(resp.content)
                    if not isinstance(parsed, dict):
                        parsed = {}
                except (json.JSONDecodeError, TypeError):
                    parsed = {}
                return resp, parsed

            response, llm_output = _retry_until_valid(
                _invoke_once,
                validate=lambda pair: not _is_nullish_output(pair[1].get("llm_transcript")),
                description=f"speech_human_loop {model_name} sample={sample_id}",
            )
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
            scores = llm_output.get("scores", {})
            if not isinstance(scores, dict):
                scores = {}
            merged_scores = {**default_scores, **scores}

            confidence = llm_output.get("confidence")
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = 0.0

            llm_transcript = llm_output.get("llm_transcript", "n/a")
            fallback_used = confidence_value < confidence_threshold
            human_label_used = ""
            if fallback_used:
                human_label_used = ground_truth.get(id_key, "") or ""
                llm_transcript = human_label_used or llm_transcript

            winner = llm_output.get("winner") if isinstance(llm_output, dict) else None
            if winner is not None and winner not in services:
                winner = None

            done_ids.add(id_key)
            return {
                "id": id_key,
                "llm_label": llm_transcript,
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
