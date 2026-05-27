from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pandas as pd
import yaml

from service_invocations.core.cost_tracker import session_tracker
from service_invocations.core.majority_voting import majority_vote, save_majority_voting
from service_invocations.core.plotting import plot_all_for_task
from service_invocations.core.results_io import write_accuracy, write_accuracy_summary
from service_invocations.core.sds import (
    compute_discrimination,
    filter_dataset,
    filter_service_results,
    save_discrimination,
    select_top_k,
)
from service_invocations.emotion_detection.emotion_oracle import generate_oracle_emotions
from service_invocations.emotion_detection.emotion_judge import judge_emotions
from service_invocations.emotion_detection.emotion_human_loop import human_loop_emotions
from service_invocations.emotion_detection.metrics import (
    compute_emotion_rows,
    compute_emotion_summary_rows,
)


# Set to a prompt filename (without .txt) under prompts/<paradigm>/ to run that paradigm.
# Set to "" to skip the paradigm.
ORACLE_PROMPT = "default"
JUDGE_PROMPT = "default"
HUMAN_LOOP_PROMPT = "default"
QUIET_SKIP_PROMPTS = False
SDS_TOP_K: int | None = None
RUN_MAJORITY_VOTING = True

_TASK_NAME = "emotion_detection"
_OUTPUT_KIND = "emotion"
_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "emotion_detection"
_SERVICES_DIR = _RESULTS_DIR / "services"
_SDS_DIR = _RESULTS_DIR / "sds"
_MV_DIR = _RESULTS_DIR / "majority_voting"

_ORACLE_DIR = _RESULTS_DIR
_JUDGE_DIR = _RESULTS_DIR
_HUMAN_LOOP_DIR = _RESULTS_DIR
_METRICS_DIR = _RESULTS_DIR
_COST_DIR = _RESULTS_DIR


def _has_oracle_results(oracle_results) -> bool:
    if oracle_results is None:
        return False
    if isinstance(oracle_results, dict):
        return bool(oracle_results)
    empty_attr = getattr(oracle_results, "empty", None)
    if empty_attr is None:
        return True
    return not empty_attr


def _load_enabled_entries(config_path: Path, task_name: str) -> list[str]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("services.yaml root must be a mapping.")

    task_cfg = config.get(task_name, {})
    if not isinstance(task_cfg, dict):
        raise ValueError(f"services.yaml '{task_name}' must be a mapping.")

    enabled = []
    for name, entry in task_cfg.items():
        if isinstance(entry, dict) and entry.get("enabled", False):
            enabled.append(name)
    return enabled


def _write_emotion_outputs(label_results, oracle_results, label_df, prompt_name: str, model_name: str) -> None:
    per_sample = compute_emotion_rows(label_results, oracle_results, label_df)
    summary = compute_emotion_summary_rows(per_sample, list(label_results.keys()))
    write_accuracy(_RESULTS_DIR, _TASK_NAME, prompt_name, model_name, per_sample)
    write_accuracy_summary(_RESULTS_DIR, _TASK_NAME, prompt_name, model_name, summary)


def run_emotion_detection(
    vea_df: pd.DataFrame,
    services_path: Path | None = None,
    models_path: Path | None = None,
):
    if services_path is None:
        services_path = Path.cwd() / "config" / "services.yaml"
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"

    if vea_df is None:
        raise ValueError("vea_df is required. Load data in main before invoking.")

    enabled_services = _load_enabled_entries(services_path, _TASK_NAME)
    if not enabled_services:
        print("--- Skipping emotion services (none enabled) ---")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _SERVICES_DIR.mkdir(parents=True, exist_ok=True)
    _SDS_DIR.mkdir(parents=True, exist_ok=True)
    _MV_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    for service_name in enabled_services:
        print(f"--- {service_name} ---")
        module = import_module(
            f"service_invocations.emotion_detection.services.{service_name}"
        )
        runner = getattr(module, "run", None)
        if runner is None or not callable(runner):
            raise AttributeError(
                f"Service script '{service_name}' must define a run(...) function."
            )
        results[service_name] = runner(vea_df)

    label_df = vea_df
    label_results = results
    if results:
        print("--- SDS (sample-based discriminatory sampling) ---")
        discrimination = compute_discrimination(
            results, vea_df["id"].tolist(), output_kind=_OUTPUT_KIND
        )
        save_discrimination(discrimination, _SDS_DIR)
        if SDS_TOP_K is not None and SDS_TOP_K > 0 and not discrimination.empty:
            top_ids = select_top_k(discrimination, SDS_TOP_K)
            print(f"--- SDS: restricting downstream labeling to top {len(top_ids)} samples ---")
            label_df = filter_dataset(vea_df, id_column="id", keep_ids=top_ids)
            label_results = filter_service_results(results, top_ids)

        if RUN_MAJORITY_VOTING:
            print("--- Majority Voting (service-output baseline oracle) ---")
            mv = majority_vote(
                results, vea_df["id"].tolist(), output_kind=_OUTPUT_KIND
            )
            save_majority_voting(mv, _MV_DIR)

    oracle_results = None
    if ORACLE_PROMPT:
        print(f"--- LLM Oracle Emotion (prompt: {ORACLE_PROMPT}) ---")
        oracle_results = generate_oracle_emotions(
            label_df,
            prompt_name=ORACLE_PROMPT,
            results_dir=_RESULTS_DIR,
            models_path=models_path,
        )
    elif not QUIET_SKIP_PROMPTS:
        print("--- Skipping LLM Oracle Emotion (ORACLE_PROMPT is empty) ---")

    if label_results and _has_oracle_results(oracle_results):
        print("--- Classification Metrics (F1 / precision / recall) ---")
        if isinstance(oracle_results, dict):
            for model_name, model_oracle in oracle_results.items():
                _write_emotion_outputs(label_results, model_oracle, label_df, ORACLE_PROMPT, model_name)
        else:
            _write_emotion_outputs(label_results, oracle_results, label_df, ORACLE_PROMPT, "default")
    elif label_results:
        print("--- Skipping Classification Metrics (no LLM oracle results) ---")
    else:
        print("--- Skipping Classification Metrics (no emotion service results) ---")

    if label_results and JUDGE_PROMPT:
        print(f"--- LLM Judging (prompt: {JUDGE_PROMPT}) ---")
        judge_emotions(
            label_results,
            label_df,
            prompt_name=JUDGE_PROMPT,
            results_dir=_RESULTS_DIR,
            services_path=services_path,
            models_path=models_path,
        )
    elif not JUDGE_PROMPT and not QUIET_SKIP_PROMPTS:
        print("--- Skipping LLM Judging (JUDGE_PROMPT is empty) ---")

    if label_results and HUMAN_LOOP_PROMPT:
        print(f"--- LLM Human-Loop (prompt: {HUMAN_LOOP_PROMPT}) ---")
        human_loop_emotions(
            label_results,
            label_df,
            prompt_name=HUMAN_LOOP_PROMPT,
            results_dir=_RESULTS_DIR,
            services_path=services_path,
            models_path=models_path,
        )
    elif not HUMAN_LOOP_PROMPT and not QUIET_SKIP_PROMPTS:
        print("--- Skipping LLM Human-Loop (HUMAN_LOOP_PROMPT is empty) ---")

    cost_log_path = session_tracker().write(results_root=_RESULTS_DIR, task_filter=_TASK_NAME)
    if cost_log_path is not None:
        print(f"--- Cost log: {cost_log_path} (session total: ${session_tracker().total_usd():.4f}) ---")

    print("--- Plots ---")
    plot_all_for_task(_RESULTS_DIR, _TASK_NAME)

    return results, oracle_results
