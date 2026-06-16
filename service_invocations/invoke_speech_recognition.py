from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pandas as pd
import yaml

from service_invocations.core import run_context as rc
from service_invocations.core.cost_tracker import session_tracker
from service_invocations.core.service_cost import (
    format_cost_summary,
    session_service_tracker,
)
from service_invocations.core.failure_report import (
    compute_failure_report,
    print_failure_summary,
    save_failure_report,
)
from service_invocations.core.majority_voting import majority_vote, save_majority_voting
from service_invocations.core.plotting import plot_all_for_task
from service_invocations.core.results_io import (
    accuracy_slice_complete,
    write_accuracy,
    write_accuracy_summary,
    write_llmaas_accuracy,
    write_llmaas_summary,
)
from service_invocations.core.llmaas import (
    LLMAAS_SERVICE,
    oracle_as_service,
    split_llmaas_rows,
)
from service_invocations.core.terminal_mirror import mirrored_run
from service_invocations.core.sds import (
    compute_discrimination,
    filter_dataset,
    filter_service_results,
    save_discrimination,
    select_top_k,
)
from service_invocations.speech_recognition.speech_oracle import generate_oracle_transcripts
from service_invocations.speech_recognition.speech_judge import judge_transcripts
from service_invocations.speech_recognition.speech_human_loop import human_loop_transcripts
from service_invocations.speech_recognition.wer import compute_wer_rows, compute_wer_summary_rows


# Set to a prompt filename (without .txt) under prompts/<paradigm>/ to run that paradigm.
# Set to "" to skip the paradigm.
ORACLE_PROMPT = "asr_service_medium"
JUDGE_PROMPT = "asr_judge_medium"
HUMAN_LOOP_PROMPT = "asr_hitl_medium"
QUIET_SKIP_PROMPTS = False
SDS_TOP_K: int | None = None
RUN_MAJORITY_VOTING = True

_TASK_NAME = "speech_recognition"
_OUTPUT_KIND = "text"

# Legacy fixed results location, used only by benchmark_prompts.py (which runs
# every prompt into one shared folder and does not use timestamped runs).
# Interactive runs resolve their directory through run_context instead.
_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / _TASK_NAME
_SDS_DIR = _RESULTS_DIR / "sds"
_MV_DIR = _RESULTS_DIR / "majority_voting"


def _completed_service(results_path: Path, expected_count: int):
    """Return an existing, fully-populated service CSV (for resume), else None."""
    if not results_path.exists():
        return None
    try:
        df = pd.read_csv(results_path)
    except (pd.errors.EmptyDataError, OSError):
        return None
    return df if len(df) >= expected_count and expected_count > 0 else None


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


def _write_wer_outputs(results_dir, label_results, oracle_results, label_df, prompt_name: str, model_name: str) -> None:
    if rc.is_continue() and accuracy_slice_complete(
        results_dir, prompt_name, model_name,
        list(label_results.keys()), label_df["id"].tolist(),
    ):
        print(f"[resume] WER for {model_name}/{prompt_name} already complete — skipping.")
        return
    augmented = {**label_results, LLMAAS_SERVICE: oracle_as_service(oracle_results)}
    per_sample = compute_wer_rows(augmented, oracle_results, label_df)
    service_rows, llmaas_rows = split_llmaas_rows(per_sample)
    summary = compute_wer_summary_rows(service_rows, list(label_results.keys()))
    write_accuracy(results_dir, _TASK_NAME, prompt_name, model_name, service_rows)
    write_accuracy_summary(results_dir, _TASK_NAME, prompt_name, model_name, summary)

    llmaas_summary = compute_wer_summary_rows(llmaas_rows, [LLMAAS_SERVICE])
    write_llmaas_accuracy(results_dir, _TASK_NAME, prompt_name, model_name, llmaas_rows)
    write_llmaas_summary(results_dir, _TASK_NAME, prompt_name, model_name, llmaas_summary)


@mirrored_run(_TASK_NAME)
def run_speech_recognition(
    edacc_df: pd.DataFrame,
    services_path: Path | None = None,
    models_path: Path | None = None,
):
    if services_path is None:
        services_path = Path.cwd() / "config" / "services.yaml"
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"

    if edacc_df is None:
        raise ValueError("edacc_df is required. Load data in main before invoking.")

    enabled_services = _load_enabled_entries(services_path, _TASK_NAME)
    if not enabled_services:
        print("--- Skipping speech services (none enabled) ---")

    results_dir = rc.task_results_dir(_TASK_NAME)
    services_dir = results_dir / "services"
    sds_dir = results_dir / "sds"
    mv_dir = results_dir / "majority_voting"
    results_dir.mkdir(parents=True, exist_ok=True)
    services_dir.mkdir(parents=True, exist_ok=True)
    sds_dir.mkdir(parents=True, exist_ok=True)
    mv_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- Results: {results_dir} ---")

    expected_count = len(edacc_df)
    resuming = rc.is_continue()
    results: dict[str, pd.DataFrame] = {}
    for service_name in enabled_services:
        print(f"--- {service_name} ---")
        module = import_module(
            f"service_invocations.speech_recognition.services.{service_name}"
        )
        runner = getattr(module, "run", None)
        if runner is None or not callable(runner):
            raise AttributeError(
                f"Service script '{service_name}' must define a run(...) function."
            )
        results_file = getattr(module, "RESULTS_FILE", f"{service_name}.csv")
        results_path = services_dir / results_file
        if resuming:
            done = _completed_service(results_path, expected_count)
            if done is not None:
                print(f"[resume] {service_name}: already complete — skipping.")
                results[service_name] = done
                continue
        results[service_name] = runner(edacc_df, results_path=results_path)

    if results:
        print("--- Service Failure Report ---")
        failure_report = compute_failure_report(results, _TASK_NAME, _OUTPUT_KIND)
        save_failure_report(failure_report, results_dir)
        print_failure_summary(failure_report, _TASK_NAME)

    label_df = edacc_df
    label_results = results
    if results:
        print("--- SDS (sample-based discriminatory sampling) ---")
        discrimination = compute_discrimination(
            results, edacc_df["id"].tolist(), output_kind=_OUTPUT_KIND
        )
        save_discrimination(discrimination, sds_dir)
        if SDS_TOP_K is not None and SDS_TOP_K > 0 and not discrimination.empty:
            top_ids = select_top_k(discrimination, SDS_TOP_K)
            print(f"--- SDS: restricting downstream labeling to top {len(top_ids)} samples ---")
            label_df = filter_dataset(edacc_df, id_column="id", keep_ids=top_ids)
            label_results = filter_service_results(results, top_ids)

        if RUN_MAJORITY_VOTING:
            print("--- Majority Voting (service-output baseline oracle) ---")
            mv = majority_vote(
                results, edacc_df["id"].tolist(), output_kind=_OUTPUT_KIND
            )
            save_majority_voting(mv, mv_dir)

    oracle_results = None
    if ORACLE_PROMPT:
        print(f"--- LLM Oracle (prompt: {ORACLE_PROMPT}) ---")
        oracle_results = generate_oracle_transcripts(
            label_df,
            prompt_name=ORACLE_PROMPT,
            results_dir=results_dir,
            models_path=models_path,
        )
    elif not QUIET_SKIP_PROMPTS:
        print("--- Skipping LLM Oracle (ORACLE_PROMPT is empty) ---")

    if label_results and _has_oracle_results(oracle_results):
        print("--- WER ---")
        if isinstance(oracle_results, dict):
            for model_name, model_oracle in oracle_results.items():
                _write_wer_outputs(results_dir, label_results, model_oracle, label_df, ORACLE_PROMPT, model_name)
        else:
            _write_wer_outputs(results_dir, label_results, oracle_results, label_df, ORACLE_PROMPT, "default")
    elif label_results:
        print("--- Skipping WER (no LLM oracle results) ---")
    else:
        print("--- Skipping WER (no speech service results) ---")

    if label_results and JUDGE_PROMPT:
        print(f"--- LLM Judging (prompt: {JUDGE_PROMPT}) ---")
        judge_transcripts(
            label_results,
            label_df,
            prompt_name=JUDGE_PROMPT,
            results_dir=results_dir,
            services_path=services_path,
            models_path=models_path,
        )
    elif not JUDGE_PROMPT and not QUIET_SKIP_PROMPTS:
        print("--- Skipping LLM Judging (JUDGE_PROMPT is empty) ---")

    if label_results and HUMAN_LOOP_PROMPT:
        print(f"--- LLM Human-Loop (prompt: {HUMAN_LOOP_PROMPT}) ---")
        human_loop_transcripts(
            label_results,
            label_df,
            prompt_name=HUMAN_LOOP_PROMPT,
            results_dir=results_dir,
            services_path=services_path,
            models_path=models_path,
        )
    elif not HUMAN_LOOP_PROMPT and not QUIET_SKIP_PROMPTS:
        print("--- Skipping LLM Human-Loop (HUMAN_LOOP_PROMPT is empty) ---")

    cost_log_path = session_tracker().write(results_root=results_dir, task_filter=_TASK_NAME)
    if cost_log_path is not None:
        print(f"--- LLM cost log: {cost_log_path} ---")
    svc_cost_path = session_service_tracker().write(results_root=results_dir, task_filter=_TASK_NAME)
    if svc_cost_path is not None:
        print(f"--- Service cost log: {svc_cost_path} ---")
    print(format_cost_summary(scope=_TASK_NAME))

    print("--- Plots ---")
    plot_all_for_task(results_dir, _TASK_NAME)

    return results, oracle_results
