from pathlib import Path

import pandas as pd
import re

from service_invocations.core.config import get_model_entries, get_service_set, load_config
from registry.speech_recognition import run_speech_services
from service_invocations.speech_recognition.speech_judge import judge_transcripts
from service_invocations.speech_recognition.speech_oracle import generate_oracle_transcripts
from service_invocations.speech_recognition.wer import compute_wer_counts, compute_wer_summary


def _slugify_model(name: str) -> str:
    # Safe file suffix from model name (used in per-model output files).
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "model"


def _has_oracle_results(oracle_results) -> bool:
    if oracle_results is None:
        return False
    if isinstance(oracle_results, dict):
        return bool(oracle_results)
    empty_attr = getattr(oracle_results, "empty", None)
    if empty_attr is None:
        return True
    return not empty_attr


def run_speech_recognition(edacc_df, use_existing=False,
                           config_path: str | Path | None = None,
                           service_set_name: str = "speech_transcription",
                           models_path: str | Path | None = None,
                           model_set_name: str | None = "all"):
    if config_path is None:
        config_path = Path.cwd() / "config" / "services.yaml"
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"

    config = load_config(config_path)
    service_set = get_service_set(config, service_set_name)
    models_config = load_config(models_path)
    # model_set_name can be a top-level list set (multi-model), a single model under models:,
    # or "all" to include every model entry.
    model_entries = None
    if model_set_name is not None:
        model_entries = get_model_entries(models_config, model_set_name)

    # Service outputs go into the speech_recognition results folder.
    results = run_speech_services(
        edacc_df,
        service_set,
        use_existing=use_existing,
        config_path=config_path,
        results_dir=Path.cwd() / "service_invocations" / "results" / "speech_recognition",
    )

    results_dir = Path.cwd() / "service_invocations" / "results" / "speech_recognition"
    results_dir.mkdir(parents=True, exist_ok=True)

    print('--- LLMaaS ---')

    # Run LLM oracle transcripts; split outputs by model if model_set_name is provided.
    oracle_results = None
    if model_entries is None:
        print("--- Skipping LLM Oracle Transcript (no model set specified) ---")
    elif not model_entries:
        print("--- Skipping LLM Oracle Transcript (no enabled models in set) ---")
    else:
        oracle_results = generate_oracle_transcripts(
            edacc_df,
            use_existing=use_existing,
            results_dir=results_dir,
            model_entries=model_entries,
            return_by_model=model_entries is not None,
        )

    if results and _has_oracle_results(oracle_results):
        print('--- WER ---')
        # WER outputs are per-model when multiple models are used.
        if isinstance(oracle_results, dict):
            for model_name, model_oracle in oracle_results.items():
                model_slug = _slugify_model(model_name)
                wer_counts = compute_wer_counts(results, model_oracle, edacc_df)
                wer_counts.to_csv(results_dir / f"wer_counts__{model_slug}.csv", index=False)
                wer_summary = compute_wer_summary(wer_counts, list(results.keys()))
                wer_summary.to_csv(results_dir / f"wer_summary__{model_slug}.csv", index=False)
        else:
            wer_counts = compute_wer_counts(results, oracle_results, edacc_df)
            wer_counts.to_csv(results_dir / "wer_counts.csv", index=False)
            wer_summary = compute_wer_summary(wer_counts, list(results.keys()))
            wer_summary.to_csv(results_dir / "wer_summary.csv", index=False)
    elif results:
        print("--- Skipping WER (no LLM oracle results) ---")
    else:
        print("--- Skipping WER (no speech service results) ---")

    """
    if results:
        print('--- LLM Judging ---')
        # Judge transcripts using the same model set (if provided).
        judge_transcripts(
            results,
            edacc_df,
            use_existing=use_existing,
            config_path=config_path,
            model_entries=model_entries,
            results_dir=results_dir,
        )
    else:
        print("--- Skipping LLM Judging (no speech results) ---")
    """
        
    return results, model_entries


def _load_existing_speech_results(service_set, service_registry, results_dir: Path) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for entry in service_set:
        if not isinstance(entry, dict):
            raise ValueError("Each service entry must be a mapping.")
        if not entry.get("enabled", True):
            continue
        name = entry.get("name")
        if not name:
            raise ValueError("Each service entry must include a name.")
        registry_entry = service_registry.get(name)
        if registry_entry is None:
            raise KeyError(f"Unknown speech service: {name}")
        if registry_entry.get("task") != "stt":
            continue
        results_file = registry_entry.get("results_file")
        if not results_file:
            raise ValueError(f"service_registry entry '{name}' missing results_file.")
        results_path = results_dir / results_file
        if not results_path.exists():
            print(f"Skipping {name} (missing results file: {results_path})")
            continue
        results[name] = pd.read_csv(results_path)
    return results


def run_speech_oracle_and_judge_from_results(edacc_df, use_existing_oracle: bool = True,
                                             config_path: str | Path | None = None,
                                             service_set_name: str = "speech_transcription",
                                             models_path: str | Path | None = None,
                                             model_set_name: str | None = "all"):
    if config_path is None:
        config_path = Path.cwd() / "config" / "services.yaml"
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"

    config = load_config(config_path)
    service_set = get_service_set(config, service_set_name)
    service_registry = config.get("service_registry", {})
    if not isinstance(service_registry, dict):
        raise ValueError("service_registry must be a mapping.")

    results_dir = Path.cwd() / "service_invocations" / "results" / "speech_recognition"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = _load_existing_speech_results(service_set, service_registry, results_dir)
    if not results:
        raise ValueError("No existing speech results found to judge.")

    models_config = load_config(models_path)
    model_entries = None
    if model_set_name is not None:
        model_entries = get_model_entries(models_config, model_set_name)

    print('--- LLMaaS ---')
    oracle_results = None
    if model_entries is None:
        print("--- Skipping LLM Oracle Transcript (no model set specified) ---")
    elif not model_entries:
        print("--- Skipping LLM Oracle Transcript (no enabled models in set) ---")
    else:
        oracle_results = generate_oracle_transcripts(
            edacc_df,
            use_existing=use_existing_oracle,
            results_dir=results_dir,
            model_entries=model_entries,
            return_by_model=model_entries is not None,
        )

    """
    print('--- LLM Judging ---')
    judge_transcripts(
        results,
        edacc_df,
        use_existing=False,
        config_path=config_path,
        model_entries=model_entries,
        results_dir=results_dir,
        service_registry=service_registry,
    )
    """

    return results, model_entries, oracle_results


def prompt_speech_recognition_run(edacc_df):
    choice = input(
        "Speech Recognition Mode\n"
        "1.) Full pipeline (services + oracle + WER)\n"
        "2.) Oracle + Judge using existing service CSVs\n"
        "Select: "
    ).strip()

    if choice == "2":
        return run_speech_oracle_and_judge_from_results(edacc_df)
    return run_speech_recognition(edacc_df)
