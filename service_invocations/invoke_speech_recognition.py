from pathlib import Path

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


def run_speech_recognition(edacc_df, use_existing=False,
                           config_path: str | Path | None = None,
                           service_set_name: str = "speech_transcription",
                           models_path: str | Path | None = None,
                           model_set_name: str | None = None):
    if config_path is None:
        config_path = Path.cwd() / "config" / "services.yaml"
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"

    config = load_config(config_path)
    service_set = get_service_set(config, service_set_name)
    models_config = load_config(models_path)
    # model_set_name can be a top-level list set (multi-model) or a single model under models:.
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

    # LLMaaS
    if results:
        print('--- LLMaaS ---')

        # Run LLM oracle transcripts; split outputs by model if model_set_name is provided.
        oracle_results = generate_oracle_transcripts(
            edacc_df,
            use_existing=use_existing,
            results_dir=results_dir,
            model_entries=model_entries,
            return_by_model=model_entries is not None,
        )

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
