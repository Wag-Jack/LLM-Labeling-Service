from pathlib import Path

from service_invocations.core.config import get_model_set, get_service_set, load_config
from registry.speech_recognition import run_speech_services
from service_invocations.speech_recognition.speech_judge import judge_transcripts
from service_invocations.speech_recognition.speech_oracle import generate_oracle_transcripts
from service_invocations.speech_recognition.wer import compute_wer_counts, compute_wer_summary

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
    model_set = None
    if model_set_name is not None:
        model_set = get_model_set(models_config, model_set_name)

    results = run_speech_services(
        edacc_df,
        service_set,
        use_existing=use_existing,
        config_path=config_path,
    )

    results_dir = Path.cwd() / "service_invocations" / "results"

    # LLMaaS
    if results:
        print('--- LLMaaS ---')

        oracle_results = generate_oracle_transcripts(edacc_df, use_existing=use_existing)

        print('--- WER ---')
        wer_counts = compute_wer_counts(results, oracle_results, edacc_df)
        wer_counts.to_csv(results_dir / "wer_counts.csv", index=False)
        wer_summary = compute_wer_summary(wer_counts, list(results.keys()))
        wer_summary.to_csv(results_dir / "wer_summary.csv", index=False)

    """
    if results:
        print('--- LLM Judging ---')
        judge_transcripts(
            results,
            edacc_df,
            use_existing=use_existing,
            config_path=config_path,
        )
    else:
        print("--- Skipping LLM Judging (no speech results) ---")
    """
        
    return results, model_set
