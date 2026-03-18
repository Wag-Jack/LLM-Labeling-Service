from pathlib import Path
import re

from service_invocations.core.config import get_model_entries, get_service_set, load_config
from registry.language_translation import run_translation_services
from service_invocations.language_translation.comet import compute_comet_scores, compute_comet_summary
from service_invocations.language_translation.language_oracle import generate_oracle_translations

def _slugify_model(name: str) -> str:
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


def run_language_translation(europarl_df, use_existing=False,
                             config_path: str | Path | None = None,
                             service_set_name: str = "language_translation",
                             models_path: str | Path | None = None,
                             model_set_name: str | None = None):
    if config_path is None:
        config_path = Path.cwd() / "config" / "services.yaml"
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"

    config = load_config(config_path)
    service_set = get_service_set(config, service_set_name)
    models_config = load_config(models_path)
    model_entries = None
    if model_set_name is not None:
        model_entries = get_model_entries(models_config, model_set_name)

    results = run_translation_services(
        europarl_df,
        service_set,
        use_existing=use_existing,
        config_path=config_path,
        results_dir=Path.cwd() / "service_invocations" / "results" / "language_translation",
    )

    results_dir = Path.cwd() / "service_invocations" / "results" / "language_translation"
    results_dir.mkdir(parents=True, exist_ok=True)

    oracle_results = None
    if model_entries is None:
        print("--- Skipping LLM Oracle Translation (no model set specified) ---")
    elif not model_entries:
        print("--- Skipping LLM Oracle Translation (no enabled models in set) ---")
    else:
        print('--- LLM Oracle Translation ---')
        oracle_results = generate_oracle_translations(
            europarl_df,
            use_existing=use_existing,
            results_dir=results_dir,
            model_entries=model_entries,
            return_by_model=model_entries is not None,
        )

    if results and _has_oracle_results(oracle_results):
        print('--- COMET ---')
        if isinstance(oracle_results, dict):
            for model_name, model_oracle in oracle_results.items():
                model_slug = _slugify_model(model_name)
                comet_scores = compute_comet_scores(results, model_oracle, europarl_df)
                comet_scores.to_csv(results_dir / f"comet_scores__{model_slug}.csv", index=False)
                comet_summary = compute_comet_summary(comet_scores, list(results.keys()))
                comet_summary.to_csv(results_dir / f"comet_summary__{model_slug}.csv", index=False)
        else:
            comet_scores = compute_comet_scores(results, oracle_results, europarl_df)
            comet_scores.to_csv(results_dir / "comet_scores.csv", index=False)
            comet_summary = compute_comet_summary(comet_scores, list(results.keys()))
            comet_summary.to_csv(results_dir / "comet_summary.csv", index=False)
    elif results:
        print("--- Skipping COMET (no LLM oracle results) ---")
    else:
        print("--- Skipping COMET (no translation service results) ---")

    return results, model_entries
