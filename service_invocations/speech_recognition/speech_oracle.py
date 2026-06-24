from pathlib import Path

import pandas as pd

from service_invocations.core.cost_tracker import (
    make_attempt_recorder as _make_attempt_recorder,
    session_tracker,
)
from service_invocations.core.model_failover import run_with_failover
from service_invocations.core.oracle_utils import (
    extract_oracle as _extract_oracle,
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
    load_completed_rows,
    write_oracle,
)
from service_invocations.models import get_enabled_models, get_model_generator

_TASK_NAME = "speech_recognition"
_PARADIGM_NAME = "oracle"

_PROMPTS_ROOT = Path(__file__).parent / "prompts"
_PARADIGM = "oracle"

_DEFAULT_TASK_DIR = (
    Path.cwd() / "service_invocations" / "results" / "speech_recognition"
)


def generate_oracle_transcripts(
    edacc_data,
    prompt_name: str,
    use_existing: bool = False,
    results_dir: Path | None = None,
    models_path: Path | None = None,
    fresh_run: bool = False,
):
    task_dir = results_dir if results_dir is not None else _DEFAULT_TASK_DIR
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"
    task_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = _resolve_prompt_path(_PROMPTS_ROOT, _PARADIGM, prompt_name)

    enabled_models = get_enabled_models(models_path)
    if not enabled_models:
        print("Skipping LLM Oracle Transcript (no enabled models).")
        return {}

    fresh_run = _is_fresh_run_requested(fresh_run)
    if fresh_run:
        if use_existing:
            print(
                "[fresh] speech_oracle: fresh_run=True overrides use_existing — "
                "ignoring on-disk results."
            )
            use_existing = False
        removed = clear_completed_slice(task_dir, "oracle", prompt_name, enabled_models)
        if removed:
            print(
                f"[fresh] speech_oracle: cleared {removed} prior row(s) "
                f"for prompt='{prompt_name}'."
            )

    results_by_model: dict[str, pd.DataFrame] = {}

    samples = [
        {"id": row["id"], "audio": row["audio"]}
        for _, row in edacc_data.iterrows()
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
        done_ids: set[str] = (
            set()
            if fresh_run
            else set(load_completed_ids(task_dir, "oracle", prompt_name, model_name))
        )
        if done_ids:
            print(
                f"[resume] speech_oracle {model_name}: "
                f"{len(done_ids)} sample(s) already in CSV — will skip."
            )

        def process(sample: dict) -> dict | None:
            sample_id = sample["id"]
            audio_file = sample["audio"]
            id_key = _normalize_id(sample_id)
            if id_key in done_ids:
                return None
            print(f"LLM Oracle Transcript ({model_name}): {audio_file}")

            def _invoke_once():
                resp = generator(prompt, inputs={"audio": audio_file})
                print(resp.content)
                return resp, _extract_oracle(resp.content, key="transcript")

            on_attempt, total_cost = _make_attempt_recorder(
                tracker,
                task=_TASK_NAME,
                paradigm=_PARADIGM_NAME,
                model=model_name,
                sample_id=sample_id,
                # A usable oracle label is a non-nullish transcript.
                usable=lambda pair: not _is_nullish_output(pair[1]),
                models_path=models_path,
            )
            response, llm_oracle = _retry_until_valid(
                _invoke_once,
                validate=lambda pair: not _is_nullish_output(pair[1]),
                description=f"speech_oracle {model_name} sample={sample_id}",
                on_attempt=on_attempt,
            )
            cost = total_cost()
            done_ids.add(id_key)
            return {
                "id": id_key,
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
            full_slice = load_completed_rows(task_dir, "oracle", prompt_name, model_name)
            if not full_slice.empty:
                results_by_model[model_name] = full_slice
            elif rows:
                results_by_model[model_name] = pd.DataFrame(rows)
            # else: this model produced no oracle rows (e.g. it was permanently
            # unavailable). Leave it unregistered rather than storing an empty,
            # column-less pd.DataFrame([]) — that frame later crashed the metric
            # layer with KeyError: 'id'.

    if pending_models:
        run_with_failover(
            models=pending_models,
            samples=samples,
            make_processor=make_processor,
            on_progress=on_progress,
            progress_task=_TASK_NAME,
            progress_paradigm=_PARADIGM_NAME,
            progress_prompt=prompt_name,
            progress_total=len(samples),
            progress_task_dir=task_dir,
        )

    if len(enabled_models) > 1:
        return results_by_model
    if not results_by_model:
        return None
    return next(iter(results_by_model.values()))
