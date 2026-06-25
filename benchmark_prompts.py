from pathlib import Path

import yaml

from service_invocations import invoke_speech_recognition as isr
from service_invocations import invoke_language_translation as ilt
from service_invocations import invoke_emotion_detection as ied
from service_invocations.core import run_context as rc
from service_invocations.core.cost_tracker import session_tracker
from service_invocations.core.oracle_utils import (
    is_fresh_run_requested,
    oracle_frame_usable,
)
from service_invocations.core.service_cost import (
    format_cost_summary,
    session_service_tracker,
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
from service_invocations.core.sds import compute_discrimination, save_discrimination
from service_invocations.invoke_speech_recognition import run_speech_recognition
from service_invocations.invoke_language_translation import run_language_translation
from service_invocations.invoke_emotion_detection import run_emotion_detection
from service_invocations.speech_recognition import speech_oracle, speech_judge, speech_human_loop
from service_invocations.speech_recognition.wer import compute_wer_rows, compute_wer_summary_rows
from service_invocations.language_translation import language_oracle, language_judge, language_human_loop
from service_invocations.language_translation.comet import compute_comet_rows, compute_comet_summary_rows
from service_invocations.emotion_detection import emotion_oracle, emotion_judge, emotion_human_loop
from service_invocations.emotion_detection.metrics import (
    compute_emotion_rows,
    compute_emotion_summary_rows,
    oracle_top_emotion,
)
from data_management.edacc import load_edacc
from data_management.en_fr import load_en_fr
from data_management.affectnet import load_affectnet

DEFAULT_NUM_SAMPLES = 5


def _load_prompt_selection(task_name: str, paradigm: str) -> dict[str, bool] | None:
    """Return the configured prompt allow-list for a ``task``/``paradigm``.

    ``None`` means "no restriction — run every prompt in the folder" (used when
    config/prompts.yaml is missing or the relevant section is absent/null). A
    dict maps prompt stems to whether they should run.
    """
    prompts_path = rc.config_path("prompts.yaml")
    if not prompts_path.exists():
        return None
    with prompts_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        return None
    task_cfg = config.get(task_name)
    if not isinstance(task_cfg, dict):
        return None
    selection = task_cfg.get(paradigm)
    if not isinstance(selection, dict):
        return None
    return {str(name): bool(enabled) for name, enabled in selection.items()}


def _list_prompts(prompts_root: Path, paradigm: str, task_name: str) -> list[str]:
    folder = prompts_root / paradigm
    if not folder.is_dir():
        return []
    available = sorted(p.stem for p in folder.glob("*.txt"))

    selection = _load_prompt_selection(task_name, paradigm)
    if selection is None:
        return available

    chosen = [stem for stem in available if selection.get(stem, False)]
    missing = [name for name, enabled in selection.items() if enabled and name not in available]
    if missing:
        print(f"[prompts] {task_name}/{paradigm}: configured prompt(s) not found, "
              f"skipping: {', '.join(sorted(missing))}")
    skipped = [stem for stem in available if stem not in chosen]
    if skipped:
        print(f"[prompts] {task_name}/{paradigm}: skipping {len(skipped)} "
              f"prompt(s) per config: {', '.join(skipped)}")
    return chosen


def _run_services_only(invoke_module, df, runner):
    saved = (invoke_module.ORACLE_PROMPT, invoke_module.JUDGE_PROMPT, invoke_module.HUMAN_LOOP_PROMPT)
    saved_quiet = getattr(invoke_module, "QUIET_SKIP_PROMPTS", False)
    invoke_module.ORACLE_PROMPT = ""
    invoke_module.JUDGE_PROMPT = ""
    invoke_module.HUMAN_LOOP_PROMPT = ""
    invoke_module.QUIET_SKIP_PROMPTS = True
    try:
        out = runner(df)
    finally:
        invoke_module.ORACLE_PROMPT, invoke_module.JUDGE_PROMPT, invoke_module.HUMAN_LOOP_PROMPT = saved
        invoke_module.QUIET_SKIP_PROMPTS = saved_quiet
    if isinstance(out, tuple):
        return out[0]
    return out


def _write_accuracy_for(task_dir, task_name, prompt, oracle_results, label_results, label_df,
                        compute_rows, compute_summary, oracle_transform=None):
    """Run the per-task accuracy helpers and persist into accuracy.csv / summary.

    The model's own oracle answer is also scored as a standalone pseudo-service
    (LLMaaS), so its quality-without-services lands in llmaas_accuracy.csv /
    llmaas_summary.csv and the LLM shows up as a peer bar in the accuracy plots.
    ``oracle_transform`` collapses an oracle cell into the form the metric reads
    from a service output (None for ASR/MT text; the top-1 label for FER).

    On a continued run, a (prompt, model) slice that is already fully present in
    accuracy.csv is skipped — so we don't reload the COMET checkpoint or re-score
    metrics that were computed in an earlier pass.
    """
    if not label_results or oracle_results is None:
        return
    services = list(label_results.keys())
    sample_ids = label_df["id"].tolist()

    def _persist(model_name, model_oracle):
        if not oracle_frame_usable(model_oracle):
            print(f"[skip] {task_name} metrics for {model_name}/{prompt}: "
                  f"no usable oracle labels (model produced no rows) — skipping.")
            return
        if rc.is_continue() and accuracy_slice_complete(task_dir, prompt, model_name, services, sample_ids):
            print(f"[resume] {task_name} metrics for {model_name}/{prompt} already complete — skipping.")
            return
        # Score the real services AND the oracle-as-a-service in one pass, then
        # split: accuracy.csv stays services-only (it drives best-service /
        # winner-consistency logic) while the LLMaaS rows go to their own files.
        augmented = {
            **label_results,
            LLMAAS_SERVICE: oracle_as_service(model_oracle, transform=oracle_transform),
        }
        per_sample = compute_rows(augmented, model_oracle, label_df)
        service_rows, llmaas_rows = split_llmaas_rows(per_sample)
        summary = compute_summary(service_rows, services)
        write_accuracy(task_dir, task_name, prompt, model_name, service_rows)
        write_accuracy_summary(task_dir, task_name, prompt, model_name, summary)
        if llmaas_rows:
            llmaas_summary = compute_summary(llmaas_rows, [LLMAAS_SERVICE])
            write_llmaas_accuracy(task_dir, task_name, prompt, model_name, llmaas_rows)
            write_llmaas_summary(task_dir, task_name, prompt, model_name, llmaas_summary)

    if isinstance(oracle_results, dict):
        for model_name, model_oracle in oracle_results.items():
            _persist(model_name, model_oracle)
    else:
        _persist("default", oracle_results)


def _benchmark_speech(edacc_df, service_results):
    if not service_results:
        print("--- Skipping speech prompts (no enabled services) ---")
        return
    prompts_root = speech_oracle._PROMPTS_ROOT

    for prompt in _list_prompts(prompts_root, "oracle", "speech_recognition"):
        print(f"=== [speech] oracle prompt: {prompt} ===")
        oracle_results = speech_oracle.generate_oracle_transcripts(
            edacc_df, prompt_name=prompt, results_dir=rc.task_results_dir("speech_recognition"),
        )
        if not service_results or oracle_results is None:
            continue
        print(f"=== [speech] WER for oracle prompt: {prompt} ===")
        _write_accuracy_for(
            rc.task_results_dir("speech_recognition"), "speech_recognition", prompt,
            oracle_results, service_results, edacc_df,
            compute_wer_rows, compute_wer_summary_rows,
        )

    if service_results:
        for prompt in _list_prompts(prompts_root, "judge", "speech_recognition"):
            print(f"=== [speech] judge prompt: {prompt} ===")
            speech_judge.judge_transcripts(
                service_results, edacc_df, prompt_name=prompt, results_dir=rc.task_results_dir("speech_recognition"),
            )

        for prompt in _list_prompts(prompts_root, "human-loop", "speech_recognition"):
            print(f"=== [speech] human-loop prompt: {prompt} ===")
            speech_human_loop.human_loop_transcripts(
                service_results, edacc_df, prompt_name=prompt, results_dir=rc.task_results_dir("speech_recognition"),
            )

        for prompt in _list_prompts(prompts_root, "human-loop-no-threshold", "speech_recognition"):
            print(f"=== [speech] human-loop-no-threshold prompt: {prompt} ===")
            speech_human_loop.human_loop_transcripts(
                service_results, edacc_df, prompt_name=prompt,
                paradigm="human-loop-no-threshold",
                results_dir=rc.task_results_dir("speech_recognition"),
            )


def _benchmark_language(europarl_df, service_results):
    if not service_results:
        print("--- Skipping translation prompts (no enabled services) ---")
        return
    prompts_root = language_oracle._PROMPTS_ROOT

    for prompt in _list_prompts(prompts_root, "oracle", "language_translation"):
        print(f"=== [language] oracle prompt: {prompt} ===")
        oracle_results = language_oracle.generate_oracle_translations(
            europarl_df, prompt_name=prompt, results_dir=rc.task_results_dir("language_translation"),
        )
        if not service_results or oracle_results is None:
            continue
        print(f"=== [language] COMET for oracle prompt: {prompt} ===")
        _write_accuracy_for(
            rc.task_results_dir("language_translation"), "language_translation", prompt,
            oracle_results, service_results, europarl_df,
            compute_comet_rows, compute_comet_summary_rows,
        )

    if service_results:
        for prompt in _list_prompts(prompts_root, "judge", "language_translation"):
            print(f"=== [language] judge prompt: {prompt} ===")
            language_judge.judge_translations(
                service_results, europarl_df, prompt_name=prompt, results_dir=rc.task_results_dir("language_translation"),
            )

        for prompt in _list_prompts(prompts_root, "human-loop", "language_translation"):
            print(f"=== [language] human-loop prompt: {prompt} ===")
            language_human_loop.human_loop_translations(
                service_results, europarl_df, prompt_name=prompt, results_dir=rc.task_results_dir("language_translation"),
            )

        for prompt in _list_prompts(prompts_root, "human-loop-no-threshold", "language_translation"):
            print(f"=== [language] human-loop-no-threshold prompt: {prompt} ===")
            language_human_loop.human_loop_translations(
                service_results, europarl_df, prompt_name=prompt,
                paradigm="human-loop-no-threshold",
                results_dir=rc.task_results_dir("language_translation"),
            )


def _benchmark_emotion(affectnet_df, service_results):
    if not service_results:
        print("--- Skipping emotion prompts (no enabled services) ---")
        return
    prompts_root = emotion_oracle._PROMPTS_ROOT

    for prompt in _list_prompts(prompts_root, "oracle", "emotion_detection"):
        print(f"=== [emotion] oracle prompt: {prompt} ===")
        oracle_results = emotion_oracle.generate_oracle_emotions(
            affectnet_df, prompt_name=prompt, results_dir=rc.task_results_dir("emotion_detection"),
        )
        if not service_results or oracle_results is None:
            continue
        print(f"=== [emotion] classification metrics for oracle prompt: {prompt} ===")
        _write_accuracy_for(
            rc.task_results_dir("emotion_detection"), "emotion_detection", prompt,
            oracle_results, service_results, affectnet_df,
            compute_emotion_rows, compute_emotion_summary_rows,
            oracle_transform=oracle_top_emotion,
        )

    if service_results:
        for prompt in _list_prompts(prompts_root, "judge", "emotion_detection"):
            print(f"=== [emotion] judge prompt: {prompt} ===")
            emotion_judge.judge_emotions(
                service_results, affectnet_df, prompt_name=prompt, results_dir=rc.task_results_dir("emotion_detection"),
            )

        for prompt in _list_prompts(prompts_root, "human-loop", "emotion_detection"):
            print(f"=== [emotion] human-loop prompt: {prompt} ===")
            emotion_human_loop.human_loop_emotions(
                service_results, affectnet_df, prompt_name=prompt, results_dir=rc.task_results_dir("emotion_detection"),
            )

        for prompt in _list_prompts(prompts_root, "human-loop-no-threshold", "emotion_detection"):
            print(f"=== [emotion] human-loop-no-threshold prompt: {prompt} ===")
            emotion_human_loop.human_loop_emotions(
                service_results, affectnet_df, prompt_name=prompt,
                paradigm="human-loop-no-threshold",
                results_dir=rc.task_results_dir("emotion_detection"),
            )


def _compute_plan(*tasks) -> dict:
    """Total model-sample jobs across the benchmark, for progress percentages.

    Each ``tasks`` entry is ``(task_name, prompts_root, df, has_results)``. A
    job is one (prompt, model, sample) unit; the total is the sum, over each
    task that will run, of ``samples × models × prompts`` (oracle + judge +
    human-loop prompts). Best-effort: returns an empty plan if the model
    config can't be read, so a missing total just omits the percentage.
    """
    try:
        from service_invocations.models import get_enabled_models
        models_path = rc.config_path("models.yaml")
        n_models = len(get_enabled_models(models_path))
    except Exception:
        n_models = 0
    if not n_models:
        return {}
    total = 0
    for task_name, prompts_root, df, has_results in tasks:
        if not has_results:
            continue
        n_prompts = sum(
            len(_list_prompts(prompts_root, paradigm, task_name))
            for paradigm in ("oracle", "judge", "human-loop", "human-loop-no-threshold")
        )
        total += len(df) * n_models * n_prompts
    if not total:
        return {}
    return {"total_samples": total, "num_models": n_models}


def _load_or_restore(name: str, loader, datasets: dict | None, banner: str):
    """Restore a continued run's samples, else draw fresh ones and persist them."""
    if datasets is not None:
        return datasets[name]
    print(banner)
    df = loader()
    if rc.active_run_dir() is not None:
        rc.save_samples(df, name=name)
    return df


def run_all_prompts(num_samples: int = DEFAULT_NUM_SAMPLES, randomize: bool = True,
                    seed: int | None = None, datasets: dict | None = None):
    # Order: MT -> ASR -> FER. FER now runs the two local libraries (FER,
    # DeepFace) alongside the cloud services; it runs last so the network-bound
    # tasks complete first.
    europarl_df = _load_or_restore(
        "language_translation",
        lambda: load_en_fr(num_samples, randomize=randomize, seed=seed),
        datasets, "--- Retrieving EuroParl data pairs ---",
    )
    edacc_df = _load_or_restore(
        "speech_recognition",
        lambda: load_edacc(num_samples, randomize=randomize, seed=seed),
        datasets, "--- Gathering viable EdAcc samples ---",
    )
    affectnet_df = _load_or_restore(
        "emotion_detection",
        lambda: load_affectnet(num_samples, randomize=randomize, seed=seed),
        datasets, "--- Retrieving AffectNet-7 samples ---",
    )

    print("=== Running language services (once) ===")
    language_results = _run_services_only(ilt, europarl_df, run_language_translation)
    print("=== Running speech services (once) ===")
    speech_results = _run_services_only(isr, edacc_df, run_speech_recognition)
    print("=== Running emotion services (once) ===")
    emotion_results = _run_services_only(ied, affectnet_df, run_emotion_detection)

    if language_results:
        print("=== [language] SDS ranking + Majority Voting baseline ===")
        task_dir = rc.task_results_dir("language_translation")
        sds_df = compute_discrimination(language_results, europarl_df["id"].tolist(), output_kind="text")
        save_discrimination(sds_df, task_dir / "sds")
        mv_df = majority_vote(language_results, europarl_df["id"].tolist(), output_kind="text")
        save_majority_voting(mv_df, task_dir / "majority_voting")

    if speech_results:
        print("=== [speech] SDS ranking + Majority Voting baseline ===")
        task_dir = rc.task_results_dir("speech_recognition")
        sds_df = compute_discrimination(speech_results, edacc_df["id"].tolist(), output_kind="text")
        save_discrimination(sds_df, task_dir / "sds")
        mv_df = majority_vote(speech_results, edacc_df["id"].tolist(), output_kind="text")
        save_majority_voting(mv_df, task_dir / "majority_voting")

    if emotion_results:
        print("=== [emotion] SDS ranking + Majority Voting baseline ===")
        task_dir = rc.task_results_dir("emotion_detection")
        sds_df = compute_discrimination(emotion_results, affectnet_df["id"].tolist(), output_kind="emotion")
        save_discrimination(sds_df, task_dir / "sds")
        mv_df = majority_vote(emotion_results, affectnet_df["id"].tolist(), output_kind="emotion")
        save_majority_voting(mv_df, task_dir / "majority_voting")

    # Register the planned scope so run_status.json can report a global
    # percentage as the per-slice progress is recorded.
    rc.set_plan(**_compute_plan(
        ("language_translation", language_oracle._PROMPTS_ROOT, europarl_df, bool(language_results)),
        ("speech_recognition", speech_oracle._PROMPTS_ROOT, edacc_df, bool(speech_results)),
        ("emotion_detection", emotion_oracle._PROMPTS_ROOT, affectnet_df, bool(emotion_results)),
    ))

    _seed_existing_costs()

    _benchmark_language(europarl_df, language_results)
    _flush_task_cost("language_translation")
    _benchmark_speech(edacc_df, speech_results)
    _flush_task_cost("speech_recognition")
    _benchmark_emotion(affectnet_df, emotion_results)
    _flush_task_cost("emotion_detection")

    run_dir = rc.active_run_dir()
    if run_dir is not None:
        cost_log = session_tracker().write(results_root=run_dir)
        if cost_log is not None:
            print(f"=== Benchmark LLM cost log: {cost_log} ===")
        svc_cost_log = session_service_tracker().write(results_root=run_dir)
        if svc_cost_log is not None:
            print(f"=== Benchmark service cost log: {svc_cost_log} ===")
    else:
        # No active run — avoid polluting the legacy results root with cost CSVs.
        print("=== Benchmark cost (no active run, not persisted) ===")
    print(format_cost_summary(scope="benchmark"))

    replot_all()


def _seed_existing_costs() -> None:
    """Preload prior per-task cost CSVs into the session trackers on resume.

    A resumed benchmark skips already-completed samples, so their costs never
    re-enter the trackers. Without seeding, the per-task ``_flush_task_cost``
    and the run-level write at the end would replace each task slice with only
    this run's freshly-invoked subset and under-report the cumulative cost.
    No-op on a fresh run or when no prior CSV exists.
    """
    if not rc.is_continue() or is_fresh_run_requested():
        return
    for task_name in _TASK_NAMES:
        task_dir = rc.task_results_dir(task_name)
        session_tracker().seed_from_csv(task_dir / "cost.csv", task_filter=task_name)
        session_service_tracker().seed_from_csv(
            task_dir / "service_cost.csv", task_filter=task_name
        )


def _flush_task_cost(task_name: str) -> None:
    """Re-write ``<task_dir>/cost.csv`` so it reflects every tracked entry for the task.

    Called after each ``_benchmark_*`` block so the per-task CSV captures the
    prompt-side LLM calls (oracle/judge/human-loop), not just the service-side
    costs that ``_run_services_only`` flushed earlier.
    """
    task_dir = rc.task_results_dir(task_name)
    if not task_dir.exists():
        return
    session_tracker().write(results_root=task_dir, task_filter=task_name)


_TASK_NAMES = ("language_translation", "speech_recognition", "emotion_detection")


def replot_all(only: list[str] | None = None) -> None:
    """Regenerate per-paradigm and summary plots from existing CSVs.

    Reads only the consolidated CSVs already on disk (oracle.csv, judge.csv,
    human_loop.csv, accuracy.csv, cost.csv, sds/, majority_voting/) — no LLM
    calls. Task directories resolve through run_context, so this targets the
    active timestamped run when one exists and the legacy location otherwise.
    Pass ``only`` to limit to a subset of task names.
    """
    print("=== Plots (all tasks) ===")
    for task_name in _TASK_NAMES:
        if only is not None and task_name not in only:
            continue
        task_dir = rc.task_results_dir(task_name)
        if not task_dir.exists():
            print(f"  - skip {task_name}: {task_dir} does not exist")
            continue
        print(f"  - {task_name}")
        plot_all_for_task(task_dir, task_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the prompt benchmark or regenerate plots only.")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip all LLM/service calls; just rebuild plots from existing results CSVs.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=list(_TASK_NAMES),
        help="Limit --plots-only to one task (may be repeated). Ignored without --plots-only.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Target a specific run directory for --plots-only (absolute, or "
             "relative to service_invocations/results/, e.g. "
             "'2026-06-02/22-02-36_benchmark'). If omitted, uses the legacy "
             "results/<task>/ location.",
    )
    args = parser.parse_args()

    if args.plots_only:
        only = args.task
        if args.run is not None:
            run_dir = Path(args.run)
            if not run_dir.is_absolute():
                run_dir = rc.results_root() / run_dir
            info = rc.attach_run(run_dir)
            print(f"=== Replotting run: {info.display} ({info.dir}) ===")
            if not info.subdir_by_task and only is None:
                only = [info.label] if info.label in _TASK_NAMES else None
        replot_all(only=only)
    else:
        # Start a timestamped benchmark run so every artifact (including
        # cost.csv) lands under results/<date>/<time>_benchmark/ instead of
        # leaking into the legacy results/ root.
        started_here = False
        if rc.active_run_dir() is None:
            info = rc.start_run("benchmark", subdir_by_task=True)
            started_here = True
            print(f"=== Starting benchmark run: {info.display} ({info.dir}) ===")
        try:
            run_all_prompts()
            if started_here:
                rc.mark_finished()
        finally:
            if started_here:
                rc.end_run()
