from pathlib import Path

from service_invocations import invoke_speech_recognition as isr
from service_invocations import invoke_language_translation as ilt
from service_invocations import invoke_emotion_detection as ied
from service_invocations.core import run_context as rc
from service_invocations.core.cost_tracker import session_tracker
from service_invocations.core.majority_voting import majority_vote, save_majority_voting
from service_invocations.core.plotting import plot_all_for_task
from service_invocations.core.results_io import (
    accuracy_slice_complete,
    write_accuracy,
    write_accuracy_summary,
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
)
from data_management.edacc import load_edacc
from data_management.en_fr import load_en_fr
from data_management.vea import load_vea

DEFAULT_NUM_SAMPLES = 5


def _list_prompts(prompts_root: Path, paradigm: str) -> list[str]:
    folder = prompts_root / paradigm
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.txt"))


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


def _write_accuracy_for(task_dir, task_name, prompt, oracle_results, label_results, label_df, compute_rows, compute_summary):
    """Run the per-task accuracy helpers and persist into accuracy.csv / summary.

    On a continued run, a (prompt, model) slice that is already fully present in
    accuracy.csv is skipped — so we don't reload the COMET checkpoint or re-score
    metrics that were computed in an earlier pass.
    """
    if not label_results or oracle_results is None:
        return
    services = list(label_results.keys())
    sample_ids = label_df["id"].tolist()

    def _persist(model_name, model_oracle):
        if rc.is_continue() and accuracy_slice_complete(task_dir, prompt, model_name, services, sample_ids):
            print(f"[resume] {task_name} metrics for {model_name}/{prompt} already complete — skipping.")
            return
        per_sample = compute_rows(label_results, model_oracle, label_df)
        summary = compute_summary(per_sample, services)
        write_accuracy(task_dir, task_name, prompt, model_name, per_sample)
        write_accuracy_summary(task_dir, task_name, prompt, model_name, summary)

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

    for prompt in _list_prompts(prompts_root, "oracle"):
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
        for prompt in _list_prompts(prompts_root, "judge"):
            print(f"=== [speech] judge prompt: {prompt} ===")
            speech_judge.judge_transcripts(
                service_results, edacc_df, prompt_name=prompt, results_dir=rc.task_results_dir("speech_recognition"),
            )

        for prompt in _list_prompts(prompts_root, "human-loop"):
            print(f"=== [speech] human-loop prompt: {prompt} ===")
            speech_human_loop.human_loop_transcripts(
                service_results, edacc_df, prompt_name=prompt, results_dir=rc.task_results_dir("speech_recognition"),
            )


def _benchmark_language(europarl_df, service_results):
    if not service_results:
        print("--- Skipping translation prompts (no enabled services) ---")
        return
    prompts_root = language_oracle._PROMPTS_ROOT

    for prompt in _list_prompts(prompts_root, "oracle"):
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
        for prompt in _list_prompts(prompts_root, "judge"):
            print(f"=== [language] judge prompt: {prompt} ===")
            language_judge.judge_translations(
                service_results, europarl_df, prompt_name=prompt, results_dir=rc.task_results_dir("language_translation"),
            )

        for prompt in _list_prompts(prompts_root, "human-loop"):
            print(f"=== [language] human-loop prompt: {prompt} ===")
            language_human_loop.human_loop_translations(
                service_results, europarl_df, prompt_name=prompt, results_dir=rc.task_results_dir("language_translation"),
            )


def _benchmark_emotion(vea_df, service_results):
    if not service_results:
        print("--- Skipping emotion prompts (no enabled services) ---")
        return
    prompts_root = emotion_oracle._PROMPTS_ROOT

    for prompt in _list_prompts(prompts_root, "oracle"):
        print(f"=== [emotion] oracle prompt: {prompt} ===")
        oracle_results = emotion_oracle.generate_oracle_emotions(
            vea_df, prompt_name=prompt, results_dir=rc.task_results_dir("emotion_detection"),
        )
        if not service_results or oracle_results is None:
            continue
        print(f"=== [emotion] classification metrics for oracle prompt: {prompt} ===")
        _write_accuracy_for(
            rc.task_results_dir("emotion_detection"), "emotion_detection", prompt,
            oracle_results, service_results, vea_df,
            compute_emotion_rows, compute_emotion_summary_rows,
        )

    if service_results:
        for prompt in _list_prompts(prompts_root, "judge"):
            print(f"=== [emotion] judge prompt: {prompt} ===")
            emotion_judge.judge_emotions(
                service_results, vea_df, prompt_name=prompt, results_dir=rc.task_results_dir("emotion_detection"),
            )

        for prompt in _list_prompts(prompts_root, "human-loop"):
            print(f"=== [emotion] human-loop prompt: {prompt} ===")
            emotion_human_loop.human_loop_emotions(
                service_results, vea_df, prompt_name=prompt, results_dir=rc.task_results_dir("emotion_detection"),
            )


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
    edacc_df = _load_or_restore(
        "speech_recognition",
        lambda: load_edacc(num_samples, randomize=randomize, seed=seed),
        datasets, "--- Gathering viable EdAcc samples ---",
    )
    europarl_df = _load_or_restore(
        "language_translation",
        lambda: load_en_fr(num_samples, randomize=randomize, seed=seed),
        datasets, "--- Retrieving EuroParl data pairs ---",
    )
    vea_df = _load_or_restore(
        "emotion_detection",
        lambda: load_vea(num_samples, randomize=randomize, seed=seed),
        datasets, "--- Retrieving Visual Emotional Analysis samples ---",
    )

    print("=== Running speech services (once) ===")
    speech_results = _run_services_only(isr, edacc_df, run_speech_recognition)
    print("=== Running language services (once) ===")
    language_results = _run_services_only(ilt, europarl_df, run_language_translation)
    print("=== Running emotion services (once) ===")
    emotion_results = _run_services_only(ied, vea_df, run_emotion_detection)

    if speech_results:
        print("=== [speech] SDS ranking + Majority Voting baseline ===")
        task_dir = rc.task_results_dir("speech_recognition")
        sds_df = compute_discrimination(speech_results, edacc_df["id"].tolist(), output_kind="text")
        save_discrimination(sds_df, task_dir / "sds")
        mv_df = majority_vote(speech_results, edacc_df["id"].tolist(), output_kind="text")
        save_majority_voting(mv_df, task_dir / "majority_voting")

    if language_results:
        print("=== [language] SDS ranking + Majority Voting baseline ===")
        task_dir = rc.task_results_dir("language_translation")
        sds_df = compute_discrimination(language_results, europarl_df["id"].tolist(), output_kind="text")
        save_discrimination(sds_df, task_dir / "sds")
        mv_df = majority_vote(language_results, europarl_df["id"].tolist(), output_kind="text")
        save_majority_voting(mv_df, task_dir / "majority_voting")

    if emotion_results:
        print("=== [emotion] SDS ranking + Majority Voting baseline ===")
        task_dir = rc.task_results_dir("emotion_detection")
        sds_df = compute_discrimination(emotion_results, vea_df["id"].tolist(), output_kind="emotion")
        save_discrimination(sds_df, task_dir / "sds")
        mv_df = majority_vote(emotion_results, vea_df["id"].tolist(), output_kind="emotion")
        save_majority_voting(mv_df, task_dir / "majority_voting")

    _benchmark_speech(edacc_df, speech_results)
    _benchmark_language(europarl_df, language_results)
    _benchmark_emotion(vea_df, emotion_results)

    cost_log = session_tracker().write(results_root=rc.active_run_dir())
    if cost_log is not None:
        print(f"=== Benchmark cost log: {cost_log} (total ${session_tracker().total_usd():.4f}) ===")

    replot_all()


_TASK_NAMES = ("speech_recognition", "language_translation", "emotion_detection")


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
        run_all_prompts()
