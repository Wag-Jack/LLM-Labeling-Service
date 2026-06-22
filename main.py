from service_invocations.invoke_speech_recognition import run_speech_recognition
from service_invocations.invoke_language_translation import run_language_translation
from service_invocations.invoke_emotion_detection import run_emotion_detection
from service_invocations.core.terminal_mirror import trial_log
from service_invocations.core import run_context as rc
from data_management.edacc import load_edacc
from data_management.en_fr import load_en_fr
from data_management.affectnet import load_affectnet
from benchmark_prompts import run_all_prompts

# Amount of samples from each dataset to run through services
NUM_SAMPLES = 100
# Randomization controls. Set RANDOM_SEED to an integer for reproducible draws,
# or leave as None for a fresh random sample on every run.
RANDOMIZE_SAMPLES = True
RANDOM_SEED: int | None = None


# Each task: (results task name, trial-log label, loader, runner).
_TASK_SPECS = {
    "asr": (
        "speech_recognition",
        "asr",
        lambda: load_edacc(NUM_SAMPLES, randomize=RANDOMIZE_SAMPLES, seed=RANDOM_SEED),
        run_speech_recognition,
        "--- Gathering viable EdAcc samples ---",
    ),
    "fer": (
        "emotion_detection",
        "fer",
        lambda: load_affectnet(NUM_SAMPLES, randomize=RANDOMIZE_SAMPLES, seed=RANDOM_SEED),
        run_emotion_detection,
        "--- Retrieving AffectNet-7 samples ---",
    ),
    "mt": (
        "language_translation",
        "mt",
        lambda: load_en_fr(NUM_SAMPLES, randomize=RANDOMIZE_SAMPLES, seed=RANDOM_SEED),
        run_language_translation,
        "--- Retrieving EuroParl data pairs ---",
    ),
}


def _choose_run(label: str):
    """Pop-up: offer to continue an unfinished run, or start a new one.

    Returns the run directory to continue, or ``None`` to start fresh.
    """
    runs = rc.find_continuable_runs(label)
    if not runs:
        return None
    print(f"\nFound {len(runs)} unfinished '{label}' run(s):")
    print("  0) Start a NEW run")
    for i, r in enumerate(runs, 1):
        print(f"  {i}) Continue {r.info.display}   [started {r.started}; {r.progress}]")
    raw = input("Select a run to continue (0 for new): ").strip()
    if raw.isdigit():
        choice = int(raw)
        if 1 <= choice <= len(runs):
            return runs[choice - 1].info.dir
    return None


def _run_task(key: str) -> None:
    task, label, loader, runner, banner = _TASK_SPECS[key]

    continue_dir = _choose_run(task)

    # Restore the exact samples for a continued run; otherwise draw fresh ones.
    if continue_dir is not None:
        try:
            df = rc.load_samples(continue_dir)
            info = rc.start_run(task, continue_dir=continue_dir)
            append = True
            print(f"--- Continuing run {info.display} ---")
        except FileNotFoundError as exc:
            print(f"[continue] {exc} Starting a new run instead.")
            continue_dir = None

    if continue_dir is None:
        info = rc.start_run(task)
        append = False
        df = None  # loaded inside the trial log so its output is captured

    try:
        with trial_log(label, path=info.log_path, append=append):
            if df is None:
                print(banner)
                df = loader()
                rc.save_samples(df)
            runner(df)
        rc.mark_finished()
    finally:
        rc.end_run()


def _run_benchmark() -> None:
    """Benchmark every prompt, into a timestamped multi-task run folder."""
    label = "benchmark"
    continue_dir = _choose_run(label)

    datasets = None
    if continue_dir is not None:
        try:
            datasets = {
                "speech_recognition": rc.load_samples(continue_dir, name="speech_recognition"),
                "language_translation": rc.load_samples(continue_dir, name="language_translation"),
                "emotion_detection": rc.load_samples(continue_dir, name="emotion_detection"),
            }
            info = rc.start_run(label, continue_dir=continue_dir, subdir_by_task=True)
            append = True
            print(f"--- Continuing run {info.display} ---")
        except FileNotFoundError as exc:
            print(f"[continue] {exc} Starting a new run instead.")
            continue_dir = None

    if continue_dir is None:
        info = rc.start_run(label, subdir_by_task=True)
        append = False
        datasets = None

    try:
        with trial_log(label, path=info.log_path, append=append):
            run_all_prompts(
                NUM_SAMPLES,
                randomize=RANDOMIZE_SAMPLES,
                seed=RANDOM_SEED,
                datasets=datasets,
            )
        rc.mark_finished()
    finally:
        rc.end_run()


def main():
    while True:
        command = input(
            "LLM Labeling\n"
            "1.) ASR - Automatic Speech Recognition\n"
            "2.) FER - Facial Emotion Detection\n"
            "3.) MT - Machine Translation\n"
            "4.) Benchmark all prompts\n"
            "5.) Exit\n"
            "Select: "
        )

        try:
            choice = int(command.strip())
        except ValueError:
            print("Invalid. Select an option between 1-5.")
            continue

        if 1 <= choice < 6:
            match choice:
                case 1:
                    _run_task("asr")
                case 2:
                    _run_task("fer")
                case 3:
                    _run_task("mt")
                case 4:
                    _run_benchmark()
                case 5:
                    break
        else:
            print("Invalid. Select an option between 1-5.")


if __name__ == "__main__":
    main()
