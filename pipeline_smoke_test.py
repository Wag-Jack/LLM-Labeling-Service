"""Pre-flight pipeline smoke test (main.py -> 5).

Exercises ONE example from each requested dataset against every model, service,
and prompt declared in the YAML configs -- enabled or not -- so configuration
problems (missing API keys, bad model ids, out-of-credit services, unreadable
prompt files) surface in a couple of minutes instead of partway through a full
run. The goal is breadth, not metrics: each check is a single call whose only
verdict is "did this component work in the pipeline?".

What it covers, per task:
  * Models   - every entry in config/models.yaml is invoked once with the task's
               representative oracle prompt and the sample input, confirming the
               provider/key/modality all work end-to-end.
  * Services - every entry under the task in config/services.yaml has its run()
               called on the single sample, confirming the provider returns a
               usable output.
  * Prompts  - every *.txt under prompts/<paradigm>/ is enumerated (config/
               prompts.yaml selection is ignored on purpose). Oracle prompts are
               run through one working model; judge / human-loop prompts are
               load-checked only (full execution needs service outputs, i.e. the
               real pipeline).

Disabled components are tested too -- a disabled-but-now-working service or a
still-broken one is exactly the insight this report is meant to give.

After the pass/fail summary, a combined cost report (LLM token spend + priced
service calls, same format as the real pipeline) is printed for the record. The
session cost trackers are reset at the start of each run so the total reflects
only this smoke test.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from data_management.affectnet import load_affectnet
from data_management.edacc import load_edacc
from data_management.en_fr import load_en_fr
from service_invocations.core.cost_tracker import (
    compute_cost,
    reset_session,
    session_tracker,
)
from service_invocations.core.oracle_utils import (
    extract_oracle,
    is_nullish_output,
    load_prompt,
)
from service_invocations.core.service_cost import (
    format_cost_summary,
    reset_session_service,
)
from service_invocations.emotion_detection import emotion_oracle
from service_invocations.language_translation import language_oracle
from service_invocations.models import (
    _get_models_section,
    _load_models_config,
    get_model_generator,
)
from service_invocations.speech_recognition import speech_oracle

_CONFIG_DIR = Path.cwd() / "config"
_MODELS_PATH = _CONFIG_DIR / "models.yaml"
_SERVICES_PATH = _CONFIG_DIR / "services.yaml"

_PARADIGMS = ("oracle", "judge", "human-loop")

# Status tags, padded so the result columns line up in the terminal.
_OK, _FAIL, _SKIP = "OK", "FAIL", "SKIP"
_TAG = {_OK: "[ OK ]", _FAIL: "[FAIL]", _SKIP: "[SKIP]"}


@dataclass
class TaskSpec:
    key: str                              # short menu/selection key (asr/fer/mt)
    task_name: str                        # config + module namespace
    label: str                            # human-facing banner
    loader: Callable[[], pd.DataFrame]    # draws a 1-row sample DataFrame
    services_package: str                 # import root for service modules
    prompts_root: Path                    # prompts/ folder for the task
    oracle_key: str                       # JSON key the oracle prompt returns
    output_kind: str                      # "text" | "emotion" (service output)
    build_inputs: Callable[[pd.Series], dict]  # model inputs from a sample row


def _asr_inputs(row: pd.Series) -> dict:
    return {"audio": row["audio"]}


def _mt_inputs(row: pd.Series) -> dict:
    return {"text": row["english"]}


def _fer_inputs(row: pd.Series) -> dict:
    return {"image": row["image"]}


# EdAcc uploads its audio to S3 (AWS Transcribe reads it from there), mirroring
# the real pipeline. _load_task_sample falls back to aws=False if the upload
# fails, so the non-S3 ASR services can still be exercised.
_TASKS: dict[str, TaskSpec] = {
    "asr": TaskSpec(
        key="asr",
        task_name="speech_recognition",
        label="ASR - Automatic Speech Recognition (EdAcc)",
        loader=lambda: load_edacc(1, aws=True, randomize=True, seed=None),
        services_package="service_invocations.speech_recognition.services",
        prompts_root=speech_oracle._PROMPTS_ROOT,
        oracle_key="transcript",
        output_kind="text",
        build_inputs=_asr_inputs,
    ),
    "fer": TaskSpec(
        key="fer",
        task_name="emotion_detection",
        label="FER - Facial Emotion Recognition (AffectNet-7)",
        loader=lambda: load_affectnet(1, randomize=True, seed=None),
        services_package="service_invocations.emotion_detection.services",
        prompts_root=emotion_oracle._PROMPTS_ROOT,
        oracle_key="scores",
        output_kind="emotion",
        build_inputs=_fer_inputs,
    ),
    "mt": TaskSpec(
        key="mt",
        task_name="language_translation",
        label="MT - Machine Translation (EuroParl en->fr)",
        loader=lambda: load_en_fr(1, randomize=True, seed=None),
        services_package="service_invocations.language_translation.services",
        prompts_root=language_oracle._PROMPTS_ROOT,
        oracle_key="translation",
        output_kind="text",
        build_inputs=_mt_inputs,
    ),
}


@dataclass
class CheckResult:
    name: str
    status: str               # _OK | _FAIL | _SKIP
    detail: str = ""
    enabled: bool | None = None  # YAML enabled flag, where applicable


@dataclass
class TaskReport:
    task: TaskSpec
    data_ok: bool = False
    data_detail: str = ""
    models: list[CheckResult] = field(default_factory=list)
    services: list[CheckResult] = field(default_factory=list)
    prompts: dict[str, list[CheckResult]] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------


def _oneline(value: object, limit: int = 200) -> str:
    """Collapse any value to a single trimmed line for compact reporting."""
    text = re.sub(r"\s+", " ", str(value).replace("\n", " ").replace("\r", " ")).strip()
    return text[:limit]


def _enabled_tag(enabled: bool | None) -> str:
    if enabled is None:
        return ""
    return "enabled" if enabled else "disabled"


def _emit(result: CheckResult) -> None:
    """Print one check line as it completes (live progress)."""
    flag = _enabled_tag(result.enabled)
    flag = f"({flag}) " if flag else ""
    detail = f"  - {result.detail}" if result.detail else ""
    print(f"    {_TAG[result.status]} {result.name:<26} {flag}{detail}".rstrip(), flush=True)


# --------------------------------------------------------------------------
# YAML enumeration (enabled AND disabled entries)
# --------------------------------------------------------------------------


def _all_models() -> list[tuple[str, dict]]:
    config = _load_models_config(_MODELS_PATH)
    section = _get_models_section(config)
    return [(name, entry) for name, entry in section.items() if isinstance(entry, dict)]


def _all_services(task_name: str) -> list[tuple[str, dict]]:
    if not _SERVICES_PATH.exists():
        raise FileNotFoundError(f"Services config not found: {_SERVICES_PATH}")
    with _SERVICES_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    task_cfg = config.get(task_name) or {}
    if not isinstance(task_cfg, dict):
        return []
    return [(name, entry) for name, entry in task_cfg.items() if isinstance(entry, dict)]


def _all_prompts(prompts_root: Path, paradigm: str) -> list[Path]:
    folder = prompts_root / paradigm
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.txt"))


def _representative_oracle_prompt(prompts_root: Path) -> Path | None:
    """Pick the oracle prompt used for the per-model probe (prefer *medium*)."""
    prompts = _all_prompts(prompts_root, "oracle")
    if not prompts:
        return None
    for path in prompts:
        if "medium" in path.stem.lower():
            return path
    return prompts[0]


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------


def _load_task_sample(task: TaskSpec) -> tuple[pd.DataFrame | None, str]:
    """Load a single-sample DataFrame for a task; (df, detail) on success."""
    try:
        df = task.loader()
    except Exception as exc:  # noqa: BLE001 - any loader failure is a reportable result
        # EdAcc's S3 upload is the most likely failure; retry without it so the
        # non-S3 services remain testable (AWS Transcribe will then fail, which
        # the report will show).
        if task.key == "asr":
            print(f"    [warn] EdAcc load with S3 upload failed ({_oneline(exc)}); "
                  "retrying without upload (AWS Transcribe will be unavailable).",
                  flush=True)
            try:
                df = load_edacc(1, aws=False, randomize=True, seed=None)
            except Exception as exc2:  # noqa: BLE001
                return None, _oneline(exc2)
        else:
            return None, _oneline(exc)

    if df is None or len(df) == 0:
        return None, "loader returned no rows"
    return df.iloc[[0]].reset_index(drop=True), f"sample id={df.iloc[0].get('id', '?')}"


# --------------------------------------------------------------------------
# Per-component checks
# --------------------------------------------------------------------------


def _exercise_oracle(
    model_name: str,
    prompt_path: Path,
    task: TaskSpec,
    sample: pd.Series,
    gen_cache: dict[str, tuple[Callable | None, str | None]],
    call_cache: dict[tuple[str, str], tuple[str, str]],
) -> tuple[str, str]:
    """Invoke one (model, oracle-prompt) cell once; returns (status, detail).

    Results are memoized on (model, prompt stem) so the per-model probe and the
    per-prompt probe never bill the same cell twice.
    """
    cache_key = (model_name, prompt_path.stem)
    if cache_key in call_cache:
        return call_cache[cache_key]

    if model_name not in gen_cache:
        try:
            gen_cache[model_name] = (get_model_generator(model_name, models_path=_MODELS_PATH), None)
        except Exception as exc:  # noqa: BLE001 - missing key / bad provider etc.
            gen_cache[model_name] = (None, _oneline(exc))
    generator, init_err = gen_cache[model_name]
    if generator is None:
        result = (_FAIL, f"init error: {init_err}")
        call_cache[cache_key] = result
        return result

    try:
        prompt_text = load_prompt(prompt_path)
    except Exception as exc:  # noqa: BLE001
        result = (_FAIL, f"prompt load error: {_oneline(exc)}")
        call_cache[cache_key] = result
        return result

    try:
        response = generator(prompt_text, inputs=task.build_inputs(sample))
    except Exception as exc:  # noqa: BLE001 - any provider error is a reportable failure
        result = (_FAIL, _oneline(exc))
        call_cache[cache_key] = result
        return result

    value = extract_oracle(response.content, key=task.oracle_key)
    usable = not is_nullish_output(value)

    # Record the billed LLM call on the session tracker so the end-of-run cost
    # summary reflects the smoke test's own spend. Mirrors the real pipeline:
    # every billed call counts, regardless of whether its output was usable.
    cost = compute_cost(
        model_name,
        getattr(response, "input_tokens", None),
        getattr(response, "output_tokens", None),
        _MODELS_PATH,
        audio_input_tokens=getattr(response, "audio_input_tokens", None),
    )
    session_tracker().record(
        task=task.task_name,
        paradigm="oracle",
        model=model_name,
        sample_id=sample.get("id", "?") if hasattr(sample, "get") else "?",
        input_tokens=getattr(response, "input_tokens", None),
        output_tokens=getattr(response, "output_tokens", None),
        cost_usd=cost,
        status="success" if usable else "failed",
        latency_ms=getattr(response, "latency_ms", None),
        audio_input_tokens=getattr(response, "audio_input_tokens", None),
    )

    if usable:
        result = (_OK, f"{response.latency_ms:.0f} ms")
    else:
        snippet = _oneline(response.content) or "(empty response)"
        result = (_FAIL, f"no usable '{task.oracle_key}': {snippet[:60]}")
    call_cache[cache_key] = result
    return result


def _summarize_service_output(df: object, output_kind: str) -> tuple[bool, str]:
    """Decide whether a service's run() produced a usable result."""
    if df is None:
        return False, "run() returned None"
    try:
        n_rows = len(df)
    except TypeError:
        return False, "run() did not return a DataFrame"
    if n_rows == 0:
        return False, "run() returned 0 rows"
    columns = getattr(df, "columns", [])
    if "service_output" not in columns:
        return False, "no service_output column"
    output = df.iloc[0]["service_output"]
    if output_kind == "emotion":
        return _summarize_emotion_output(output)
    return _summarize_text_output(output)


def _summarize_text_output(output: object) -> tuple[bool, str]:
    if output is None:
        return False, "empty output"
    text = str(output).strip()
    if not text or text.lower() in {"nan", "none"}:
        return False, "empty output"
    return True, f'"{_oneline(text)[:60]}"'


def _summarize_emotion_output(output: object) -> tuple[bool, str]:
    try:
        payload = json.loads(output) if isinstance(output, str) else {}
    except (json.JSONDecodeError, TypeError):
        return False, f"unparseable output: {_oneline(output)[:60]}"
    if not isinstance(payload, dict):
        return False, "non-dict output"
    error = payload.get("error")
    top = (payload.get("top_emotion") or {}).get("name")
    if top:
        return True, f"top={top}" + (f" (note: {_oneline(error)[:40]})" if error else "")
    if error:
        return False, _oneline(error)[:80]
    return False, "no face / no emotion returned"


def _check_service(
    service_name: str,
    entry: dict,
    task: TaskSpec,
    sample_df: pd.DataFrame,
    tmp_dir: Path,
) -> CheckResult:
    enabled = bool(entry.get("enabled", False))
    try:
        module = import_module(f"{task.services_package}.{service_name}")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(service_name, _FAIL, f"import error: {_oneline(exc)}", enabled)
    runner = getattr(module, "run", None)
    if not callable(runner):
        return CheckResult(service_name, _FAIL, "module has no run() function", enabled)

    results_file = getattr(module, "RESULTS_FILE", f"{service_name}.csv")
    results_path = tmp_dir / f"{service_name}__{results_file}"
    try:
        out = runner(sample_df, results_path=results_path)
    except Exception as exc:  # noqa: BLE001 - provider/credential failures are the point
        return CheckResult(service_name, _FAIL, _oneline(exc), enabled)

    ok, detail = _summarize_service_output(out, task.output_kind)
    return CheckResult(service_name, _OK if ok else _FAIL, detail, enabled)


# --------------------------------------------------------------------------
# Per-task driver
# --------------------------------------------------------------------------


def _smoke_test_task(task: TaskSpec) -> TaskReport:
    report = TaskReport(task=task)
    print(f"\n=== {task.label} ===", flush=True)

    print("--- Loading one sample ---", flush=True)
    sample_df, data_detail = _load_task_sample(task)
    report.data_ok = sample_df is not None
    report.data_detail = data_detail
    if sample_df is not None:
        print(f"    {_TAG[_OK]} dataset loaded ({data_detail})", flush=True)
    else:
        print(f"    {_TAG[_FAIL]} dataset load failed - {data_detail}", flush=True)
        print("    Skipping model + service probes; prompts are still load-checked.", flush=True)

    sample = sample_df.iloc[0] if sample_df is not None else None
    gen_cache: dict[str, tuple[Callable | None, str | None]] = {}
    call_cache: dict[tuple[str, str], tuple[str, str]] = {}

    # --- Models -----------------------------------------------------------
    print("--- Models (config/models.yaml) ---", flush=True)
    rep_prompt = _representative_oracle_prompt(task.prompts_root)
    models = _all_models()
    if not models:
        print("    (no models declared)", flush=True)
    for model_name, entry in models:
        enabled = bool(entry.get("enabled", False))
        if sample is None or rep_prompt is None:
            reason = "no sample" if sample is None else "no oracle prompt to probe with"
            result = CheckResult(model_name, _SKIP, reason, enabled)
        else:
            status, detail = _exercise_oracle(
                model_name, rep_prompt, task, sample, gen_cache, call_cache
            )
            result = CheckResult(model_name, status, detail, enabled)
        report.models.append(result)
        _emit(result)

    # Reference model for exercising oracle prompts: the first one that worked.
    reference_model = next((r.name for r in report.models if r.status == _OK), None)

    # --- Services ---------------------------------------------------------
    print("--- Services (config/services.yaml) ---", flush=True)
    services = _all_services(task.task_name)
    if not services:
        print("    (no services declared)", flush=True)
    if sample_df is None:
        for service_name, entry in services:
            result = CheckResult(service_name, _SKIP, "no sample", bool(entry.get("enabled", False)))
            report.services.append(result)
            _emit(result)
    else:
        with tempfile.TemporaryDirectory(prefix="smoke_test_") as tmp:
            tmp_dir = Path(tmp)
            for service_name, entry in services:
                result = _check_service(service_name, entry, task, sample_df, tmp_dir)
                report.services.append(result)
                _emit(result)

    # --- Prompts ----------------------------------------------------------
    print("--- Prompts (all paradigms, config/prompts.yaml selection ignored) ---", flush=True)
    for paradigm in _PARADIGMS:
        prompts = _all_prompts(task.prompts_root, paradigm)
        if paradigm == "oracle":
            note = (f"run through model '{reference_model}'" if reference_model
                    else "load-check only (no working model to probe with)")
        else:
            note = "load-check only (full run needs service outputs)"
        print(f"  {paradigm}  [{note}]:", flush=True)
        results: list[CheckResult] = []
        if not prompts:
            print(f"    {_TAG[_FAIL]} (no .txt prompts found in {paradigm}/)", flush=True)
            results.append(CheckResult(f"{paradigm}/*", _FAIL, "no prompt files found"))
            report.prompts[paradigm] = results
            continue
        for prompt_path in prompts:
            result = _check_prompt(
                prompt_path, paradigm, task, sample, reference_model, gen_cache, call_cache
            )
            results.append(result)
            _emit(result)
        report.prompts[paradigm] = results

    return report


def _check_prompt(
    prompt_path: Path,
    paradigm: str,
    task: TaskSpec,
    sample: pd.Series | None,
    reference_model: str | None,
    gen_cache: dict,
    call_cache: dict,
) -> CheckResult:
    name = prompt_path.stem
    # Every prompt is load-checked first (catches missing/empty/unreadable files).
    try:
        text = load_prompt(prompt_path)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, _FAIL, f"load error: {_oneline(exc)}")
    if not text.strip():
        return CheckResult(name, _FAIL, "prompt file is empty")

    # Oracle prompts get exercised end-to-end through one working model; judge /
    # human-loop prompts are load-only (executing them is the real pipeline).
    if paradigm != "oracle":
        return CheckResult(name, _OK, f"loaded ({len(text)} chars)")
    if reference_model is None or sample is None:
        return CheckResult(name, _SKIP, f"loaded ({len(text)} chars); no model to probe with")
    status, detail = _exercise_oracle(
        reference_model, prompt_path, task, sample, gen_cache, call_cache
    )
    return CheckResult(name, status, detail)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def _tally(results: list[CheckResult]) -> tuple[int, int, int]:
    ok = sum(1 for r in results if r.status == _OK)
    fail = sum(1 for r in results if r.status == _FAIL)
    skip = sum(1 for r in results if r.status == _SKIP)
    return ok, fail, skip


def _print_grand_summary(reports: list[TaskReport]) -> None:
    print("\n" + "=" * 70, flush=True)
    print("SMOKE TEST SUMMARY", flush=True)
    print("=" * 70, flush=True)

    failures: list[str] = []
    for report in reports:
        task = report.task
        model_t = _tally(report.models)
        service_t = _tally(report.services)
        prompt_results = [r for results in report.prompts.values() for r in results]
        prompt_t = _tally(prompt_results)

        data_mark = _TAG[_OK] if report.data_ok else _TAG[_FAIL]
        print(f"\n{task.label}", flush=True)
        print(f"  dataset : {data_mark} {report.data_detail}", flush=True)
        print(f"  models  : {model_t[0]} ok / {model_t[1]} fail / {model_t[2]} skip", flush=True)
        print(f"  services: {service_t[0]} ok / {service_t[1]} fail / {service_t[2]} skip", flush=True)
        print(f"  prompts : {prompt_t[0]} ok / {prompt_t[1]} fail / {prompt_t[2]} skip", flush=True)

        for category, results in (
            ("model", report.models),
            ("service", report.services),
            ("prompt", prompt_results),
        ):
            for r in results:
                if r.status == _FAIL:
                    failures.append(f"  [{task.key}] {category} '{r.name}': {r.detail}")

    print("\n" + "-" * 70, flush=True)
    if failures:
        print(f"NEEDS ATTENTION ({len(failures)} failing component(s)):", flush=True)
        for line in failures:
            print(line, flush=True)
    else:
        print("All exercised components passed. Pipeline looks ready.", flush=True)
    print("-" * 70, flush=True)


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def _resolve_task_selection(tasks: list[str] | None) -> list[str]:
    if tasks:
        return [t for t in tasks if t in _TASKS]
    raw = input(
        "\nDatasets to smoke test - any of [asr, fer, mt] (comma/space separated), "
        "or 'all' [all]: "
    ).strip().lower()
    if not raw or raw == "all":
        return list(_TASKS.keys())
    chosen = [t.strip() for t in raw.replace(",", " ").split()]
    valid = [t for t in chosen if t in _TASKS]
    if not valid:
        print("No valid datasets recognized; defaulting to all.")
        return list(_TASKS.keys())
    return valid


def run_smoke_test(tasks: list[str] | None = None) -> list[TaskReport]:
    """Run the smoke test for the requested tasks (interactive if ``tasks`` is None)."""
    selected = _resolve_task_selection(tasks)

    # The cost trackers are process-lifetime singletons shared with the real
    # pipeline; reset them so the end-of-run summary reflects only this smoke
    # test's spend rather than whatever menu options ran earlier in the session.
    reset_session()
    reset_session_service()

    print("\n" + "=" * 70, flush=True)
    print(f"PIPELINE SMOKE TEST - {', '.join(t.upper() for t in selected)}", flush=True)
    print("One sample per dataset, every model/service/prompt in the YAMLs "
          "(enabled or not).", flush=True)
    print("=" * 70, flush=True)

    reports = [_smoke_test_task(_TASKS[key]) for key in selected]
    _print_grand_summary(reports)

    # Cost of running the smoke test itself, for the record (LLM token spend +
    # priced service calls). Service/LLM list prices in the YAMLs are approximate.
    print("\n" + format_cost_summary(scope="smoke test"), flush=True)
    return reports


if __name__ == "__main__":
    cli_tasks = [a.lower() for a in sys.argv[1:]] or None
    run_smoke_test(cli_tasks)
