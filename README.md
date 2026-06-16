# LLM-Labeling-Service

Code repository for the ICWS 2026 publication on using LLMs to label similar web
services with quality scores. The pipeline calls a panel of commercial services
(ASR, MT, FER), then has LLMs act as oracle, judge, and human-in-the-loop over
the service outputs.

## Setup

```bash
# Create / activate the virtualenv
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Provide API keys (one .env file at the repo root)
cp env/.env.example .env   # if a template exists; otherwise create .env manually
```

Configuration lives under `config/`:
- `config/models.yaml` — enabled LLMs.
- `config/services.yaml` — enabled commercial services + per-service pricing.
- `config/prompts.yaml` — which prompt files to include in the benchmark sweep.

## Running iterations

### Interactive menu

```bash
python main.py
```

Pick from the menu:
1. **ASR** — Automatic Speech Recognition (EdAcc samples)
2. **FER** — Facial Emotion Recognition (Visual Emotional Analysis samples)
3. **MT**  — Machine Translation (EuroParl en→fr pairs)
4. **Benchmark all prompts** — sweep every enabled prompt across all three tasks
5. **Exit**

When unfinished runs exist for a task, the menu offers to continue them
instead of starting a new one. Sample count and randomization are controlled by
the `NUM_SAMPLES`, `RANDOMIZE_SAMPLES`, and `RANDOM_SEED` constants at the top
of [main.py](main.py).

### Benchmark directly (non-interactive)

```bash
# Run the full prompt sweep into a fresh timestamped run folder
python benchmark_prompts.py
```

Results land in `service_invocations/results/<YYYY-MM-DD>/<HH-MM-SS>_benchmark/`
with one subdirectory per task.

## Regenerating plots after iterations are run

Both commands re-read the CSVs an existing run produced and re-render the
plots — no LLM or service calls are made.

### From `benchmark_prompts.py` (plots only)

```bash
# Re-plot the legacy results/<task>/ location
python benchmark_prompts.py --plots-only

# Re-plot a specific run
python benchmark_prompts.py --plots-only \
    --run 2026-06-02/22-02-36_benchmark

# Limit to one or more tasks (repeatable)
python benchmark_prompts.py --plots-only \
    --run 2026-06-02/22-02-36_benchmark \
    --task speech_recognition \
    --task emotion_detection
```

`--run` accepts an absolute path or a path relative to
`service_invocations/results/`.

### From `regenerate_plots.py` (plots + LLMaaS backfill)

```bash
# Re-plot every task in a run directory
python -m service_invocations.regenerate_plots \
    service_invocations/results/2026-06-02/22-02-36_benchmark

# Re-plot a single task directory
python -m service_invocations.regenerate_plots \
    service_invocations/results/2026-06-02/22-02-36_benchmark/speech_recognition

# Backfill the LLMaaS metric only (no plot rebuild)
python -m service_invocations.regenerate_plots \
    service_invocations/results/2026-06-02/22-02-36_benchmark \
    --no-plots

# Plots only — skip the LLMaaS backfill
python -m service_invocations.regenerate_plots \
    service_invocations/results/2026-06-02/22-02-36_benchmark \
    --plots-only

# Restrict to one task
python -m service_invocations.regenerate_plots \
    service_invocations/results/2026-06-02/22-02-36_benchmark \
    --task speech_recognition
```

The LLMaaS backfill scores each model's stored oracle output against the human
reference so the LLM appears as a peer "service" in the accuracy plots. It is
currently implemented for `speech_recognition`; other tasks re-plot only.
