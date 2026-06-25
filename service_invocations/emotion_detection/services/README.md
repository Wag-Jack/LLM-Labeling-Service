# FER Services — setup, API keys & contempt capability

Facial-emotion-recognition (FER) services for the `emotion_detection` task,
discovered from `config/services.yaml` (any entry with `enabled: true`); each
exposes `run(affectnet_data, results_path)`.

Ground truth is **AffectNet-7** (`data_management/affectnet.py`), classes
`anger, disgust, fear, happy, neutral, sad, surprise`. AffectNet-7 has **no
contempt class**, so the canonical set drops contempt everywhere
(`services/_shared.py`).

**Live-verified status (June 2026)** — every service was smoke-tested end to end
on real AffectNet-7 images:

| Service | Status | Type | Notes |
|---|---|---|---|
| AWS Rekognition | ✅ enabled | cloud API | works as-is |
| Luxand | ✅ enabled | cloud API | works as-is |
| Face++ | ✅ enabled | cloud API | added 403 concurrency-limit retry |
| **FER** | ✅ enabled | **local lib** | `fer` library (Keras mini-Xception); no API key, $0/image |
| **DeepFace** | ✅ enabled | **local lib** | `deepface` library; no API key, $0/image |

> **Imentiv and SkyBiometry were removed** and replaced by the two local
> libraries above (FER, DeepFace). They run entirely on-device — no API key, no
> network call, no per-image cost — and merge into the pipeline through the exact
> same `run(...)` / `service_output` contract as every other service.

---

## 1. Dataset: AffectNet-7 (Kaggle)

The loader pulls `lcdngthnh/affectnet-7` via **kagglehub** and samples the
**training** split. The download is ~307 MB; its real layout is a flat image dir
plus an annotation file, both handled by the loader:

```
AffectNet/train_set/images/<emotion>_<source>_<n>.{jpg,png}
AffectNet/train_set/train_annotations_7class.txt   # "<filename> <expression_label>"
```

> **kagglehub needs a username AND a key.** Your `.env` has `KAGGLE_USERNAME`
> and `KAGGLE_API_TOKEN` (the loader maps `KAGGLE_API_TOKEN → KAGGLE_KEY`).
> Alternatively use a `~/.kaggle/kaggle.json`.

Each sampled image is preprocessed (see below) into the working image
`Data/AffectNet/images/NNNN.png`, with the untouched original kept at
`Data/AffectNet/images_raw/NNNN.{ext}`. Full provenance is written to
`Data/AffectNet/affectnet_metadata.csv`:
`id, image, raw_image, label, label_name, source_class, source_filename,
source_relpath, split, annotation` — so the exact draw and its original
AffectNet labels are traceable. First run caches the dataset under
`~/.cache/kagglehub/`.

### Preprocessing (applied uniformly — fairness)

To keep the comparison fair, **the same preprocessing is applied once at load
time** so every service AND every LLM paradigm (oracle/judge/human-loop) reads
the identical `image`. It adds a uniform gray border around AffectNet's tight
96×96 crops, which is **required** by detectors that need surrounding context
(Imentiv finds 0 faces without it) and harmless to the rest (AWS/Luxand/Face++
still detect the same face; the native face resolution is preserved — padding
only, no upscale by default). Controlled by env vars:

| Var | Default | Meaning |
|---|---|---|
| `FER_PREPROCESS` | `1` | master on/off |
| `FER_PREPROCESS_PAD_RATIO` | `0.4` | border as a fraction of the larger dimension |
| `FER_PREPROCESS_MIN_SIZE` | `0` | upscale so the larger side ≥ this (0 = pad only) |

The original image is always preserved at `raw_image` for reproducibility.

---

## 2. API keys per service (`.env`)

| Service | Required env vars | Optional env vars (defaults) | Get the key |
|---|---|---|---|
| **AWS Rekognition** | `AWS_REGION`, `AWS_ACCESS_KEY`, `AWS_SECRET_ACCESS_KEY` | — | AWS IAM user w/ `rekognition:DetectFaces` |
| **Luxand** | `LUXAND_API_TOKEN` | `LUXAND_EMOTION_URL` | <https://dashboard.luxand.cloud> |
| **Face++** | `FACEPP_API_KEY`, `FACEPP_API_SECRET` | `FACEPP_DETECT_URL`, `FACEPP_REQUEST_DELAY` (`0.6` s between calls) | <https://console.faceplusplus.com> |
| **FER** *(local)* | — *(none)* | `FER_MTCNN` (`0`; set `1` for the slower/more-accurate MTCNN detector) | — `pip install fer` |
| **DeepFace** *(local)* | — *(none)* | `DEEPFACE_BACKEND` (`opencv`), `DEEPFACE_ENFORCE_DETECTION` (`0`) | — `pip install deepface` |

FER and DeepFace need **no credentials** — they run locally. On first use each
downloads a small pretrained model once (FER bundles its weights; DeepFace
fetches `facial_expression_model_weights.h5` ~6 MB to `~/.deepface/weights/`).

Setup steps: `pip install -r requirements.txt` (installs `fer`, `deepface`,
`tf-keras` and the pinned TensorFlow/OpenCV — see the requirements note on why
TensorFlow is capped at `<2.18`); ensure `KAGGLE_USERNAME` is set; the cloud FER
keys are already in `.env`. `usd_per_image` rates in `services.yaml` are
approximate for the cloud services; FER and DeepFace are priced at `$0` (local).

---

## 3. Contempt capability (flag for your results)

AffectNet-7 has no contempt class. Verified per-service behavior:

| Service | Reports contempt? | Multi-emotion? | Handling |
|---|---|---|---|
| AWS Rekognition | **No** | Yes | native confidences kept |
| Face++ | **No** | Yes | 7-emotion dist (~100), kept |
| Luxand | **Yes** | Yes | contempt **dropped + renormalized** to 1.0 |
| **FER** | **No** | Yes | 7-emotion dist (0-1, ~1.0), kept |
| **DeepFace** | **No** | Yes | 7-emotion dist (rescaled 0-100 → 0-1), kept |

**Services that CANNOT read contempt:** `aws_rekognition`, `faceplusplus`, `fer`,
`deepface` (none of these has a contempt class — it is simply absent from their
output, so nothing is dropped or renormalized).
**Services that CAN read contempt** (dropped + renormalized, since AffectNet-7
omits it): `luxand_facesdk`.
Recorded in `SERVICE_EMOTION_CAPABILITIES`; logic in `renormalize_scores()` /
`contempt_was_reported()`.

**Multi-emotion:** AWS, Luxand, Face++, FER and DeepFace all return full
seven-class distributions.

---

## 4. Notes / caveats (from the live run)

- **Microsoft Azure Face removed** — Microsoft retired emotion recognition from
  the Azure Face API; module + config entry deleted.
- **Imentiv and SkyBiometry removed** — replaced by the local **FER** and
  **DeepFace** libraries (`services/imentiv.py` and `services/skybiometry.py`
  deleted, config entries and capability rows removed). The two cloud services
  are no longer part of the pipeline.
- **FER** (`services/fer.py`, local `fer` library):
  - Runs on-device via TensorFlow — **no API key, no network call, $0/image**.
    `detect_emotions()` returns one entry per detected face with a seven-class
    `emotions` dict already on a 0-1 scale (~1.0 sum); no contempt class.
  - Uses the OpenCV Haar-cascade detector by default; set `FER_MTCNN=1` for the
    slower, more accurate MTCNN detector (needs `facenet-pytorch`, installed as a
    transitive dep). The class moved from `fer.FER` to `fer.fer.FER` in the 25.x
    rewrite — the service imports from whichever path is present.
  - The detector is built **once per run** and reused across images.
- **DeepFace** (`services/deepface.py`, local `deepface` library):
  - Runs on-device via TensorFlow — **no API key, no network call, $0/image**.
    `analyze(actions=["emotion"])` returns a seven-class `emotion` dict on a
    0-100 scale (rescaled to 0-1 here); no contempt class.
  - `enforce_detection=False` by default, so a result is always returned even on
    a low-confidence crop (set `DEEPFACE_ENFORCE_DETECTION=1` to require a
    confident face). Face-detector backend is `opencv` by default
    (`DEEPFACE_BACKEND`).
  - **Needs `tf-keras`** under Keras-3 TensorFlow (DeepFace/RetinaFace require
    the legacy Keras 2 backend); it is in `requirements.txt`.
- **TensorFlow pin (important).** FER and DeepFace pull in TensorFlow. TF `>=2.18`
  requires `protobuf>=5` / `numpy>=2`, which **conflicts with `unbabel-comet`**
  (the MT scorer: `protobuf<5`, `numpy<2`). `requirements.txt` therefore caps
  `tensorflow<2.18` (resolves to 2.17) and `opencv-python<4.11`, keeping the
  whole pipeline — FER, DeepFace, STT (`torchaudio`), MT (`comet`) — on one
  mutually-compatible set (`pip check` clean).
- **Face++** — free tier returns `403 CONCURRENCY_LIMIT_EXCEEDED` under load;
  now retried with backoff (`extra_retryable`) plus a `FACEPP_REQUEST_DELAY`
  (0.6 s) between calls → 8/8. Re-verified: `v3/detect` returns 7 emotions
  summing to ~100, **no contempt** (stale `contempt` mapping removed).
- **Retry on empty result (all FER services).** Beyond the HTTP-level backoff in
  `request_with_retry`, every service now re-issues the **whole** per-image call
  when it comes back with no usable emotion — whether an error, no face
  registered, or an empty response — via `call_until_emotion` in `_shared.py`.
  Bounded by `FER_EMPTY_MAX_ATTEMPTS` (default `3`); for the cloud services
  **each attempt is a billed request**, so the recorded cost is `count=attempts`
  (FER/DeepFace are local, so their per-image cost stays `$0` regardless).
  **Permanent client errors (a non-transient 4xx — 401/402/403/404)
  short-circuit:** retrying can't fix them, so the loop stops on the first one
  (e.g. a cloud service returning `402 Insufficient credits` — top up the
  account; not a code issue). For the local libraries an empty result means the
  detector found no face; the retry is deterministic, so it simply records the
  no-face outcome.
