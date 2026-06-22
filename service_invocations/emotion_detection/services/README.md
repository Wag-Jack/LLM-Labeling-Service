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

| Service | Status | On 8 AffectNet imgs | Notes |
|---|---|---|---|
| AWS Rekognition | ✅ enabled | 8/8 | works as-is |
| Luxand | ✅ enabled | 8/8 | works as-is |
| Face++ | ✅ enabled | 8/8 | added 403 concurrency-limit retry |
| Imentiv | ✅ enabled | 8/8 | needed auth + async + image-padding fixes |
| SkyBiometry | ⛔ **disabled** | n/a | parser bug fixed + faces register, but **mood not provisioned on this key** (account-level) |

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
| **Imentiv** | `IMENTIV_API_KEY` | `IMENTIV_API_BASE`, `IMENTIV_REFERER` (`https://imentiv.ai`), `IMENTIV_POLL_INTERVAL` (`2.0`), `IMENTIV_POLL_TIMEOUT` (`90.0`) | <https://imentiv.ai> → My Account → My Profile |
| **SkyBiometry** *(disabled)* | `SKYBIOMETRY_API_KEY`, `SKYBIOMETRY_API_SECRET` | `SKYBIOMETRY_DETECT_URL`, `SKYBIOMETRY_ATTRIBUTES` (`all`), `SKYBIOMETRY_DETECTOR` (`aggressive`), `SKYBIOMETRY_IMAGE_FIELD` (`image`) | <https://skybiometry.com> — emotion via `mood`, but mood not provisioned on this key (see §4) |

Setup steps: `pip install -r requirements.txt` (adds `kagglehub`); ensure
`KAGGLE_USERNAME` is set; all FER keys are already in `.env` (the
`SKYBIOMETRY_API_SECRE` typo was corrected to `SKYBIOMETRY_API_SECRET`).
`usd_per_image` rates in `services.yaml` are approximate — verify per plan.

---

## 3. Contempt capability (flag for your results)

AffectNet-7 has no contempt class. Verified per-service behavior:

| Service | Reports contempt? | Multi-emotion? | Handling |
|---|---|---|---|
| AWS Rekognition | **No** | Yes | native confidences kept |
| Face++ | **No** | Yes | 7-emotion dist (~100), kept |
| **Imentiv** | **Yes** (8 emotions, ~1.0) | Yes | contempt **dropped + renormalized** to 1.0 |
| Luxand | **Yes** | Yes | contempt **dropped + renormalized** to 1.0 |
| SkyBiometry *(disabled)* | **No** | n/a | mood not provisioned on this key; would be multi-emotion if enabled (see §4) |

**Services that CANNOT read contempt:** `aws_rekognition`, `faceplusplus`.
**Services that CAN read contempt** (dropped + renormalized, since AffectNet-7
omits it): `imentiv`, `luxand_facesdk`.
Recorded in `SERVICE_EMOTION_CAPABILITIES`; logic in `renormalize_scores()` /
`contempt_was_reported()`.

**Multi-emotion:** AWS, Luxand, Face++, Imentiv all return full distributions.
**SkyBiometry** would too (an independent confidence per basic emotion) once
mood is provisioned, but it is **disabled** because this key returns no mood
(see §4); the parser already handles the multi-emotion shape.

---

## 4. Notes / caveats (from the live run)

- **Microsoft Azure Face removed** — Microsoft retired emotion recognition from
  the Azure Face API; module + config entry deleted.
- **Imentiv** (verified live; required three fixes):
  - Auth is `X-API-Key` **+ a `Referer` header** (without Referer → `403
    Missing Referer`). The Bearer route expects a JWT, not the API key.
  - `POST /v2/images` needs a **`title`** field and is **async** (`status:
    queue`). Results polled at `GET /v1/images/{id}`, emotions in
    `record["faces"][i]["emotions"]` once `completed`. First poll often 500s
    (absorbed by retry).
  - **Needs a face margin.** Imentiv found **0 faces** in raw 96×96 AffectNet
    crops (even upscaled). This is now handled by the **global load-time
    preprocessing** (a gray border, applied to all systems for fairness — §1),
    which gives Imentiv **8/8** detection; the service sends the image as-is.
- **Face++** — free tier returns `403 CONCURRENCY_LIMIT_EXCEEDED` under load;
  now retried with backoff (`extra_retryable`) plus a `FACEPP_REQUEST_DELAY`
  (0.6 s) between calls → 8/8. Re-verified: `v3/detect` returns 7 emotions
  summing to ~100, **no contempt** (stale `contempt` mapping removed).
- **SkyBiometry — still disabled, but two distinct problems untangled.**
  Re-checked against the [live docs](https://classic.skybiometry.com/documentation/)
  and re-tested live (2026-06-17).
  - **(1) Real parser bug — fixed.** Per the docs, `faces/detect` reports emotion
    two ways under each tag's `attributes`: a `mood` object (single dominant
    label, `{"value":"<happy|sad|angry|surprised|disgusted|scared|neutral>",
    "confidence":0-100}` — **not** a nested per-emotion dict) **plus** seven
    *sibling* per-emotion attributes (`anger, disgust, fear, happiness, sadness,
    surprise, neutral_mood`, each a boolean-style `{"value":"true"/"false",
    "confidence":0-100}`). The old code iterated `mood.items()` expecting
    `{emotion:{confidence}}`, matched nothing, and dropped **every** result.
    `skybiometry.py` now reads the seven per-emotion attributes into a full
    distribution (boolean+confidence → "present" probability), falling back to
    the single `mood` label. Also: **`detector=aggressive`** (lowercase, per
    docs) and a configurable upload field (`SKYBIOMETRY_IMAGE_FIELD`, default
    `image`).
  - **Faces register fine.** The live `detect` returns **1 tag** for the
    preprocessed crop — face detection / "registering" is not the blocker.
  - **(2) Blocker — mood not provisioned on this key (account-level).**
    Live-verified with ample quota (39,878 left): `attributes=all` returns every
    attribute **except `mood` and `ethnicity`** (`age_est, beard, dark_glasses,
    eyes, face, gender, glasses, hat, lips, liveness, mustache, smiling`), and
    `attributes=mood` is *accepted* (no `BAD_ATTRIBUTES`) but returns only
    `face`. So mood is gated at the **account/plan** level on SkyBiometry's side,
    not in code. Enable mood for the key (SkyBiometry dashboard / support), then
    re-run the self-test and flip `enabled: true` — no code change needed.
  ```bash
  # When mood is provisioned this prints `mood` + per-emotion attrs; today it
  # prints only the non-emotion attribute keys (gender/glasses/...).
  curl -s -F api_key=$SKYBIOMETRY_API_KEY -F api_secret=$SKYBIOMETRY_API_SECRET \
       -F attributes=all -F detector=aggressive \
       -F image=@Data/AffectNet/images/0001.png \
       https://api.skybiometry.com/fc/faces/detect.json \
    | python -m json.tool | grep -iE 'mood|anger|happiness|sadness|status|tags'
  ```
- **Retry on empty result (all FER services).** Beyond the HTTP-level backoff in
  `request_with_retry`, every service now re-issues the **whole** per-image call
  when it comes back with no usable emotion — whether an error, no face
  registered, or an empty response — via `call_until_emotion` in `_shared.py`.
  Bounded by `FER_EMPTY_MAX_ATTEMPTS` (default `3`); **each attempt is a billed
  request**, so the recorded cost is `count=attempts`. **Permanent client errors
  (a non-transient 4xx — 401/402/403/404) short-circuit:** retrying can't fix
  them, so the loop stops on the first one (e.g. **Imentiv `402 Insufficient
  credits`** — top up the account; not a code issue).
