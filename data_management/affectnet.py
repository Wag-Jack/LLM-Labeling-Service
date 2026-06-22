"""AffectNet-7 dataset loader.

Replaces the former VEA (FastJobs/Visual_Emotional_Analysis) loader. AffectNet-7
is the seven-class AffectNet expression set; unlike VEA it does NOT include a
"contempt" class, which is why the canonical emotion set and all downstream FER
comparisons drop contempt.

The dataset is pulled from Kaggle via kagglehub:
    https://www.kaggle.com/datasets/lcdngthnh/affectnet-7
Only the TRAINING split is sampled.

Observed layout of this dataset (handled as the primary case):
    AffectNet/
      train_set/
        images/                       <- flat dir, files like happy_ffhq_4112.png
        train_annotations_7class.txt  <- lines: "<filename> <expression_label>"
      valid_set/ ...
The annotation file is the source of truth for labels and is preserved per
sample (filename + expression label + any extra annotation columns). The loader
also falls back to filename-prefix parsing or a folder-per-class layout if a
different re-upload is used.

Set up Kaggle auth once: kagglehub needs a username AND a key. Put a kaggle.json
at ~/.kaggle/kaggle.json, or set KAGGLE_USERNAME plus KAGGLE_KEY (the
KAGGLE_API_TOKEN name is accepted as an alias for the key).
"""
import os
import random
import shutil
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from service_invocations.emotion_detection.services._shared import (
    LABEL_MAP,
    NAME_TO_LABEL,
    preprocess_face_image,
)

_KAGGLE_DATASET = "lcdngthnh/affectnet-7"

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Map the various class spellings AffectNet re-uploads use (folder names or
# filename prefixes) onto the canonical AffectNet-7 emotion names.
_CLASS_ALIASES = {
    "anger": "anger",
    "angry": "anger",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "afraid": "fear",
    "happy": "happy",
    "happiness": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
    "contempt": "contempt",  # not in AffectNet-7; recognized so it can be skipped
}


def _ensure_kaggle_credentials() -> None:
    """Expose Kaggle credentials from the project's .env to kagglehub.

    kagglehub authenticates with KAGGLE_USERNAME + KAGGLE_KEY (or a
    ~/.kaggle/kaggle.json file). This project stores the key in .env; the
    KAGGLE_API_TOKEN name is also accepted as an alias for KAGGLE_KEY.
    """
    load_dotenv()
    token = os.getenv("KAGGLE_API_TOKEN")
    if token and not os.getenv("KAGGLE_KEY"):
        os.environ["KAGGLE_KEY"] = token.strip()
    if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) and not (
        (Path.home() / ".kaggle" / "kaggle.json").exists()
        or (Path.home() / ".config" / "kaggle" / "kaggle.json").exists()
    ):
        raise RuntimeError(
            "Kaggle credentials not found. kagglehub needs BOTH a username and "
            "a key. Set KAGGLE_USERNAME and KAGGLE_KEY (or KAGGLE_API_TOKEN) in "
            ".env, or place a kaggle.json at ~/.kaggle/kaggle.json."
        )


def _canonical_class(name: str) -> str | None:
    """Turn a class label (word or AffectNet numeric id) into a canonical
    AffectNet-7 emotion, or None if it is not one (e.g. "contempt")."""
    key = str(name).strip().lower()
    if key in _CLASS_ALIASES:
        canonical = _CLASS_ALIASES[key]
        return canonical if canonical in NAME_TO_LABEL else None
    if key.isdigit():
        return LABEL_MAP.get(int(key))
    return None


def _find_split_dir(root: Path, keyword: str) -> Path | None:
    """Find the directory for a split (keyword e.g. 'train') in the download.

    Matches any directory whose name contains the keyword (so 'train_set',
    'train', 'training' all match), preferring one that actually holds images
    or an 'images/' subdir, then the shallowest.
    """
    matches = [
        p for p in root.rglob("*")
        if p.is_dir() and keyword in p.name.lower()
    ]
    if not matches:
        return None

    def score(p: Path):
        has_images_subdir = (p / "images").is_dir()
        has_images = any(
            c.is_file() and c.suffix.lower() in _IMAGE_EXTENSIONS for c in p.iterdir()
        )
        return (has_images_subdir or has_images, -len(p.parts))

    return max(matches, key=score)


def _images_dir_for(split_dir: Path) -> Path:
    sub = split_dir / "images"
    return sub if sub.is_dir() else split_dir


def _find_annotation_file(split_dir: Path) -> Path | None:
    """Pick the most specific annotation file (prefers *7class* annotations)."""
    cands = [
        p for p in split_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".csv"}
    ]
    if not cands:
        return None

    def rank(p: Path):
        n = p.name.lower()
        return ("7class" in n, "annotation" in n, p.suffix.lower() == ".txt")

    return max(cands, key=rank)


def _parse_annotations(ann_file: Path) -> list[tuple[str, int | None, list[str]]]:
    """Parse '<filename> <expression_label> [extra...]' (or CSV) annotation lines.

    Returns (filename, expression_label_int_or_None, extra_tokens). Skips header
    or non-image lines. The expression label is the first integer token in
    0..6 (AffectNet's ordering); any remaining tokens (e.g. valence/arousal in
    fuller AffectNet annotation files) are preserved as extras.
    """
    rows: list[tuple[str, int | None, list[str]]] = []
    for line in ann_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [t.strip() for t in (line.split(",") if "," in line else line.split())]
        if not parts:
            continue
        fname = parts[0]
        if not fname.lower().endswith(tuple(_IMAGE_EXTENSIONS)):
            continue  # header or non-image row
        rest = parts[1:]
        label_int = None
        for tok in rest:
            try:
                iv = int(float(tok))
            except (TypeError, ValueError):
                continue
            if iv in LABEL_MAP:
                label_int = iv
                break
        rows.append((fname, label_int, rest))
    return rows


def _collect_samples(split_dir: Path) -> list[dict]:
    """Enumerate every (image, label) pair for the split.

    Primary path: read the annotation file. Fallbacks: filename-prefix parsing
    of a flat images dir, then a folder-per-class layout.
    """
    images_dir = _images_dir_for(split_dir)
    samples: list[dict] = []

    # --- Primary: annotation file ---
    ann_file = _find_annotation_file(split_dir)
    if ann_file is not None:
        print(f"[affectnet] Using annotation file: {ann_file.name}")
        missing = 0
        for fname, label_int, extra in _parse_annotations(ann_file):
            img_path = images_dir / fname
            if not img_path.exists():
                missing += 1
                continue
            if label_int is None:
                canonical = _canonical_class(fname.split("_", 1)[0])
                if canonical is None:
                    continue
                label_int = NAME_TO_LABEL[canonical]
            label_name = LABEL_MAP[label_int]
            samples.append({
                "source_path": img_path,
                "source_class": label_name,
                "label_name": label_name,
                "label": label_int,
                "annotation": " ".join(extra),
            })
        if missing:
            print(f"[affectnet] {missing} annotated file(s) not found on disk (skipped).")
        if samples:
            return samples

    # --- Fallback: flat images dir, class from filename prefix ---
    if images_dir.is_dir():
        for entry in sorted(images_dir.rglob("*")):
            if entry.is_file() and entry.suffix.lower() in _IMAGE_EXTENSIONS:
                canonical = _canonical_class(entry.name.split("_", 1)[0])
                if canonical is None:
                    continue
                samples.append({
                    "source_path": entry,
                    "source_class": canonical,
                    "label_name": canonical,
                    "label": NAME_TO_LABEL[canonical],
                    "annotation": "",
                })
        if samples:
            return samples

    # --- Fallback: folder-per-class ---
    skipped: set[str] = set()
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        canonical = _canonical_class(class_dir.name)
        if canonical is None:
            if class_dir.name.lower() != "images":
                skipped.add(class_dir.name)
            continue
        for entry in class_dir.rglob("*"):
            if entry.is_file() and entry.suffix.lower() in _IMAGE_EXTENSIONS:
                samples.append({
                    "source_path": entry,
                    "source_class": class_dir.name,
                    "label_name": canonical,
                    "label": NAME_TO_LABEL[canonical],
                    "annotation": "",
                })
    if skipped:
        print(
            f"[affectnet] Skipped non-AffectNet-7 class folder(s): {sorted(skipped)} "
            "(AffectNet-7 has no contempt class)."
        )
    return samples


def load_affectnet(amount=50, aws=False, randomize=True, seed=None):
    """Download AffectNet-7, sample `amount` training images, and persist them.

    Returns a DataFrame with id/image/label/label_name plus provenance columns
    (source_class, source_filename, source_relpath, split, annotation). Image
    files are copied locally to Data/AffectNet/images so downstream services
    read a stable path independent of the kagglehub cache.

    `aws=True` additionally uploads the sampled images to S3 (not required by the
    pipeline -- AWS Rekognition reads image bytes directly). Defaults to False.
    """
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "kagglehub is required to download AffectNet-7. Install it "
            "(`pip install kagglehub`) and configure Kaggle credentials."
        ) from exc

    output_dir = Path.cwd() / "Data" / "AffectNet"
    images_dir = output_dir / "images"          # working images (preprocessed)
    raw_images_dir = output_dir / "images_raw"  # untouched originals (provenance)
    # Start from clean image folders so a re-sampled run never carries stale
    # files from a previous, differently-seeded draw.
    for d in (images_dir, raw_images_dir):
        if d.exists():
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # Preprocessing is applied uniformly to the working image so EVERY service
    # and the LLM paradigms receive identical input (fairness). Padding a margin
    # around AffectNet's tight 96x96 crops is required for some detectors (e.g.
    # Imentiv) and harmless for the rest. Disable with FER_PREPROCESS=0.
    preprocess = os.getenv("FER_PREPROCESS", "1").strip().lower() not in ("0", "false", "no", "")
    pad_ratio = float(os.getenv("FER_PREPROCESS_PAD_RATIO", "0.4"))
    min_size = int(os.getenv("FER_PREPROCESS_MIN_SIZE", "0"))  # 0 = pad only, no upscale
    print(
        f"[affectnet] Preprocessing={'on' if preprocess else 'off'} "
        f"(pad_ratio={pad_ratio}, min_size={min_size})"
    )

    _ensure_kaggle_credentials()
    dataset_root = Path(kagglehub.dataset_download(_KAGGLE_DATASET))

    train_dir = _find_split_dir(dataset_root, "train")
    if train_dir is None:
        train_dir = dataset_root
    print(f"[affectnet] Using training split: {train_dir}")

    samples = _collect_samples(train_dir)
    if not samples:
        raise RuntimeError(
            f"No AffectNet-7 images found under {train_dir}. The dataset layout "
            f"may differ from what the loader expects; inspect {dataset_root}."
        )

    indices = list(range(len(samples)))
    if randomize:
        rng = random.Random(seed)
        rng.shuffle(indices)
    indices = indices[:amount]

    data = {
        "id": [],
        "image": [],
        "raw_image": [],
        "label": [],
        "label_name": [],
        "source_class": [],
        "source_filename": [],
        "source_relpath": [],
        "split": [],
        "annotation": [],
    }

    split_name = train_dir.name

    for new_id, sample_idx in enumerate(indices, start=1):
        sample = samples[sample_idx]
        src: Path = sample["source_path"]
        ext = src.suffix.lower() if src.suffix.lower() in _IMAGE_EXTENSIONS else ".png"

        # Always keep an untouched copy of the original for provenance.
        raw_dest = raw_images_dir / f"{new_id:04d}{ext}"
        shutil.copyfile(src, raw_dest)

        # The working image (what every service + LLM reads) is the preprocessed
        # version, or a plain copy if preprocessing is disabled.
        if preprocess:
            processed = preprocess_face_image(
                src.read_bytes(), pad_ratio=pad_ratio, min_size=min_size
            )
            dest = images_dir / f"{new_id:04d}.png"
            dest.write_bytes(processed)
        else:
            dest = images_dir / f"{new_id:04d}{ext}"
            shutil.copyfile(src, dest)

        try:
            rel = src.relative_to(dataset_root)
        except ValueError:
            rel = src
        data["id"].append(new_id)
        data["image"].append(str(dest))
        data["raw_image"].append(str(raw_dest))
        data["label"].append(sample["label"])
        data["label_name"].append(sample["label_name"])
        data["source_class"].append(sample["source_class"])
        data["source_filename"].append(src.name)
        data["source_relpath"].append(str(rel))
        data["split"].append(split_name)
        data["annotation"].append(sample.get("annotation", ""))

    affectnet_df = pd.DataFrame(data)
    affectnet_df.to_csv(output_dir / "affectnet_metadata.csv", index=False)

    if aws:
        import boto3

        s3 = boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        bucket_name = os.getenv("AWS_BUCKET")
        s3_prefix = (
            os.getenv("AWS_S3_PREFIX_IMAGES")
            or os.getenv("AWS_S3_PREFIX")
            or "affectnet/images"
        )
        for _, row in affectnet_df.iterrows():
            local_file_path = row["image"]
            s3.upload_file(
                local_file_path,
                bucket_name,
                f"{s3_prefix}/{Path(local_file_path).name}",
            )

    return affectnet_df
