import boto3
from datasets import ClassLabel, Image, load_dataset
import os
import random
import pandas as pd
from pathlib import Path

"""
Labels:
0 = anger
1 = contempt
2 = disgust
3 = fear
4 = happy
5 = neutral
6 = sad
7 = surprise
"""

def load_vea(amount=50, aws=True, randomize=True, seed=None):
    # Ensure output directory exists
    output_dir = Path.cwd() / "Data" / "VEA"
    images_dir = output_dir / "images"
    os.makedirs(images_dir, exist_ok=True)

    # Load the full train split so randomization draws uniformly across all emotions
    # rather than over-sampling whichever class happens to lead the split.
    split = "train" if randomize else f"train[:{amount}]"
    dataset = load_dataset(
        "FastJobs/Visual_Emotional_Analysis",
        split=split,
    ).cast_column("image", Image())

    indices = list(range(len(dataset)))
    if randomize:
        rng = random.Random(seed)
        rng.shuffle(indices)
    indices = indices[:amount]

    # Data dictionary to help with DataFrame creation for comprehensive metadata
    data = {
        "id": [],
        "image": [],
        "label": [],
    }

    for new_id, dataset_idx in enumerate(indices, start=1):
        row = dataset[dataset_idx]
        # Copy everything except image path to data dictionary
        data["id"].append(new_id)
        data["label"].append(row["label"])

        # Create local png file for each entry in dataset
        image = row["image"]
        image_path = images_dir / f"{new_id:04d}.png"
        data["image"].append(str(image_path))

        # Save image file locally if it does not exist (overwrite when randomized to
        # avoid carrying stale pixels from a previous, differently-seeded sample).
        if randomize or not image_path.exists():
            image.save(image_path)

    vea_df = pd.DataFrame(data)
    vea_df.to_csv(output_dir / "vea_metadata.csv", index=False)

    # TODO: Check if requested data already in S3 bucket, if so avoid re-uploading to cut down on time
    if aws:
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
            or "vea/images"
        )

        for _, row in vea_df.iterrows():
            local_file_path = row["image"]
            s3.upload_file(
                local_file_path,
                bucket_name,
                f"{s3_prefix}/{int(row['id']):04d}.png",
            )

    return vea_df
