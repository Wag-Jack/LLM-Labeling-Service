import boto3
from datasets import Audio, load_dataset
import os
import pandas as pd
from pathlib import Path
import soundfile as sf

def load_edacc(amount=5, aws=False):
    # Ensure output directory exists
    output_dir = Path.cwd().parent / "Data" / "EdAcc"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the audio files from the dataset and get the audio path to put into transcription request
    edacc = load_dataset(
        "edinburghcstr/edacc",
        split=f"test[:{amount}]"
    ).cast_column("audio", Audio())

    # Data dictionary to help with DataFrame creation for comprehensive metadata of EdAcc
    data = {
        "id": [],
        "speaker": [],
        "text": [],
        "accent": [],
        "raw_accent": [],
        "gender": [],
        "l1": [],
        "audio": []
    }

    i = 1
    # speaker, text, accent, raw_accent, gender, l1, audio
    for row in edacc:
        # Copy everything except audio path to data dictionary
        data["id"].append(i)
        data["speaker"].append(row["speaker"])
        data["text"].append(row["text"])
        data["accent"].append(row["accent"])
        data["raw_accent"].append(row["raw_accent"])
        data["gender"].append(row["gender"])
        data["l1"].append(row["l1"])

        # Create local WAV file for each entry in dataset
        audio = row["audio"]
        wav_path = os.path.join(Path.cwd(), "Data", "EdAcc", "wav", f"{i:04d}.wav")
        data["audio"].append(wav_path)

        # Save the audio file locally if it doesn't already exist
        if not os.path.exists(wav_path):
            sf.write(wav_path, audio["array"], audio["sampling_rate"])

        i += 1

    edacc_df = pd.DataFrame(data)
    edacc_df.to_csv(Path.cwd() / "Data" / "EdAcc" / "edacc_metadata.csv", index=False)

    # TODO: Check if requested data already in S3 bucket, if so avoid re-uploading to cut down on time
    if aws:
        s3 = boto3.client('s3',
                          region_name=os.getenv("AWS_REGION"),
                          aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                          aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
                          )
        bucket_name = os.getenv("AWS_BUCKET")
        s3_prefix = os.getenv("AWS_S3_PREFIX")
        
        for _, row in edacc_df.iterrows():
            local_file_path = row["audio"]
            s3.upload_file(local_file_path, bucket_name, f"{s3_prefix}/{row['id']:04d}.wav")

    return edacc_df