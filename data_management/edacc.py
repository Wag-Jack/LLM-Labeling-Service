import boto3
from datasets import Audio, load_dataset
import os
import pandas as pd
from pathlib import Path
import soundfile as sf

def load_edacc(amount=50, min_duration=2.0, max_duration=10.0, aws=True):
    # Ensure output directory exists
    output_dir = Path.cwd() / "Data" / "EdAcc"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "wav", exist_ok=True)
    
    # Load the audio files from the dataset and get the audio path to put into transcription request
    edacc = load_dataset(
        "edinburghcstr/edacc",
        split=f"test[:{amount*10}]"
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
        "duration": [],
        "audio": []
    }

    # Since there's issues with sending small audio to each transcription service, we'll have to choose data with appropriate length
    curr_amnt, i = 0, 0
    while curr_amnt < amount:        
        # Check if the audio sample reaches a certain duration of time
        row = edacc[i]
        audio = row["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]

        if min_duration <= duration <= max_duration:
            # Copy everything except audio path to data dictionary
            data["id"].append(curr_amnt+1)
            data["speaker"].append(row["speaker"])
            data["text"].append(row["text"])
            data["accent"].append(row["accent"])
            data["raw_accent"].append(row["raw_accent"])
            data["gender"].append(row["gender"])
            data["l1"].append(row["l1"])
            data["duration"].append(duration)

            # Create local WAV file for each entry in dataset
            audio = row["audio"]
            wav_path = os.path.join(Path.cwd(), "Data", "EdAcc", "wav", f"{curr_amnt+1:04d}.wav")
            data["audio"].append(wav_path)

            # Save the audio file locally if it doesn't already exist
            if not os.path.exists(wav_path):
                sf.write(wav_path, audio["array"], audio["sampling_rate"])

            curr_amnt += 1

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