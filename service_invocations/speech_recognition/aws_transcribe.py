import boto3
from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path
import requests

load_dotenv()

S3_URI = "s3://llm-as-a-judge-edacc-storage"

# Start the transcription job for specific audio files
def start_transcription_job(id, transcribe):
    transcript_job_name = f"{id:04d}_job"
    response = transcribe.start_transcription_job(
        TranscriptionJobName=transcript_job_name,
        Media={"MediaFileUri": f"{S3_URI}/edacc/audio/{id:04d}.wav"},
        MediaFormat="wav",
        LanguageCode="en-US"
    )

    return transcript_job_name

# Poll for the transcription job to complete
def wait_for_job(job_name, transcribe):
    while True:
        response = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        status = response['TranscriptionJob']['TranscriptionJobStatus']
        if status == 'COMPLETED':
            uri = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
            transcribe.delete_transcription_job(TranscriptionJobName=job_name)
            return uri
        elif status != 'IN_PROGRESS':
            return None
        
# Method to extract the transcript from job's resulting URI
def retrieve_transcript(transcirpt_uri):
    response = requests.get(transcirpt_uri).json()
    transcript = response['results']['transcripts'][0]['transcript']
    return transcript

def run_aws_transcribe(edacc_data):
    # Initialize AWS clients to Transcribe
    transcribe = boto3.client('transcribe',
                              region_name=os.getenv("AWS_REGION"),
                              aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                              aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
                              )

    # Data dictionary to hold results
    data = {
        "id": [],
        "wav_file": [],
        "service_output": []
    }

    for _, row in edacc_data.iterrows():
        # Run the service on the selected file
        audio_file = row["audio"]
        id = row["id"]

        # Run transcription job and return transcript upon completion
        job = start_transcription_job(id, transcribe)
        response_uri = wait_for_job(job, transcribe)
        transcript = retrieve_transcript(response_uri)

        data["id"].append(f"aws_stt_{id:04d}")
        data["wav_file"].append(audio_file)
        print(transcript, end='\n\n')
        data["service_output"].append(transcript)

    # Add in blank column for LLM judge score
    data["llm_judge_score"] = [0.0 for r in data["id"]]

    # Convert into DataFrame and save to CSV
    aws_stt_df = pd.DataFrame(data)
    aws_stt_df.to_csv("service_invocations/results/aws_stt.csv", index=False)
    return aws_stt_df