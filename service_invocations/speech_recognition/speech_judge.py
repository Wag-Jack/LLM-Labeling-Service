import base64
from dotenv import load_dotenv
from openai import OpenAI
import os
import pandas as pd
from pathlib import Path


def judge_transcripts(google_cloud, aws_transcribe, assemblyai, edacc_data):
    # Initiate OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # TEMPORARY: Read each CSV file and get a DataFrame from its contents
    google_cloud = pd.read_csv(Path.cwd() / 'service_invocations/results/gc_stt.csv')
    aws_transcribe = pd.read_csv(Path.cwd() / 'service_invocations/results/aws_stt.csv')
    assemblyai = pd.read_csv(Path.cwd() / 'service_invocations/results/aa_stt.csv')

    # Read each service's dataframe for its id and transcript
    gc_transcripts = dict(zip(google_cloud['id'], google_cloud['service_output']))
    aws_transcripts = dict(zip(aws_transcribe['id'], aws_transcribe['service_output']))
    assemblyai_transcripts = dict(zip(assemblyai['id'], assemblyai['service_output']))

    # Open metadata of EdAcc and retrieve all the audio paths as a list
    wav_files = edacc_data['audio'].tolist()
    ids = edacc_data['id'].tolist()

    for wav, gc, aws, aai in zip(wav_files, gc_transcripts.keys(), aws_transcripts.keys(), assemblyai_transcripts.keys()):
        # Set up prompt for the LLM
        prompt = f"""
                  You are acting as a judge for similar web services that are used for speech recognition.
                  Each service receives an input of a WAV file and will output a textual transcript of the audio file.
                  Your job is the following:
                  1. Listen to the audio file given.
                  2. Give your textual transcript of the given audio file that you will use to compare each service's output.
                  3. For each service, give a score (1.0-10.0, scoring in intervals of 0.1) on what you believe is the accuracy of each output.
                  You MUST return the output as a JSON object in the following format:
                  LLM Transcript: (your transcript)
                  Google Cloud STT: (score from 1.0-10.0)
                  AWS Transcribe: (score from 1.0-10.0)
                  AssemblyAI STT: (score from 1.0-10.0)
                  You MUST return ONLY valid JSON.
                  Do not include markdown, code fences, or explanations.
                  If you violate this, the output will be discarded.
                  JSON schema:
                  {{
                    "llm_transcript": string|null,
                    "google_cloud": number,
                    "aws": number,
                    "assemblyai": number
                  }}
                  If you do not receive the WAV file, enter llm_transcript as 'n/a' and the scores as -1.
                  Below are the services' transcript output:
                  {gc}: {gc_transcripts[gc]}
                  {aws}: {aws_transcripts[aws]}
                  {aai}: {assemblyai_transcripts[aai]}
                  """
        
        # Open designated audio file
        with open(wav, 'rb') as f:
            audio_bytes = f.read()

        audio = base64.b64encode(audio_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-audio",
            modalities=['text'],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
                    ]
                }
            ]
        )

        llm_output = response.choices[0].message.content

        print(f"{response.choices[0].message.content}")