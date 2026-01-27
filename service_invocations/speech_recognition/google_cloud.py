from datasets import load_dataset, Audio
from google.oauth2 import service_account
from google.cloud import speech
import numpy as np
from pathlib import Path
import soundfile as sf

# Grab secret key from credentials to tap into Google Cloud Speech-to-Text API
cred_path = Path.cwd().parent.parent / 'credentials'
client_file = cred_path / 'speech_recognition/llm-as-a-judge_gc.json'
credentials = service_account.Credentials.from_service_account_file(client_file)
client = speech.SpeechClient(credentials=credentials)

# Load the audio file from the dataset and get the audio path to put into transcription request
edacc = load_dataset(
    "edinburghcstr/edacc",
    split="validation[:5]"
).cast_column("audio", Audio(decode=False))

sample_path = edacc[0]['audio']['path']

# Open audio file as bytes
with open(sample_path, 'rb') as f:
    audio_bytes = f.read()

audio = speech.RecognitionAudio(content=audio_bytes)
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    #sample_rate_hertz=16000,
    language_code="en-US"
)

response = client.recognize(config=config, audio=audio)
print(response)