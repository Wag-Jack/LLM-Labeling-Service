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

# Load the audio file from the dataset
edacc = load_dataset("edinburghcstr/edacc")
sample = edacc['test'][0]
sample_path = sample['audio']['path']

# Convert audio file into proper format (CONVERT TO GCS)
audio_array, sample_rate = sf.read(sample)
pcm16 = (audio_array * 32767).astype(np.int16).tobytes()
audio = speech.RecognitionAudio(content=pcm16)

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="en-US")

response = client.recognize(config=config, audio=audio)
print(response)