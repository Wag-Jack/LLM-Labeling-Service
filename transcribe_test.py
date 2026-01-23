from dotenv import load_dotenv
from openai import OpenAI
import os
from pathlib import Path

load_dotenv()

# Initiate OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

interspeech_files = Path('Data/Interspeech2020-Accented-English-Speech-Recognition-Competition-Data')
interspeech_wavs = list(interspeech_files.rglob('*.wav'))

for wav in interspeech_wavs:
    # Transcribe audio file with Whisper model
    transcript = client.audio.transcriptions.create(
        file=open(wav, "rb"),
        model="whisper-1"
    )

    # Infer about the input
    analysis = client.responses.create(
        model="gpt-4o-mini",
        input=f"Determine what accent is being spoken in the following transcript: {transcript.text}"
    )

    print(analysis.output_text)