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
    print(wav)

    # Transcribe audio file with Whisper model
    transcript = client.audio.transcriptions.create(
        file=open(wav, "rb"),
        model="whisper-1"
    )

    # Infer about the input
    analysis = client.responses.create(
        model="gpt-4o-mini",
        input=f"Determine what accent is being spoken in the following transcript: {transcript.text}" \
        "Choose from the following options and only give a confidence score as reasoning behind your answer:" \
        "Chinese (CHN), Indian (IND), Japanese (JPN), Korean (KOR), Russian (RU), Portuguese (PT), Spanish (ES), American (US)"
    )

    print(analysis.output_text)