import assemblyai as aai
from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path

load_dotenv()

def run_assemblyai(edacc_data):
    # Establish connection with AsssemblyAI API
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

    # Data dictionary to hold results
    data = {
        "id": [],
        "wav_file": [],
        "service_output": [],
    }

    for _, row in edacc_data.iterrows():
        # Run the service on the selected file
        audio_file = row["audio"]
        print(audio_file)

        # Upload the audio file to AssemblyAI
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_file)

        data["id"].append(f"aa_stt_{row['id']:04d}")
        data["wav_file"].append(row["audio"])
        print(transcript.text, end='\n\n')

        data["service_output"].append(transcript.text)

    # Add in blank column for LLM judge score
    data["llm_judge_score"] = [0.0 for r in data["id"]]

    # Convert into DataFrame and save to CSV
    aa_stt_df = pd.DataFrame(data)
    aa_stt_df.to_csv("service_invocations/results/aa_stt.csv", index=False)
    return aa_stt_df