from google.oauth2 import service_account
from google.cloud import speech
import pandas as pd
from pathlib import Path
import time

# Results folder organized by task.
_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "speech_recognition"  # Task-scoped outputs.

# Helper function to deal with multiple transcript results from service
def combine_response(response_json):
    combined_transcript = ""
    for result in response_json.results:
        combined_transcript += result.alternatives[0].transcript + " "
    return combined_transcript.strip()

def run_gc_stt(edacc_data):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Grab secret key from credentials to tap into Google Cloud Speech-to-Text API
    cred_path = Path.cwd() / 'credentials'
    client_file = cred_path / 'speech_recognition/llm-as-a-judge_gc.json'
    credentials = service_account.Credentials.from_service_account_file(client_file)
    client = speech.SpeechClient(credentials=credentials)
    
    data = {
        "id": [],
        "wav_file": [],
        "service_output": [],
        "latency_ms": [],
    }

    for _, row in edacc_data.iterrows():
        # Run the service on the selected file
        audio_file = row["audio"]
        print(f"Google Cloud STT: {audio_file}")
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code="en-US"
        )

        start_time = time.perf_counter()
        response = client.recognize(config=config, audio=audio)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        data["id"].append(f"gc_stt_{row['id']:04d}")
        data["wav_file"].append(row["audio"])
        data["latency_ms"].append(round(latency_ms, 2))
        
        formatted_response = combine_response(response)
        print(formatted_response, end='\n')
        
        data["service_output"].append(formatted_response)

    # Add in blank column for LLM judge score
    data["llm_judge_score"] = [0.0 for r in data["id"]]

    # Convert into DataFrame and save to CSV
    gc_stt_df = pd.DataFrame(data)
    gc_stt_df.to_csv(_RESULTS_DIR / "gc_stt.csv", index=False)
    return gc_stt_df
