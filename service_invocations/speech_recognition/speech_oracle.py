import base64
from dotenv import load_dotenv
import json
from openai import OpenAI
import os
import pandas as pd
from pathlib import Path

load_dotenv()


def generate_oracle_transcripts(edacc_data, use_existing=False, results_path=None):
    if results_path is None:
        results_path = Path.cwd() / "service_invocations" / "results" / "speech_oracle.csv"

    if use_existing and results_path.exists():
        return pd.read_csv(results_path)

    # Initiate OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Open metadata of EdAcc and retrieve all the audio paths as a list
    wav_files = edacc_data['audio'].tolist()
    ids = edacc_data['id'].tolist()

    # Data dictionary for LLM output
    data = {
        "id": [],
        "llm_oracle": []
    }

    for id, wav in zip(ids, wav_files):
        print(f"LLM Oracle Transcript: {wav}")

        prompt = f"""
                  Please give me a transcript for the following audio file.
                  You MUST return ONLY valid JSON. Do not include markdown, code fences, or explanations.
                  JSON schema:
                  {{
                    "llm_oracle": string|null
                  }}
                  If you do not receive the WAV file, enter llm_oracle as 'n/a'.
                  Do NOT mention that you need the WAV file, only give the JSON schema output.
                  If you violate this, the output will be discarded.
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

        # Compile JSON object from LLM output
        print(f"{response.choices[0].message.content}")
        try:
            llm_output = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            llm_output = {"llm_oracle": "n/a"}

        # Append data to resultant data dictionary
        data['id'].append(f'{id:04d}')
        data['llm_oracle'].append(llm_output.get('llm_oracle', "n/a"))

    # Create report for LLM transcripts
    oracle_results = pd.DataFrame(data)
    oracle_results.to_csv(results_path, index=False)
    return oracle_results
