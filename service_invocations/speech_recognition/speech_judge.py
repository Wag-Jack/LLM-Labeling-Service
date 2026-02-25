import base64
from dotenv import load_dotenv
import json
from openai import OpenAI
import os
import pandas as pd
from pathlib import Path

load_dotenv()

from registry.speech_recognition import load_service_registry


def _normalize_id(value) -> str:
    if isinstance(value, (int, float)) and float(value).is_integer():
        return f"{int(value):04d}"
    return str(value)


def _load_results_for_service(results_dir: Path, results_file: str) -> pd.DataFrame:
    return pd.read_csv(results_dir / results_file)


def judge_transcripts(results_by_service, edacc_data, use_existing=False,
                      results_dir=None, config_path: Path | None = None):
    if results_dir is None:
        results_dir = Path.cwd() / "service_invocations" / "results"

    service_registry = load_service_registry(config_path)
    active_services = {
        name: entry
        for name, entry in service_registry.items()
        if entry.get("task") == "stt"
    }
    if results_by_service:
        active_services = {
            name: entry
            for name, entry in active_services.items()
            if name in results_by_service
        }
    if not active_services:
        raise ValueError("No speech services found in service_registry.")

    if use_existing:
        results_by_service = {
            name: _load_results_for_service(results_dir, entry["results_file"])
            for name, entry in active_services.items()
        }
    if not results_by_service:
        raise ValueError("No speech results provided for judging.")

    # Initiate OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Read each service's dataframe for its id and transcript
    transcripts_by_service = {}
    for name, df in results_by_service.items():
        if "id" not in df.columns or "service_output" not in df.columns:
            raise ValueError(f"Missing required columns for {name}.")
        normalized = dict(zip(df["id"].map(_normalize_id), df["service_output"]))
        transcripts_by_service[name] = normalized

    # Open metadata of EdAcc and retrieve all the audio paths as a list
    wav_files = edacc_data['audio'].tolist()
    ids = edacc_data['id'].tolist()

    # Data dictionary for judging output
    data = {
        "id": [],
        "scores": [],
        "llm_transcript": []
    }

    for id, wav in zip(ids, wav_files):
        id_key = _normalize_id(id)
        print(f"LLM Judging: {wav}")

        service_blocks = []
        for name, transcripts in transcripts_by_service.items():
            transcript = transcripts.get(id_key, "")
            service_blocks.append(f"{name}: {transcript}")

        # Set up prompt for the LLM
        prompt = f"""
                  You are acting as a judge for similar web services that are used for speech recognition.
                  Each service receives an input of a WAV file and will output a textual transcript of the audio file.
                  Your job is the following:
                  1. Listen to the audio file given.
                  2. Give your textual transcript of the given audio file that you will use to compare each service's output.
                  3. For each service, give a score (1.0-10.0, scoring in intervals of 0.1) on what you believe is the accuracy of each output.
                  You MUST return ONLY valid JSON. 
                  Do not include markdown, code fences, or explanations.
                  If you violate this, the output will be discarded.
                  JSON schema:
                  {{
                    "llm_transcript": string|null,
                    "scores": {{
                      "<service_name>": number
                    }}
                  }}
                  If you do not receive the WAV file, enter llm_transcript as 'n/a' and the scores as -1.
                  Do NOT mention that you need the WAV file, only give the JSON schema output.
                  Below are the services' transcript output:
                  {'\n'.join(service_blocks)}
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
        default_scores = {name: -1 for name in transcripts_by_service.keys()}
        try:
            llm_output = json.loads(response.choices[0].message.content)
            scores = llm_output.get('scores', {}) if isinstance(llm_output, dict) else {}
        except json.JSONDecodeError:
            llm_output = {"llm_transcript": "n/a"}
            scores = {}

        # Append data to resultant data dictionary
        data['id'].append(id_key)
        data['scores'].append({**default_scores, **scores})
        data['llm_transcript'].append(llm_output.get('llm_transcript', "n/a"))

    # For each service, update their respective score CSVs with LLM judge score
    scores_by_service = {}
    for id_key, score_map in zip(data["id"], data["scores"]):
        for name, score in score_map.items():
            scores_by_service.setdefault(name, {})[id_key] = score

    for name, df in results_by_service.items():
        score_map = scores_by_service.get(name, {})
        df = df.drop(columns=["llm_judge_score"], errors="ignore")
        df["llm_judge_score"] = df["id"].map(_normalize_id).map(score_map).fillna(-1)
        results_file = active_services.get(name, {}).get("results_file")
        if not results_file:
            continue
        df.to_csv(results_dir / results_file, index=False)

    # Create report for LLM scores
    judge_results = pd.DataFrame({
        "id": data["id"],
        "llm_transcript": data["llm_transcript"],
        "scores": data["scores"],
    })
    judge_results.to_csv(results_dir / "speech_results.csv", index=False)
