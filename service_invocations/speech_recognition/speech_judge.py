import base64
from dotenv import load_dotenv
import json
from openai import OpenAI
import os
import pandas as pd
from pathlib import Path

load_dotenv()

def judge_transcripts(google_cloud, aws_transcribe, assemblyai, edacc_data):
    # Initiate OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Read each service's dataframe for its id and transcript
    gc_transcripts = dict(zip(google_cloud['id'], google_cloud['service_output']))
    aws_transcripts = dict(zip(aws_transcribe['id'], aws_transcribe['service_output']))
    assemblyai_transcripts = dict(zip(assemblyai['id'], assemblyai['service_output']))

    # Open metadata of EdAcc and retrieve all the audio paths as a list
    wav_files = edacc_data['audio'].tolist()
    ids = edacc_data['id'].tolist()

    # Prompt order labels with explicit service ordering
    prompt_orders = [
        ("123", ["gc", "aws", "aai"]),
        ("132", ["gc", "aai", "aws"]),
        ("213", ["aws", "gc", "aai"]),
        ("231", ["aws", "aai", "gc"]),
        ("312", ["aai", "gc", "aws"]),
        ("321", ["aai", "aws", "gc"]),
    ]

    # Data dictionary for judging output per prompt order
    results_by_order = {
        label: {
            "id": [],
            "gc_score": [],
            "aws_score": [],
            "aai_score": [],
            "llm_transcript": []
        }
        for label, _ in prompt_orders
    }

    for id, wav, gc, aws, aai in zip(ids, wav_files, gc_transcripts.keys(), aws_transcripts.keys(), assemblyai_transcripts.keys()):
        print(f"LLM Judging: {wav}")
        
        service_lines = {
            "gc": f"{gc}: {gc_transcripts[gc]}",
            "aws": f"{aws}: {aws_transcripts[aws]}",
            "aai": f"{aai}: {assemblyai_transcripts[aai]}",
        }

        for label, order in prompt_orders:
            ordered_services = "\n".join(service_lines[service_key] for service_key in order)
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
                      Do NOT mention that you need the WAV file, only give the JSON schema output.
                      Below are the services' transcript output:
                      {ordered_services}
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
            llm_output = json.loads(response.choices[0].message.content)

            # Append data to resultant data dictionary
            results_by_order[label]['id'].append(f'{id:04d}')
            results_by_order[label]['gc_score'].append(llm_output['google_cloud'])
            results_by_order[label]['aws_score'].append(llm_output['aws'])
            results_by_order[label]['aai_score'].append(llm_output['assemblyai'])
            results_by_order[label]['llm_transcript'].append(llm_output['llm_transcript'])

    for label in results_by_order:
        # For each service, update their respective score CSVs with LLM judge score
        google_cloud = google_cloud.drop(columns=['llm_judge_score'])
        google_cloud['llm_judge_score'] = results_by_order[label]['gc_score']
        google_cloud.to_csv(Path.cwd() / f'service_invocations/results/gc_stt_{label}.csv', index=False)

        aws_transcribe = aws_transcribe.drop(columns=['llm_judge_score'])
        aws_transcribe['llm_judge_score'] = results_by_order[label]['aws_score']
        aws_transcribe.to_csv(Path.cwd() / f'service_invocations/results/aws_stt_{label}.csv', index=False)

        assemblyai = assemblyai.drop(columns=['llm_judge_score'])
        assemblyai['llm_judge_score'] = results_by_order[label]['aai_score']
        assemblyai.to_csv(Path.cwd() / f'service_invocations/results/aa_stt_{label}.csv', index=False)

        # Create report for LLM scores
        judge_results = pd.DataFrame(results_by_order[label])
        judge_results.to_csv(f"service_invocations/results/speech_results_{label}.csv", index=False)
