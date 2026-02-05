from dotenv import load_dotenv
import json
from openai import OpenAI
import os
import pandas as pd
from pathlib import Path

load_dotenv

def judge_translations(google_cloud, aws_translate, microsoft_translate, europarl_data):
    # Initiate OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Read each service's DataFrame for its id and French output
    gc_french = dict(zip(google_cloud['id'], google_cloud['service_output']))
    aws_french = dict(zip(aws_translate['id'], aws_translate['service_output']))
    ms_french = dict(zip(microsoft_translate['id'], microsoft_translate['service_output']))

    # Retrieve all input from metadata as a list
    english_input = europarl_data['english'].tolist()
    ids = europarl_data['id'].tolist()

    # Data dictionary for judging output
    data = {
        "id": [],
        "gc_score": [],
        "aws_score": [],
        "ms_score": [],
        "llm_translation": []
    }

    for id, eng, gc, aws, ms in zip(ids, english_input, gc_french.keys(), aws_french.keys(), ms_french.keys()):
        print(f"LLM Judging: ({id:04d}) {eng}")

        # Set up prompt for the LLM
        prompt = f"""
                 You are acting as a judge for similar web services that are used for language translation.
                 Each service receives an input of English text and will output a French translation.
                 Your job is the following:
                 1. Read in the English text.
                 2. Give your French translation of the given English text that you will use to compare each service's output.
                 3. For each service, give a score (1.0-10.0, scoring in intervals of 0.1) on what you believe is the accuracy of each output.
                 You MUST return the output as a JSON object in the following format:
                 LLM Transcript: (your transcript)
                 Google Cloud Translate: (score from 1.0-10.0)
                 AWS Translate: (score from 1.0-10.0)
                 Microsoft Azure Translator: (score from 1.0-10.0)
                 You MUST return ONLY valid JSON.
                 Do not include markdown, code fences, or explanations.
                 If you violate this, the output will be discarded.
                 JSON scheme:
                 {{
                    "llm_translation": string|null,
                    "google_cloud": number,
                    "aws": number,
                    "microsoft": number
                 }}
                 If you do not recieve the English input, enter the translation as 'n/a' and the scores as -1.
                 Do NOT mention that you need the English input, only give the JSON schema output.
                 Below are the services' transcript output:
                 {gc}: {gc_french[gc]}
                 {aws}: {aws_french[aws]}
                 {ms}: {ms_french[ms]}
                 """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            modalities=['text'],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )

        # Compile JSON object from LLM output
        print(f"{response.choices[0].message.content}")
        llm_output = json.loads(response.choices[0].message.content)

        # Append data to resultant data dictionary
        data['id'].append(f'{id:04d}')
        data['gc_score'].append(llm_output['google_cloud'])
        data['aws_score'].append(llm_output['aws'])
        data['ms_score'].append(llm_output['microsoft'])
        data['llm_translation'].append(llm_output['llm_translation'])

    # For each service, update their respective score CSVs with LLM judge score
    google_cloud = google_cloud.drop(columns=['llm_judge_score'])
    google_cloud['llm_judge_score'] = data['gc_score']
    google_cloud.to_csv(Path.cwd() / 'service_invocations/results/gc_trans.csv', index=False)

    aws_translate = aws_translate.drop(columns=['llm_judge_score'])
    aws_translate['llm_judge_score'] = data['aws_score']
    aws_translate.to_csv(Path.cwd() / 'service_invocations/results/aws_trans.csv', index=False)

    microsoft_translate = microsoft_translate.drop(columns=['llm_judge_score'])
    microsoft_translate['llm_judge_score'] = data['ms_score']
    microsoft_translate.to_csv(Path.cwd() / 'service_invocations/results/ms_trans.csv', index=False)

    # Create report for LLM scores
    judge_results = pd.DataFrame(data)
    judge_results.to_csv("service_invocations/results/language_results.csv", index=False)