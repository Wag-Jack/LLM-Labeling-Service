from dotenv import load_dotenv
import json
import os
import pandas as pd
from pathlib import Path
import requests
import uuid

load_dotenv()

def run_micro_translation(europarl_data):
    # Establish connection to Microsoft Translator
    key = os.getenv("MICROSOFT_ACCESS_KEY")
    endpoint = "https://api.cognitive.microsofttranslator.com"
    location = "eastus"
    path = '/translate'
    constructed_url = endpoint + path
    params = {
        'api-version': '3.0',
        'from': 'en',
        'to': 'fr'
    }
    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # Data dictionary for collection
    data = {
        'id': [],
        'english_input': [],
        "service_output": []
    }

    for _, row in europarl_data.iterrows():
        # Run the service on the selected text
        id, english = row["id"], row["english"]
        print(f"Microsoft Translator: ({id:04d}) {english}")

        # You can pass more than one object in body.
        body = [{'text': english}]

        request = requests.post(constructed_url, params=params, headers=headers, json=body)
        response = request.json()
        request = json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': '))
        french = response[0]["translations"][0]["text"]

        data["id"].append(f'ms_trans_{id:04d}')
        data["english_input"].append(english)
        data["service_output"].append(french)

    # Add in blank column for LLM judge score
    data["llm_judge_score"] = [0.0 for r in data["id"]]

    # Convert into DataFrame and save to CSV
    ms_trans_df = pd.DataFrame(data)
    ms_trans_df.to_csv("service_invocations/results/ms_trans.csv", index=False)