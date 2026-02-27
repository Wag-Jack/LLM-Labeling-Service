import boto3
from dotenv import load_dotenv
import os
import pandas as pd
import requests
from time import time
from pathlib import Path

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "language_translation"  # Task-scoped outputs.

def run_aws_translation(europarl_data):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Initialize AWS client to Translate
    translate = boto3.client('translate',
                             region_name=os.getenv("AWS_REGION"),
                             aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                             aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    
    # Data dictionary to hold results
    data = {
        "id": [],
        "english_input": [],
        "service_output": []
    }

    for _, row in europarl_data.iterrows():
        # Run service on selected English input
        id, english = row["id"], row["english"]
        print(f"AWS Translate: ({id:04d}) {english}")

        # Run translation job and return translation upon completion
        french = translate.translate_text(
            Text=english,
            SourceLanguageCode='en',
            TargetLanguageCode='fr'
        ).get('TranslatedText')

        data["id"].append(f"aws_trans_{id:04d}")
        data["english_input"].append(english)
        print(french)
        data["service_output"].append(french)

    # Add in blank column for LLM judge score
    data["llm_judge_score"] = [0.0 for r in data["id"]]

    # Convert into DataFrame and save to CSV
    aws_trans_df = pd.DataFrame(data)
    aws_trans_df.to_csv(_RESULTS_DIR / "aws_trans.csv", index=False)
    return aws_trans_df
