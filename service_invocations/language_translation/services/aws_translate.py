import os
from pathlib import Path

import boto3
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "language_translation" / "services"
RESULTS_FILE = "aws_trans.csv"


def run_aws_translation(europarl_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    translate = boto3.client(
        "translate",
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    data = {
        "id": [],
        "english_input": [],
        "service_output": [],
    }

    for _, row in europarl_data.iterrows():
        sample_id = row["id"]
        english = row["english"]
        print(f"AWS Translate: ({sample_id:04d}) {english}")

        french = translate.translate_text(
            Text=english,
            SourceLanguageCode="en",
            TargetLanguageCode="fr",
        ).get("TranslatedText", "")
        print(french)

        data["id"].append(f"aws_trans_{sample_id:04d}")
        data["english_input"].append(english)
        data["service_output"].append(french)

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(europarl_data):
    return run_aws_translation(europarl_data)
