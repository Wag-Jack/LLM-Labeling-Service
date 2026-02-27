from google.oauth2 import service_account
from google.cloud import translate
from html import unescape
import pandas as pd
from pathlib import Path

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "language_translation"  # Task-scoped outputs.

def run_gc_translation(europarl_data):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Grab secret key from credentials to tap into Google Cloud Translation
    cred_path = Path.cwd() / 'credentials'
    client_file = cred_path / 'speech_recognition/llm-as-a-judge_gc.json'
    credentials = service_account.Credentials.from_service_account_file(client_file)
    client = translate.TranslationServiceClient(credentials=credentials)

    # Data dictionary for collection
    data = {
        "id": [],
        "english_input": [],
        "service_output": []
    }

    for _, row in europarl_data.iterrows():
        # Run the service on the selected text
        id, english = row["id"], row["english"]
        print(f"Google Cloud Translate: ({id:04d}) {english}")

        response = client.translate_text(
            request = {
                "contents": [english],
                "parent": "projects/llm-as-a-judge-485501",
                "target_language_code": "fr",
                "source_language_code": "en"
            }
        )
    
        french = unescape(response.translations[0].translated_text)
        print(french)

        data["id"].append(f'gc_trans_{id:04d}')
        data["english_input"].append(english)
        data["service_output"].append(french)

    # Add in blank column for LLM judge score
    data["llm_judge_score"] = [0.0 for r in data["id"]]

    # Convert into DataFrame and save to CSV
    gc_trans_df = pd.DataFrame(data)
    gc_trans_df.to_csv(_RESULTS_DIR / "gc_trans.csv", index=False)

    return gc_trans_df
