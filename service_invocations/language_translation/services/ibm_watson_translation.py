from dotenv import load_dotenv
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.language_translator_v3 import LanguageTranslatorV3
import os
import pandas as pd
from pathlib import Path

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "language_translation" / "services"  # Task-scoped outputs.


def run_ibm_watson_translation(europarl_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / "ibm_trans.csv"

    api_key = os.getenv("IBM_WATSON_TRANSLATE_API_KEY")
    service_url = os.getenv("IBM_WATSON_TRANSLATE_URL")
    version = os.getenv("IBM_WATSON_TRANSLATE_VERSION", "2018-05-01")
    if not api_key or not service_url:
        raise ValueError(
            "IBM_WATSON_TRANSLATE_API_KEY and IBM_WATSON_TRANSLATE_URL must be set in environment."
        )

    authenticator = IAMAuthenticator(api_key)
    translator = LanguageTranslatorV3(
        version=version,
        authenticator=authenticator,
    )
    translator.set_service_url(service_url)

    data = {
        "id": [],
        "english_input": [],
        "service_output": []
    }

    for _, row in europarl_data.iterrows():
        sample_id = row["id"]
        english = row["english"]
        print(f"IBM Watson Translate: ({sample_id:04d}) {english}")

        response = translator.translate(
            text=english,
            source="en",
            target="fr",
        ).get_result()
        translations = response.get("translations", [])
        french = translations[0].get("translation", "") if translations else ""

        data["id"].append(f"ibm_trans_{sample_id:04d}")
        data["english_input"].append(english)
        data["service_output"].append(french)

    data["llm_judge_score"] = [0.0 for _ in data["id"]]

    ibm_df = pd.DataFrame(data)
    ibm_df.to_csv(results_path, index=False)
    return ibm_df
