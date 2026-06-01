from service_invocations.invoke_speech_recognition import run_speech_recognition
from service_invocations.invoke_language_translation import run_language_translation
from service_invocations.invoke_emotion_detection import run_emotion_detection
from data_management.edacc import load_edacc
from data_management.en_fr import load_en_fr
from data_management.vea import load_vea
from benchmark_prompts import run_all_prompts

# Amount of samples from each dataset to run through services
NUM_SAMPLES = 10
# Randomization controls. Set RANDOM_SEED to an integer for reproducible draws,
# or leave as None for a fresh random sample on every run.
RANDOMIZE_SAMPLES = True
RANDOM_SEED: int | None = None


def main():
    while True:
        command = input(
            "LLM Labeling\n"
            "1.) ASR - Automatic Speech Recognition\n"
            "2.) FER - Facial Emotion Detection\n"
            "3.) MT - Machine Translation\n"
            "4.) Benchmark all prompts\n"
            "5.) Exit\n"
            "Select: "
        )

        if 0 <= int(command) < 6:
            match int(command):
                case 1:
                    print("--- Gathering viable EdAcc samples ---")
                    edacc_df = load_edacc(NUM_SAMPLES, randomize=RANDOMIZE_SAMPLES, seed=RANDOM_SEED)
                    run_speech_recognition(edacc_df)
                case 2:
                    print("--- Retrieving Visual Emotional Analysis samples ---")
                    vea_df = load_vea(NUM_SAMPLES, randomize=RANDOMIZE_SAMPLES, seed=RANDOM_SEED)
                    run_emotion_detection(vea_df)
                case 3:
                    print("--- Retrieving EuroParl data pairs ---")
                    europarl_df = load_en_fr(NUM_SAMPLES, randomize=RANDOMIZE_SAMPLES, seed=RANDOM_SEED)
                    run_language_translation(europarl_df)
                case 4:
                    run_all_prompts(NUM_SAMPLES, randomize=RANDOMIZE_SAMPLES, seed=RANDOM_SEED)
                case 5:
                    break
        else:
            print("Invalid. Select an option between 1-5.")


if __name__ == "__main__":
    main()
