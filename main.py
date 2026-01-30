from service_invocations.invoke_speech_recognition import run_speech_recognition
from service_invocations.invoke_language_translation import run_language_translation
from data_management.edacc import load_edacc
from data_management.en_fr import load_en_fr

# Amount of samples from each dataset to run through services
NUM_SAMPLES = 50

def main():
    while True:
        command = input("LLM as a Judge\n" \
                        "1.) Speech Recognition\n" \
                        "2.) Emotion Detection\n" \
                        "3.) Language Translation\n" \
                        "4.) Exit\n" \
                        "Select: ")
        
        if 0 <= int(command) < 5:
            match int(command):
                case 1:
                    print("--- Gathering viable EdAcc samples ---")
                    edacc_df = load_edacc(NUM_SAMPLES)
                    run_speech_recognition(edacc_df)
                case 2:
                    pass
                case 3:
                    print("---  Retrieving EuroParl data pairs ---")
                    europarl_df = load_en_fr(NUM_SAMPLES)
                    run_language_translation(europarl_df)
                    pass
                case 4:
                    break
        else:
            print("Invalid. Select an option between 1-4.")

if __name__ == "__main__":
    main()