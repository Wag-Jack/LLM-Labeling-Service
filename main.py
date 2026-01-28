from service_invocations.invoke_speech_recognition import run_speech_recognition
from data_management.edacc import load_edacc

# Amount of samples from each dataset to run through services
NUM_SAMPLES = 5

def main():
    while True:
        command = input("LLM as a Judge\n" \
                        "1.) Speech Recognition\n" \
                        "2.) Emotion Detection\n" \
                        "3.) Language Transltion\n" \
                        "4.) Exit\n" \
                        "Select: ")
        
        if 0 <= int(command) < 5:
            match int(command):
                case 1:
                    edacc_df = load_edacc(NUM_SAMPLES, True)
                    print(edacc_df.head)
                    run_speech_recognition(edacc_df)
                case 2:
                    pass
                case 3:
                    pass
                case 4:
                    break
        else:
            print("Invalid. Select an option between 1-4.")

if __name__ == "__main__":
    main()