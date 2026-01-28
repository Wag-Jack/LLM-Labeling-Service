from service_invocations.speech_recognition.assemblyai import run_assemblyai
from service_invocations.speech_recognition.aws_transcribe import run_aws_transcribe
from service_invocations.speech_recognition.google_cloud import run_gc_stt
from service_invocations.speech_recognition.speech_judge import judge_transcripts

def run_speech_recognition(edacc_df):
    # Run EdAcc samples through each speech recognition service
    
    # Google Cloud Speech-to-Text
    print('--- Google Cloud Speech-to-Text ---')
    #gc_stt = run_gc_stt(edacc_df)

    # AWS Transcribe
    print('--- AWS Transcribe Speech-to-Text ---')
    #aws_stt = run_aws_transcribe(edacc_df)

    # AssemblyAI Speech-to-Text
    print('--- AssemblyAI Speech-to-Text ---')
    #aai_stt = run_assemblyai(edacc_df)

    #judge_transcripts(gc_stt, aws_stt, aai_stt)
    print()
    judge_transcripts(None, None, None, edacc_df)