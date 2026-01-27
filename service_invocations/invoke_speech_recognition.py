from service_invocations.speech_recognition.assemblyai import run_assemblyai
from service_invocations.speech_recognition.aws_transcribe import run_aws_transcribe
from service_invocations.speech_recognition.google_cloud import run_gc_stt

def run_speech_recognition(edacc_df):
    # Run EdAcc samples through each speech recognition service
    
    # Google Cloud Speech-to-Text
    #gc_stt = run_gc_stt(edacc_df)

    # AWS Transcribe
    #aws_stt = run_aws_transcribe(edacc_df)

    # AssemblyAI Speech-to-Text
    aai_stt = run_assemblyai(edacc_df)