from service_invocations.speech_recognition.assemblyai import run_assemblyai
from service_invocations.speech_recognition.aws_transcribe import run_aws_transcribe
from service_invocations.speech_recognition.google_cloud import run_gc_stt

def run_speech_recognition(edacc_df):
    # Run EdAcc samples through each speech recognition service
    gc_stt = run_gc_stt(edacc_df)

