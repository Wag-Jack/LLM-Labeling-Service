from openai import OpenAI

# Initiate OpenAI client
client = OpenAI()

# Transcribe audio file with Whisper model
transcript = client.audio.transcriptions.create(
    file=open("audio.mp3","rb"),
    model="whisper-1"
)

# Infer about the input
analysis = client.chat.completions.create(
    model="gpt-4-turbo",
    input=f"Determine what accent is being spoken in the following transcript: {transcript.text}"
)

print(analysis.choices[0].message.content)