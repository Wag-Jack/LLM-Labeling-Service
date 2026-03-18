import os
import dashscope
from dotenv import load_dotenv

load_dotenv()

dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
# This is the base URL for the Singapore region.
messages = [
    {'role': 'system', 'content': [
        {"audio": "/Users/jackwagner/Documents/GitHub/LLM-Labeling-Service/Data/EdAcc/wav/0001.wav"},
        {"text": """
                      Please give me a transcript for the following audio file.
                      You MUST return ONLY valid JSON. Do not include markdown, code fences, or explanations.
                      JSON schema:
                      {
                        "llm_oracle": string|null
                      }
                      If you do not receive the WAV file, enter llm_oracle as 'n/a'.
                      Do NOT mention that you need the WAV file, only give the JSON schema output.
                      If you violate this, the output will be discarded.
                      """}
    ]}
]

response = dashscope.Generation.call(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model="qwen3-omni-flash-realtime-2025-12-01",
    messages=messages,
    result_format='message'
    )
print(response)