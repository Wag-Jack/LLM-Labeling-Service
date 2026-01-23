import base64
from dotenv import load_dotenv
from openai import OpenAI
import os
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

jpg_files = Path('Data/FER-2013/test_subset')
fer_jpgs = list(jpg_files.rglob('*.jpg'))

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Getting the Base64 strings
base64_images = {}
for jpg in fer_jpgs:
    base64_images[jpg] = (encode_image(jpg))

question = "What emotion is being displaying in this image? Please give me a single choice between anger, disgust, fear, happiness, neutral, sad, or surprise, along with a confidence score to back up your answer."

for jpg in base64_images:
    print(jpg)
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": question},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_images[jpg]}",
                    },
                ],
            }
        ],
    )

    print(response.output_text)