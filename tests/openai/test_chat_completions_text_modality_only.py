import base64
import requests
from openai import OpenAI

client = OpenAI()
client = OpenAI(base_url='http://localhost:8000/v1')

# Fetch the audio file and convert it to a base64 encoded string
url = "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav"
response = requests.get(url)
response.raise_for_status()
wav_data = response.content
encoded_string = base64.b64encode(wav_data).decode('utf-8')

completion = client.chat.completions.create(
    model="gpt-4o-audio-preview-2024-10-01",
#    modalities=["text", "audio"],
#    audio={"voice": "alloy", "format": "wav"},
    modalities=["text"],
    messages=[
        {
            "role": "user",
            "content": [
                { 
                    "type": "text",
                    "text": "What is in this recording?"
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_string,
                        "format": "wav"
                    }
                }
            ]
        },
    ]
)

print(completion)
print(completion.choices[0].message.content)
