# tiny-openai-whisper-api

OpenAI Whisper API-style local server, runnig on FastAPI. This is for companies behind proxies or security firewalls.

This API will be compatible with [OpenAI Whisper (speech to text) API](https://openai.com/blog/introducing-chatgpt-and-whisper-apis). See also  [Create transcription - API Reference - OpenAI API](https://platform.openai.com/docs/api-reference/audio/create).

Some of code has been copied from [whisper-ui](https://github.com/hayabhay/whisper-ui)


Now, this server emulates the following OpenAI APIs. 

- (whisper) [Speech to text - OpenAI API](https://platform.openai.com/docs/guides/speech-to-text)
- (chat) [Audio generation - OpenAI API](https://platform.openai.com/docs/guides/audio?audio-generation-quickstart-example=audio-in)

## Running Environment

This was built & tested on Python 3.10.9, Ubutu20.04/WSL2 but should also work on Python 3.9+.

- openai=1.55.0
- openai-whisper=20240930

## Setup

```bash
sudo apt install ffmpeg
pip install fastapi python-multipart pydantic uvicorn openai-whisper
# or pip install -r requirements.txt
```

or 

```bash
docker compose build
```

## Usage

### server
```bash
export WHISPER_MODEL=turbo  # 'turbo' if not supecified
export PYTHONPATH=.
uvicorn main:app --host 0.0.0.0
```

or 

```bash
docker compose up
```

### client

note: Authorization header is ignored.

example 1: typical usecase, identical to OpenAI Whisper API example

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F model="whisper-1" \
  -F file="@/path/to/file/openai.mp3"
```

example 2: set the output format as text, described in quickstart.

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F model="whisper-1" \
  -F file="@/path/to/file/openai.mp3" \
  -F response_format=text
```
## experimental: gpt-4o-audio-preview, chat-completions

currenly "output text only" mode is supported. 
If "output text and audio" is specified, the system makes "output text only" response.

- output text and audio
  - modalities=["text", "audio"], audio={"voice": "alloy", "format": "wav"}

```
completion = client.chat.completions.create(
    model="gpt-4o-audio-preview-2024-10-01",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"},
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

print(completion.choices[0].message.audio.transcript)
```

```python
ChatCompletion(id='chatcmpl-AXQt8BTMW4Gh1OcJ5qStVDNZGdzSq', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content=None, refusal=None, role='assistant', audio=ChatCompletionAudio(id='audio_6744555cc6d48190b67e70798ab606c3', data='{{base64-wav}}', expires_at=1732535148, transcript='The recording contains a voice stating that the sun rises in the east and sets in the west, a fact that has been observed by humans for thousands of years.'), function_call=None, tool_calls=None))], created=1732531546, model='gpt-4o-audio-preview-2024-10-01', object='chat.completion', service_tier=None, system_fingerprint='fp_130ac2f073', usage=CompletionUsage(completion_tokens=236, prompt_tokens=86, total_tokens=322, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=188, reasoning_tokens=0, rejected_prediction_tokens=0, text_tokens=48), prompt_tokens_details=PromptTokensDetails(audio_tokens=69, cached_tokens=0, text_tokens=17, image_tokens=0)))
```

- output text only
  - modalities=["text"]

```python
completion = client.chat.completions.create(
    model="gpt-4o-audio-preview-2024-10-01",
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

print(completion.choices[0].message.content)
```

```python
ChatCompletion(id='chatcmpl-AXTBlZypmtf1CCWrR6X5uX55r4VHY', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content="The recording contains a statement about the sun's movement, stating that the sun rises in the east and sets in the west, a fact that has been observed by humans for thousands of years.", refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1732540389, model='gpt-4o-audio-preview-2024-10-01', object='chat.completion', service_tier=None, system_fingerprint='fp_130ac2f073', usage=CompletionUsage(completion_tokens=38, prompt_tokens=86, total_tokens=124, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0, text_tokens=38), prompt_tokens_details=PromptTokensDetails(audio_tokens=69, cached_tokens=0, text_tokens=17, image_tokens=0)))
```

### experimental: Dify integration

- LLM registration ... OK
- audio file transcription ... NG
  - only open:gpt-4o-audio-preview is supported for file transcription on dify-0.12.1?
  - dify-0.12.1 doesn't pass an user prompot which contains audio file (data) to this tiny-openai-whisper-api server.
    - Probably on dify-0.12.1, "native (audio) file processing capabilities" is available only for openai:gpt-4o-audio-preview. How can we give these feature to openai-compatible LLMs?

## TODO

- more inference parameters should be supported. only `temperature` is supported.
- text prompt (to whisper module) should be supported. currently text prompt is ignored.
- some of reponse property values are dummy (static).
- 'speech-to-text' chat completion available on dify

## License

Whisper is licensed under MIT. Everything else by [morioka](https://github.com/morioka) is licensed under MIT.
