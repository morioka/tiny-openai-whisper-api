# tiny-openai-whisper-api

OpenAI Whisper API-style local server, runnig on FastAPI. This is for companies behind proxies or security firewalls.

This may be compatible with [OpenAI Whisper (speech to text) API](https://platform.openai.com/docs/guides/speech-to-text/quickstart).

## server
```bash
$ export PYTHONPATH=.
$ uvicorn main:app --host 0.0.0.0
```

## client
```bash
$ curl http://127.0.0.1:8000/v1/audio/transcriptions  -H "Content-Type: multipart/form-data"  -F model="whisper-1" -F file="@sample.mp4"
$ # set the output format as text
$ curl http://127.0.0.1:8000/v1/audio/transcriptions  -H "Content-Type: multipart/form-data"  -F model="whisper-1" -F file="@sample.mp4" -F response_format=text
```
