# tiny-openai-whisper-api

OpenAI Whisper API-style local server, runnig on FastAPI. This is for companies behind proxies or security firewalls.

This API will be compatible with [OpenAI Whisper (speech to text) API](https://openai.com/blog/introducing-chatgpt-and-whisper-apis). See also  [Create transcription - API Reference - OpenAI API](https://platform.openai.com/docs/api-reference/audio/create).

Some of code has been copied from [whisper-ui](https://github.com/hayabhay/whisper-ui)

## Setup
This was built & tested on Python 3.10.8, Ubutu20.04/WSL2 but should also work on Python 3.9+.

```bash
sudo apt install ffmpeg
pip install fastapi python-multipart pydantic uvicorn ffmpeg-python openai-whisper
# or pip install -r requirements.txt
```

or 

```bash
docker compose build
```

## Usage

### server
```bash
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

## License

Whisper is licensed under MIT.  whipser-ui excluding Whisper and Streamlit is licensed under MIT.
Everything else by [morioka](https://github.com/morioka) is licensed under MIT.
